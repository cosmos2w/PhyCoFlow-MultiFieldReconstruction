"""Checkpointable sampled coherence runtime for the existing post-training loop."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch

from ..coherence.query_runtime import ScalableQueryAdapter
from ..coherence.query_sampling import SharedQuerySampler
from ..coherence.state_compute import score_family as state_score
from .internal_calibration import PROBES, InternalCalibration, gradient_norm
from .scalable_fitting import validate_scalable_artifacts

VERSION = "scalable_coherence_runtime_v1"


class ScalableRuntime:
    def __init__(
        self,
        artifacts,
        families,
        *,
        steps_per_epoch,
        expected_provenance,
        seed=0,
        scoring_mode="conditional",
    ):
        validate_scalable_artifacts(artifacts, expected_provenance)
        if scoring_mode not in {"conditional", "deterministic"}:
            raise ValueError("invalid scalable scoring mode")
        self.scoring_mode = scoring_mode
        self.artifacts = artifacts
        provenance = artifacts["provenance"]
        required = (
            "split",
            "field_order",
            "sample_ids",
            "source_checkpoint_sha256",
            "normalizer_sha256",
            "support_sha256",
            "config_sha256",
            "definition_version",
        )
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("split") != "train"
            or any(k not in provenance for k in required)
        ):
            raise ValueError("scalable artifacts require complete TRAIN provenance")
        if any(provenance[k] != expected_provenance.get(k) for k in required):
            raise ValueError("scalable artifact provenance does not match the live run")
        if not provenance["sample_ids"] or len(set(provenance["sample_ids"])) != len(
            provenance["sample_ids"]
        ):
            raise ValueError("scalable TRAIN sample IDs must be nonempty and unique")
        self.fields = tuple(provenance["field_order"])
        if tuple(artifacts["state_bank"]["field_order"]) != self.fields:
            raise ValueError("State bank field order mismatch")
        if any(
            tuple(families[k].field_names) != self.fields
            for k in ("global_distribution", "multiscale_spectrum", "topology")
        ):
            raise ValueError("coherence family field order mismatch")
        shape_provenance = artifacts["shape_artifact"].get("provenance", {})
        if (
            shape_provenance.get("split") != "train"
            or shape_provenance.get("source_checkpoint_sha256")
            != provenance["source_checkpoint_sha256"]
        ):
            raise ValueError("Shape artifact TRAIN/SOURCE identity mismatch")
        self.families = families
        self.adapter = ScalableQueryAdapter(families, artifacts["shape_artifact"])
        self.calibration = InternalCalibration(
            artifacts["priorities"], artifacts["units"], artifacts["active"]
        )
        self.sampler = SharedQuerySampler(self.fields, seed=int(seed))
        self.probe_sampler = SharedQuerySampler(self.fields, seed=int(seed) + 104729)
        if isinstance(steps_per_epoch, bool) or not isinstance(steps_per_epoch, int):
            raise TypeError("steps_per_epoch must be a positive integer")
        self.steps_per_epoch = steps_per_epoch
        if self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive")
        self.step = 0
        self.batch = None
        self.blocks = {}
        self.difficulties = {}
        self.pending_probe = None
        self.probe_identity = None
        self._before = None

    @property
    def coefficients(self):
        return self.calibration.coefficients or {key: 1.0 for key in self.calibration.active}

    def begin_step(self, step):
        if self.batch is not None or step != self.step:
            raise ValueError("runtime cursor mismatch or pending update")
        self._before = (self.sampler.state_dict(), self.probe_sampler.state_dict())
        self.batch = self.sampler.draw()
        self.blocks, self.difficulties, self.pending_probe = {}, {}, None

    def _state(self, family, prediction, reference, context, coefficients):
        supplied = tuple(context.get("conditional_draws", ()))
        needed = 0 if self.scoring_mode == "deterministic" else int(family.conditional_draws) - 1
        if len(supplied) < needed:
            raise ValueError("conditional State requires independent draws")
        members = (prediction, *supplied[:needed])
        converted = [family._in_declared_units(x, reference)[0] for x in members]
        target = family._in_declared_units(prediction, reference)[1]
        left, right = family._endpoint_measures(prediction, reference, context, required=True)
        component = family.components_by_key["self_marginal_w2"]
        result, blocks = state_score(
            torch.stack(converted),
            target,
            left,
            right,
            self.artifacts["state_bank"],
            channel_weights=component.channel_weights,
            marginal_weight=family.component_weights["self_marginal_w2"],
            mixed_weight=family.component_weights.get("cross_joint_topk_swd", 0.0),
            marginal_coefficient=coefficients["state.marginal"],
            mixed_coefficient=coefficients["state.mixed"],
            mode="paired" if self.scoring_mode == "deterministic" else "conditional",
        )
        return result, blocks

    def _score(self, name, family, prediction, reference, coordinates, context, batch):
        coefficients = self.coefficients
        context = {**context, "scoring_mode": self.scoring_mode, "field_order": self.fields}
        if name == "global_distribution":
            result, blocks = self._state(family, prediction, reference, context, coefficients)
            return result, blocks, None
        return self.adapter.score(
            name,
            prediction,
            reference,
            coordinates=coordinates,
            context=context,
            batch=batch,
            coefficients={**coefficients, "_difficulty_units": self.calibration.units},
        )

    def score_family(self, name, family, prediction, reference, *, coordinates, context):
        result, blocks, difficulties = self._score(
            name, family, prediction, reference, coordinates, context, self.batch
        )
        self.blocks.update(blocks)
        if difficulties is not None:
            self.difficulties[name] = difficulties
        return result

    def prepare_probe(self, model, prediction, reference, coordinates, context):
        epoch = self.step // self.steps_per_epoch + 1
        if (
            epoch not in PROBES
            or self.step % self.steps_per_epoch
            or self.calibration.coefficients is not None
        ):
            return
        batch = self.probe_sampler.draw()
        blocks = {}
        for name, family in self.families.items():
            _, values, _ = self._score(
                name, family, prediction, reference, coordinates, context, batch
            )
            blocks.update(values)
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        query_indices = context.get("query_indices")
        if not isinstance(query_indices, torch.Tensor) or not context.get("sample_ids"):
            raise ValueError("calibration probe requires sample and query identity")
        self.probe_identity = {
            "sample_ids": list(context["sample_ids"]),
            "query_indices_sha256": hashlib.sha256(
                query_indices.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
        }
        self.pending_probe = {
            key: gradient_norm(blocks[key] / self.calibration.units[key], parameters)
            for key, active in self.calibration.active.items()
            if active
        }
        self.probe_sampler.finish()

    def finish_step(self, accepted):
        if self.batch is None:
            raise ValueError("no pending runtime update")
        if not accepted:
            self.sampler = SharedQuerySampler.from_state_dict(self._before[0])
            self.probe_sampler = SharedQuerySampler.from_state_dict(self._before[1])
            self.batch = None
            self.pending_probe = None
            return
        epoch = self.step // self.steps_per_epoch + 1
        if self.pending_probe is not None:
            self.calibration.observe(epoch, self.pending_probe)
        values = {}
        for kind in ("self", "pair"):
            scale = self.difficulties["multiscale_spectrum"][kind]
            shape = self.difficulties["topology"][kind]
            if set(scale) != set(shape):
                raise ValueError("Scale and Shape exposure differs")
            values[kind] = {i: (scale[i], shape[i]) for i in scale}
        exposure = []
        self.sampler.finish(
            self_difficulties=values["self"],
            pair_difficulties=values["pair"],
            exposure_sink=exposure.append,
        )
        probe = self.pending_probe
        self.step += 1
        if self.step % self.steps_per_epoch == 0:
            self.sampler.refresh(epoch)
            if epoch == 10:
                self.calibration.freeze(epoch)
        self.batch, self.pending_probe, self._before = None, None, None
        return {
            "step": self.step,
            "epoch": epoch,
            "exposure": exposure,
            "probe_norms": probe,
            "probe_identity": self.probe_identity if probe is not None else None,
            "calibration_frozen": self.calibration.coefficients is not None,
            "coefficients": dict(self.coefficients),
        }

    def state_dict(self):
        if self.batch is not None:
            raise ValueError("runtime state requires an update boundary")
        return {
            "version": VERSION,
            "provenance": dict(self.artifacts["provenance"]),
            "artifact_sha256": self.artifacts["sha256"],
            "step": self.step,
            "steps_per_epoch": self.steps_per_epoch,
            "scoring_mode": self.scoring_mode,
            "sampler": self.sampler.state_dict(),
            "probe_sampler": self.probe_sampler.state_dict(),
            "calibration": self.calibration.state_dict(),
        }

    def load_state_dict(self, state):
        if (
            state["version"] != VERSION
            or state["provenance"] != self.artifacts["provenance"]
            or state.get("artifact_sha256") != self.artifacts["sha256"]
            or state["steps_per_epoch"] != self.steps_per_epoch
            or state.get("scoring_mode") != self.scoring_mode
        ):
            raise ValueError("runtime checkpoint identity mismatch")
        step = state["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("invalid runtime checkpoint cursor")
        sampler = SharedQuerySampler.from_state_dict(state["sampler"])
        probe_sampler = SharedQuerySampler.from_state_dict(state["probe_sampler"])
        calibration = InternalCalibration.from_state_dict(state["calibration"])
        for actual, initial in ((sampler, self.sampler), (probe_sampler, self.probe_sampler)):
            keys = (
                "fields",
                "mutual_draws",
                "epsilon",
                "decay",
                "capacity",
                "refresh_epochs",
                "inner_seed",
            )
            if any(getattr(actual, k) != getattr(initial, k) for k in keys):
                raise ValueError("runtime sampler settings or stream identity changed")
            if actual.pending is not None:
                raise ValueError("checkpoint contains an unfinished update")
        expected_probes = [e for e in PROBES if step >= (e - 1) * self.steps_per_epoch + 1]
        if (
            [r["epoch"] for r in calibration.records] != expected_probes
            or sampler.counter != step
            or probe_sampler.counter != len(expected_probes)
            or (calibration.coefficients is not None) != (step >= 10 * self.steps_per_epoch)
        ):
            raise ValueError("runtime sampler or calibration cursor mismatch")
        if (
            calibration.priorities != self.calibration.priorities
            or calibration.units != self.calibration.units
            or calibration.active != self.calibration.active
            or calibration.floor != self.calibration.floor
        ):
            raise ValueError("runtime calibration settings differ from TRAIN artifacts")
        self.sampler, self.probe_sampler, self.calibration, self.step = (
            sampler,
            probe_sampler,
            calibration,
            step,
        )
