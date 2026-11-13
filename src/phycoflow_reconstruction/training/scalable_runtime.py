"""Checkpointable sampled coherence runtime for the existing post-training loop."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from time import perf_counter

import torch

from ..coherence.adaptive_fitting import spectral_filter, validate_bank
from ..coherence.query_runtime import ScalableQueryAdapter
from ..coherence.query_sampling import HierarchicalQuerySampler, SharedQuerySampler
from ..coherence.scale_compute import third_order_scores
from ..coherence.state_compute import group_projection_scores
from ..coherence.state_compute import score_family as state_score
from ..contracts import TermResult
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
        core_only=False,
    ):
        validate_scalable_artifacts(artifacts, expected_provenance)
        if scoring_mode not in {"conditional", "deterministic"}:
            raise ValueError("invalid scalable scoring mode")
        self.scoring_mode = scoring_mode
        self.core_only = bool(core_only)
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
        self.bank = None if self.core_only else artifacts.get("adaptive_bank")
        if self.bank is not None:
            validate_bank(
                self.bank,
                field_order=self.fields,
                support_identity=provenance["support_sha256"],
                source_identity=provenance["source_checkpoint_sha256"],
            )
            if self.bank["population_ids"] != list(provenance["sample_ids"]):
                raise ValueError("adaptive bank TRAIN population differs")
        priorities, units, active = (dict(artifacts[k]) for k in ("priorities", "units", "active"))
        for key, field in (
            ("state.group_projection", "directions"),
            ("scale.third_order", "triples"),
        ):
            if self.bank is not None:
                priorities[key] = 1.0
                units[key] = float(self.bank["block_units"][key])
                active[key] = bool(self.bank[field])
        self.calibration = InternalCalibration(priorities, units, active)
        sampler_class = HierarchicalQuerySampler if self.bank is not None else SharedQuerySampler
        # Uniform fallback until gradient/cost allocation is qualified.
        sampler_options = {"hierarchy_weight": 0.0} if self.bank is not None else {}
        if self.bank and self.bank.get("policy_qualified", False):
            if self.bank.get("policy_mode") != "fixed_source_profile":
                raise ValueError("qualified allocator must deploy its fixed SOURCE profile")
            sampler_options.update(fixed_probabilities=self.bank.get("policy_pair_probabilities"),
                                   profile_sha256=self.bank.get("policy_profile_sha256"))
            if sampler_options["fixed_probabilities"] is None:
                raise ValueError("qualified fixed SOURCE probabilities required")
        self.sampler = sampler_class(self.fields, seed=int(seed), **sampler_options)
        self.probe_sampler = sampler_class(self.fields, seed=int(seed) + 104729, **sampler_options)
        self._gradient_cost_measurements = None
        self.policy_seconds = 0.0
        self.occupied_seconds = 0.0
        # Online changing priorities are not the fixed policy audited offline.
        self.policy_enabled = False
        self._profile_leaves = {}
        self._profile_costs = {}
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
        self._gradient_cost_measurements = None
        self._profile_leaves, self._profile_costs = {}, {}

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

    def _expanded(self, name, prediction, reference, context):
        bank = self.bank
        if bank is None or name not in ("global_distribution", "multiscale_spectrum"):
            return None, None
        key, spec = (
            ("state.group_projection", "directions")
            if name == "global_distribution"
            else ("scale.third_order", "triples")
        )
        if not bank[spec]:
            return key, None
        supplied = tuple(context.get("conditional_draws", ()))
        if self.scoring_mode == "conditional" and not supplied:
            raise ValueError("expanded fair score requires two independent members")
        members = torch.stack(
            (prediction,) if self.scoring_mode == "deterministic" else (prediction, supplied[0])
        )
        family = self.families["global_distribution"]
        left, right = family._endpoint_measures(prediction, reference, context, required=True)
        expected = torch.as_tensor(
            bank["physical_masses"], dtype=prediction.dtype, device=prediction.device
        )
        for mass in (left, right):
            mass = torch.as_tensor(mass, device=prediction.device, dtype=prediction.dtype)
            if mass.ndim == 1:
                mass = mass.unsqueeze(0)
            if mass.shape[-1] != expected.numel() or not torch.allclose(
                mass / mass.sum(-1, keepdim=True), expected.expand_as(mass), rtol=1e-5, atol=1e-7
            ):
                raise ValueError("adaptive bank physical support mismatch")
        if name == "global_distribution":
            leaves = group_projection_scores(members, reference, left, right, bank[spec])
        else:
            scale_family = self.families["multiscale_spectrum"]
            leaves = third_order_scores(
                spectral_filter(scale_family, members),
                spectral_filter(scale_family, reference),
                bank[spec],
                bank["region_masses"],
                bank["triple_scales"],
            )
        weights = torch.as_tensor(
            bank["target_weights"][key], device=leaves.device, dtype=leaves.dtype
        )
        return key, (leaves * weights).sum(-1)

    def _score(self, name, family, prediction, reference, coordinates, context, batch):
        coefficients = self.coefficients
        context = {**context, "scoring_mode": self.scoring_mode, "field_order": self.fields}
        if name == "global_distribution":
            result, blocks = self._state(family, prediction, reference, context, coefficients)
            difficulties = None
        else:
            result, blocks, difficulties = self.adapter.score(
                name,
                prediction,
                reference,
                coordinates=coordinates,
                context=context,
                batch=batch,
                coefficients={**coefficients, "_difficulty_units": self.calibration.units},
            )
        key, value = self._expanded(name, prediction, reference, context)
        if value is not None:
            blocks[key] = value.mean()
            weighted = value * coefficients[key]
            result.component_results[key] = TermResult(value, value.mean())
            result.per_sample_cost = result.per_sample_cost + weighted
            result.scalar_loss = result.scalar_loss + weighted.mean()
            result.diagnostics.setdefault("adaptive_blocks", {})[key] = {
                "bank_sha256": self.bank["sha256"],
                "coefficient": coefficients[key],
            }
        return result, blocks, difficulties

    def observe_gradient_cost(self, *, self_measurements=None, pair_measurements=None):
        """Stage detached lagged gradient-second-moment/cost measurements."""
        if self.batch is None or self.bank is None:
            raise ValueError("adaptive update must be pending")
        self._gradient_cost_measurements = (self_measurements, pair_measurements)

    def score_family(self, name, family, prediction, reference, *, coordinates, context):
        profiling = (
            self.bank is not None
            and self.policy_enabled
            and (self.step + 1) % (10 * self.steps_per_epoch) == 0
            and name in ("multiscale_spectrum", "topology")
        )
        if profiling and prediction.device.type == "cuda":
            torch.cuda.synchronize(prediction.device)
        began = perf_counter()
        result, blocks, difficulties = self._score(
            name,
            family,
            prediction,
            reference,
            coordinates,
            {**context, "_adaptive_capture_leaves": profiling},
            self.batch,
        )
        if profiling:
            if prediction.device.type == "cuda":
                torch.cuda.synchronize(prediction.device)
            self._profile_costs[name] = perf_counter() - began
            self._profile_leaves[name] = difficulties
            difficulties = {
                kind: {i: float(value.detach().mean()) for i, value in rows.items()}
                for kind, rows in difficulties.items()
            }
        self.blocks.update(blocks)
        if difficulties is not None:
            self.difficulties[name] = difficulties
        return result

    def record_occupied_time(self, seconds):
        if not 0 <= seconds < float("inf"):
            raise ValueError("occupied time must be finite and nonnegative")
        if not self.policy_enabled:
            return
        self.occupied_seconds += float(seconds)

    def _measure_policy(self, prediction, context):
        if not self.policy_enabled or set(self._profile_leaves) != {
            "multiscale_spectrum",
            "topology",
        }:
            return
        inputs = (prediction, *tuple(context.get("conditional_draws", ()))[:1])
        if any(not x.requires_grad for x in inputs):
            return
        if prediction.device.type == "cuda":
            torch.cuda.synchronize(prediction.device)
        started = perf_counter()
        measured = {}
        # Keep unary sampling uniform until its own incremental costs are measured.
        for kind in ("pair",):
            a = self._profile_leaves["multiscale_spectrum"][kind]
            b = self._profile_leaves["topology"][kind]
            if set(a) != set(b):
                raise ValueError("gradient sketch query mismatch")
            # Offline synchronized incremental costs exclude shared anchors.
            pair_costs = self.bank.get("policy_incremental_pair_costs", ())
            if len(pair_costs) != len(self.fields) * (len(self.fields) - 1) // 2:
                raise ValueError("qualified policy needs all measured pair costs")
            measured[kind] = {}
            for i in a:
                gradients = torch.autograd.grad(
                    (a[i] + b[i]).mean(), inputs, retain_graph=True, allow_unused=True
                )
                parts = [
                    g.detach().flatten()[:: max(1, g.numel() // 128)][:128]
                    for g in gradients
                    if g is not None
                ]
                norm = float(torch.cat(parts).double().square().sum()) if parts else 0.0
                cost = pair_costs[i]
                measured[kind][i] = (norm, cost)
        if prediction.device.type == "cuda":
            torch.cuda.synchronize(prediction.device)
        self.policy_seconds += perf_counter() - started
        self.observe_gradient_cost(self_measurements=None, pair_measurements=measured["pair"])
        self._profile_leaves = {}

    def prepare_probe(self, model, prediction, reference, coordinates, context):
        self._measure_policy(prediction, context)
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
            sampler_class = (
                HierarchicalQuerySampler if self.bank is not None else SharedQuerySampler
            )
            self.sampler = sampler_class.from_state_dict(self._before[0])
            self.probe_sampler = sampler_class.from_state_dict(self._before[1])
            self.batch = None
            self.pending_probe = None
            return
        epoch = self.step // self.steps_per_epoch + 1
        if self.occupied_seconds and self.policy_seconds > 0.05 * self.occupied_seconds:
            self.policy_enabled = False
            self.sampler.hierarchy_weight = 0.0
            self._gradient_cost_measurements = None
            for kind in self.sampler.distribution:
                self.sampler.distribution[kind] = {}
                self.sampler.cache[kind].clear()
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
        if self.bank is not None:
            if self._gradient_cost_measurements is None:
                self.sampler.finish(exposure_sink=exposure.append)
            else:
                self.sampler.finish_measurements(
                    self_measurements=self._gradient_cost_measurements[0],
                    pair_measurements=self._gradient_cost_measurements[1],
                    exposure_sink=exposure.append,
                )
        else:
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
            "core_only": self.core_only,
            "adaptive_bank_sha256": self.bank["sha256"] if self.bank is not None else None,
            "policy_seconds": self.policy_seconds,
            "occupied_seconds": self.occupied_seconds,
            "policy_enabled": self.policy_enabled,
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
            or state.get("core_only", False) != self.core_only
            or state.get("adaptive_bank_sha256")
            != (self.bank["sha256"] if self.bank is not None else None)
        ):
            raise ValueError("runtime checkpoint identity mismatch")
        step = state["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("invalid runtime checkpoint cursor")
        sampler_class = HierarchicalQuerySampler if self.bank is not None else SharedQuerySampler
        sampler = sampler_class.from_state_dict(state["sampler"])
        probe_sampler = sampler_class.from_state_dict(state["probe_sampler"])
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
            if self.bank is not None:
                expected_hierarchy = initial.hierarchy_weight
                if actual is sampler and not state.get("policy_enabled", True):
                    expected_hierarchy = 0.0
                if actual.hierarchy_weight != expected_hierarchy:
                    raise ValueError("runtime hierarchy settings changed")
                if (actual.fixed_probabilities != initial.fixed_probabilities
                    or actual.profile_sha256 != initial.profile_sha256):
                    raise ValueError("runtime fixed SOURCE profile changed")
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
        self.policy_seconds = float(state.get("policy_seconds", 0.0))
        self.occupied_seconds = float(state.get("occupied_seconds", 0.0))
        if state.get("policy_enabled", False):
            raise ValueError("online changing policy has no deployment qualification")
        self.policy_enabled = False
        if min(self.policy_seconds, self.occupied_seconds) < 0:
            raise ValueError("invalid policy timing state")
        self.sampler, self.probe_sampler, self.calibration, self.step = (
            sampler,
            probe_sampler,
            calibration,
            step,
        )
