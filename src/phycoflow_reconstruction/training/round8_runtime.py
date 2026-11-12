"""Round 8 trainer lifecycle: coupled member risk and frozen internal scores.

The campaign supplies TRAIN-fitted artifacts. This runtime never fits policy,
directions or units from assessment data and never owns a model optimizer.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import hashlib
import json
import math
import struct

import torch

from ..coherence.round8_queries import SharedQuerySampler
from ..coherence.round8_state import score_family as state_score
from .round8_calibration import InternalCalibration, PROBES, gradient_norm
from .round8_policy import physical_risk

VERSION = "round8_training_runtime_v1"


class Round8Runtime:
    def __init__(self, artifacts, *, mode="warmup"):
        if mode not in {"warmup", "control", "candidate"}:
            raise ValueError("unknown Round 8 mode")
        if not isinstance(artifacts, Mapping):
            artifacts = torch.load(artifacts, map_location="cpu", weights_only=True)
        self.artifacts = dict(artifacts)
        self.mode = mode
        self.manages_queries = mode == "candidate"
        self.calibration = InternalCalibration(
            self.artifacts["priorities"], self.artifacts["units"], self.artifacts["active"])
        self.coefficients = {key: 1.0 for key in self.artifacts["active"]}
        self.sampler = None
        self.blocks = {}
        self.difficulties = {}
        self.step = 0
        self.epoch = 1

    def initialize(self, *, model, dataset, families, config, run_dir, start_step, saved_state):
        from .round8_query_runtime import Round8QueryAdapter

        provenance = self.artifacts["provenance"]
        def valid_digest(value):
            return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
        if provenance.get("split") != "train" or not valid_digest(provenance.get("manifest_sha256")):
            raise ValueError("Round 8 artifacts require frozen TRAIN provenance")
        if tuple(provenance["field_order"]) != tuple(dataset.field_names):
            raise ValueError("Round 8 field order differs from fitted artifacts")
        if not hasattr(dataset, "physical_weights"):
            raise ValueError("Round 8 requires declared native physical masses")
        self.native_weights = dataset.physical_weights.detach().double()
        if not torch.isfinite(self.native_weights).all() or (self.native_weights < 0).any() or self.native_weights.sum() <= 0:
            raise ValueError("invalid native physical masses")
        self.native_weights = self.native_weights / self.native_weights.sum()
        self.fields = tuple(dataset.field_names)
        probe_ids = tuple(self.artifacts.get("probe_sample_ids", ()))
        probe_indices = self.artifacts.get("probe_query_indices")
        if not probe_ids or len(set(probe_ids)) != len(probe_ids) or not isinstance(probe_indices, torch.Tensor) or probe_indices.ndim not in (1, 2) or probe_indices.numel() == 0:
            raise ValueError("fixed unique TRAIN probe IDs and query indices required")
        self.probe_sample_ids = probe_ids
        probe_indices = probe_indices.detach().cpu().clone()
        if probe_indices.ndim == 1:
            probe_indices = probe_indices[None].expand(len(probe_ids), -1).clone()
        if (probe_indices.shape[0] != len(probe_ids)
                or not torch.equal(probe_indices, probe_indices[0:1].expand_as(probe_indices))):
            raise ValueError("calibration probes require shared fixed spatial indices")
        self.probe_query_indices = probe_indices
        self.steps_per_epoch = int(config["optimization"]["steps_per_epoch"])
        if self.steps_per_epoch < 1: raise ValueError("steps per epoch must be positive")
        if tuple(self.artifacts["state_bank"]["field_order"]) != self.fields:
            raise ValueError("State artifact field order differs")
        if any(tuple(families[k].field_names) != self.fields for k in ("multiscale_spectrum", "topology", "global_distribution")):
            raise ValueError("all family field orders must match sampler")
        shape_provenance = self.artifacts["shape_artifact"].get("provenance", {})
        if shape_provenance.get("split") != "train": raise ValueError("Shape must be TRAIN calibrated")
        if provenance.get("source_checkpoint_sha256") != shape_provenance.get("source_checkpoint_sha256") or not valid_digest(provenance.get("source_checkpoint_sha256")):
            raise ValueError("Shape and runtime SOURCE identity differ")
        if len(self.artifacts["shape_artifact"].get("field_mean", ())) != len(self.fields):
            raise ValueError("Shape transform and field population differ")
        self.run_dir = Path(run_dir)
        self.device = next(model.parameters()).device
        self.probe_batch = None
        if self.mode == "warmup":
            from ..data.sensor_protocols import SensorProtocol, build_observation_batch
            from ..data.physical_measure import attach_batch_measure
            from .common import sensor_protocol_from_config

            indices = self.artifacts["probe_dataset_indices"]
            if len(indices) != len(probe_ids) or len(indices) != int(config["optimization"]["batch_size"]):
                raise ValueError("fixed calibration batch must retain the ordinary batch size")
            protocol = sensor_protocol_from_config(config)
            protocol = SensorProtocol(**{**protocol.to_dict(), "seed": int(self.artifacts.get("probe_seed", 800047))})
            query = self.probe_query_indices[0] if self.probe_query_indices.ndim == 2 else self.probe_query_indices
            probe = build_observation_batch(
                [dataset[int(i)] for i in indices], protocol,
                query_points=query.numel(), query_indices=query)
            self.probe_batch = attach_batch_measure(dataset, probe).to(self.device)
            if (tuple(probe.sample_ids) != probe_ids
                    or not torch.equal(probe.metadata["query_indices"], self.probe_query_indices)):
                raise ValueError("constructed probe batch differs from frozen artifact")
        self.families = families
        self.adapter = Round8QueryAdapter(families, self.artifacts["shape_artifact"])
        self.sampler = SharedQuerySampler(
            self.fields, seed=int(self.artifacts.get("probe_seed", 800047) if self.mode == "warmup" else self.artifacts.get("sampler_seed", 800048)))
        if saved_state is not None:
            self.load_state_dict(saved_state)
        elif start_step:
            raise ValueError("Round 8 resume requires its complete runtime state")
        if self.mode == "candidate" and self.calibration.coefficients is None:
            raise ValueError("candidate requires completed epoch-10 calibration")
        if self.mode == "candidate":
            self.coefficients = dict(self.calibration.coefficients)
        if self.step != int(start_step):
            raise ValueError("runtime checkpoint cursor differs from training checkpoint")
        if self.sampler.pending is not None:
            raise ValueError("trainer checkpoint must be at an update boundary")
        if saved_state is not None:
            from .round8_log_recovery import restore_journals
            restore_journals(self.run_dir, self.calibration, self.sampler, self.mode)

    def begin_step(self, *, step, epoch):
        if int(step) != self.step or int(epoch) != self.step // self.steps_per_epoch + 1:
            raise ValueError("Round 8 step/epoch cursor mismatch")
        if self.mode == "warmup" and int(epoch) > 10:
            raise ValueError("warmup ends at completed epoch 10")
        self.epoch = int(epoch)
        self.blocks, self.raw_blocks, self.difficulties, self.family_scores = {}, {}, {}, {}
        self.batch = self.sampler.draw() if self.manages_queries else None

    def _probe_due(self):
        return self.mode == "warmup" and self.epoch in PROBES and self.step % self.steps_per_epoch == 0

    def training_batch(self, ordinary_batch):
        return self.probe_batch if self._probe_due() else ordinary_batch

    def rollout_seed(self, ordinary_seed):
        return int(self.artifacts.get("probe_seed", 800047)) + 1_000_003 if self._probe_due() else ordinary_seed

    def _state_with_bank(self, family, prediction, reference, context, coefficients, bank):
        from ..coherence.base import require_field_tensor
        require_field_tensor("generated", prediction)
        require_field_tensor("reference", reference)
        if prediction.shape != reference.shape or not isinstance(context, Mapping):
            raise ValueError("conditional State endpoints and context must match")
        metadata = context.get("measure_metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("weighted State measure_metadata must be a mapping")
        claimed = metadata.get("native_measure_sha256")
        if family._measure_artifact is not None and claimed is not None and claimed != family._measure_artifact["sha256"]:
            raise ValueError("runtime State measure differs from the frozen native measure")
        supplied = tuple(context.get("conditional_draws", ()))
        required = family.conditional_draws - 1
        if len(supplied) < required:
            raise ValueError("conditional State requires all configured draws")
        draws = supplied if context.get("score_all_conditional_draws", False) else supplied[:required]
        members = (prediction, *draws)
        if any(not isinstance(x, torch.Tensor) or x.shape != prediction.shape or x.dtype != prediction.dtype or x.device != prediction.device for x in members):
            raise ValueError("conditional members must match generated shape, device and dtype")
        converted = [family._in_declared_units(x, reference)[0] for x in members]
        converted_reference = family._in_declared_units(prediction, reference)[1]
        ids = list(family.field_ids)
        left, right = family._endpoint_measures(prediction, reference, context, required=True)
        self_component = family.components_by_key["self_marginal_w2"]
        result, blocks = state_score(
            torch.stack([x[..., ids] for x in converted]), converted_reference[..., ids],
            left, right, bank,
            channel_weights=self_component.channel_weights,
            marginal_weight=family.component_weights["self_marginal_w2"],
            mixed_weight=family.component_weights.get("cross_joint_topk_swd", 0.),
            marginal_coefficient=coefficients["state.marginal"],
            mixed_coefficient=coefficients["state.mixed"],
        )
        return result, blocks

    def _state(self, family, prediction, reference, context, coefficients):
        return self._state_with_bank(family, prediction, reference, context, coefficients, self.artifacts["state_bank"])

    def _state_control(self, family, prediction, reference, context):
        from ..coherence.round8_state import VERSION as state_version
        from ..contracts import TermResult
        component = (family.components_by_key["cross_joint_topk_swd"]
                     if "cross_joint_topk_swd" in family.components_by_key else None)
        if component is None:
            directions = torch.empty((0, len(family.field_ids)), dtype=prediction.dtype, device=prediction.device)
        else:
            directions = component.directions.detach()
        checksum = hashlib.sha256(directions.cpu().contiguous().numpy().tobytes()).hexdigest()
        bank = {"version": state_version, "directions": directions,
                "sha256": checksum, "mixed_active": bool(len(directions))}
        result, blocks = self._state_with_bank(
            family, prediction, reference, context,
            {"state.marginal": 1., "state.mixed": 1.}, bank)
        result.component_results = {"global_distribution.conditional_quantile_fair":
                                    TermResult(result.per_sample_cost, result.scalar_loss)}
        result.diagnostics.update({"score_definition": family.definition,
                                   "computation_identity": "shared_exact_weighted_quantile_cache_v1",
                                   "retained_direction_count": len(directions),
                                   "directions_sha256": checksum})
        return result, blocks

    def _diagnostic_due(self):
        return self.mode in {"control", "candidate"} and self.epoch in (100, 500, 1000) and self.step % self.steps_per_epoch == 0

    def score_family(self, name, family, prediction, reference, *, coordinates, context):
        if not self.manages_queries:
            if name == "global_distribution":
                result, state_blocks = self._state_control(family, prediction, reference, context)
            else:
                result = family(prediction, reference, coordinates=coordinates, context=context)
            if self._diagnostic_due():
                self.diagnostic_identity = {"sample_ids": list(context.get("sample_ids", ())), "field_order": list(self.fields)}
                if name == "global_distribution":
                    self.blocks.update(state_blocks)
                elif name == "multiscale_spectrum":
                    from ..coherence.families.multiscale_relation import SELF_PATH, RHO_PATH
                    from ..coherence.families.multiscale_spectrum.objectives import FAMILY_NAME, SIGNED_BAND_PATH, MEAN_JOINT_ENERGY_PATH
                    for key, path in (("signed", f"{FAMILY_NAME}.{SIGNED_BAND_PATH}"), ("joint", f"{FAMILY_NAME}.{MEAN_JOINT_ENERGY_PATH}"), ("log_power_self", SELF_PATH), ("rho", RHO_PATH)):
                        term = result.component_results.get(path)
                        self.blocks[f"scale.{key}"] = None if term is None else float(result.diagnostics["components"][path]["weight"]) * term.scalar_loss
                elif name == "topology":
                    for key, component in (("self", "self.persistence"), ("mutual", "mutual.persistence")):
                        term = result.component_results.get(f"topology.{component}")
                        self.blocks[f"shape.{key}"] = None if term is None else family.spatial_objective.weights[component] * term.scalar_loss
            return result
        if self._diagnostic_due():
            self.diagnostic_identity = {"sample_ids": list(context.get("sample_ids", ())), "field_order": list(self.fields)}
        if name == "global_distribution":
            result, blocks = self._state(family, prediction, reference, context, self.coefficients)
            self.blocks.update(blocks)
            with torch.no_grad():
                raw, raw_blocks = self._state(family, prediction, reference, context, {k: 1.0 for k in self.coefficients})
            self.raw_blocks.update(raw_blocks)
            self.family_scores["state"] = {"raw": float(raw.scalar_loss.detach()), "calibrated": float(result.scalar_loss.detach())}
            return result
        result, blocks, difficulties = self.adapter.score(
            name, prediction, reference, coordinates=coordinates, context=context,
            batch=self.batch, coefficients={**self.coefficients, "_difficulty_units": self.calibration.units})
        self.blocks.update(blocks)
        self.raw_blocks.update(blocks)
        self.family_scores["scale" if name == "multiscale_spectrum" else "shape"] = {"raw": sum(float(v.detach()) for v in blocks.values()), "calibrated": float(result.scalar_loss.detach())}
        self.difficulties[name] = difficulties
        return result

    def diagnostic_context(self, family_losses, *, family_scales):
        """Live semantic blocks for the three predeclared diagnostic updates."""
        if not self._diagnostic_due(): return None
        from .round8_calibration import BLOCKS
        expected = {f"{family}.{name}" for family, names in BLOCKS.items() for name in names}
        if set(self.blocks) != expected:
            raise ValueError("Round 8 diagnostic requires all eight semantic blocks")
        families = {"state": "global_distribution", "scale": "multiscale_spectrum", "shape": "topology"}
        if set(family_losses) != set(families.values()) or set(family_scales) != set(families.values()):
            raise ValueError("Round 8 diagnostics require the three family scales")
        factors = {key: float(family_scales[name]) * float(getattr(self.families[name], "family_weight", 1.0)) for key, name in families.items()}
        if any(not math.isfinite(v) or v <= 0 for v in factors.values()):
            raise ValueError("diagnostic family factors must be positive finite")
        blocks = {}
        for key, value in self.blocks.items():
            if value is not None:
                # State channel tensors already include coefficients inside its
                # common distance; Scale/Shape are additive raw weighted leaves.
                multiplier = self.coefficients[key] if self.manages_queries and not key.startswith("state.") else 1.0
                blocks[key] = value * factors[key.split(".")[0]] * multiplier
            elif self.calibration.active[key]:
                raise ValueError(f"active diagnostic block unavailable: {key}")
        from ..coherence.round8_queries import VERSION as query_version, SHAPE_SAMPLING_VERSION
        identity = {**getattr(self, "diagnostic_identity", {}), "query_protocol": query_version,
                    "shape_sampling": SHAPE_SAMPLING_VERSION if self.manages_queries else "retained_sampler", "mode": self.mode, "step": self.step,
                    "epoch": self.epoch, "family_factors": factors, "train_units": dict(self.calibration.units),
                    "coefficients": dict(self.coefficients), "artifact_identity": self._artifact_identity(),
                    "shape_score": "pooled" if self.manages_queries else "retained_per_query",
                    "state_directions": "round8_frozen" if self.manages_queries else "retained_full_bank",
                    "state_computation": "shared_exact_weighted_quantile_cache_v1",
                    "inner_seed": self.sampler.inner_seed if self.manages_queries else None,
                    "inner_rng_state_sha256": (hashlib.sha256(repr(self.sampler.inner_rng.getstate()).encode()).hexdigest() if self.manages_queries else None),
                    "queries": None if not self.manages_queries else {"batch": self.batch.number, "self": [vars(x).copy() for x in self.batch.self_queries], "pair": [vars(x).copy() for x in self.batch.pair_queries]}}
        return {"schema": "round8_actual_step_v1", "blocks": blocks, "identity": identity}

    def after_objective(self, *, model, endpoint, result):
        if self.mode != "warmup" or self.epoch not in PROBES:
            return
        if any(row["epoch"] == self.epoch for row in self.calibration.records):
            return
        # The predeclared first ordinary batch of each probe epoch is reused;
        # no new model forward, training RNG change or optimizer update occurs.
        if self.step % self.steps_per_epoch:
            raise ValueError("missing predeclared first-batch calibration probe")
        probe_batch = endpoint["batch"]
        query_indices = probe_batch.metadata.get("query_indices")
        if tuple(probe_batch.sample_ids) != self.probe_sample_ids or not isinstance(query_indices, torch.Tensor) or not torch.equal(query_indices.detach().cpu(), self.probe_query_indices):
            raise ValueError("calibration probe batch differs from frozen IDs or spatial support")
        batch = self.sampler.draw()
        members, reference = endpoint["members"], endpoint["target"]
        context = {**endpoint["measure"], "conditional_draws": members[1:]}
        _, blocks = self._state(
            self.families["global_distribution"], members[0], reference, context,
            self.coefficients)
        for name in ("multiscale_spectrum", "topology"):
            _, leaves, _ = self.adapter.score(
                name, members[0], reference, coordinates=endpoint["batch"].query_coords,
                context=context, batch=batch, coefficients=self.coefficients)
            blocks.update(leaves)
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        norms = {}
        for key, active in self.calibration.active.items():
            if active:
                loss = blocks[key] / self.calibration.units[key]
                norms[key] = gradient_norm(loss, parameters)
        self.calibration.observe(self.epoch, norms)
        from ..coherence.round8_queries import VERSION as query_version, SHAPE_SAMPLING_VERSION
        record = {**self.calibration.records[-1], "query_protocol": query_version,
                  "shape_sampling": SHAPE_SAMPLING_VERSION, "inner_seed": self.sampler.inner_seed,
                  "inner_rng_state_sha256": hashlib.sha256(repr(self.sampler.inner_rng.getstate()).encode()).hexdigest(),
                  "query_batch": {"number": batch.number,
                                  "self": [vars(draw).copy() for draw in batch.self_queries],
                                  "pair": [vars(draw).copy() for draw in batch.pair_queries]}}
        self.sampler.finish()
        self._append("round8_calibration.jsonl", record)

    def risks(self, endpoint):
        # Model units equal physical squared error divided by the frozen s_a²;
        # this makes the controller floor dimensionless without changing bounds.
        members = torch.stack(endpoint["members"], dim=1).permute(0, 1, 3, 2)
        target = endpoint["target"].transpose(1, 2)
        ids = endpoint["batch"].metadata["query_indices"].to(self.device)
        weights = self.native_weights.to(self.device)[ids]
        if not bool(endpoint["mask"].all()):
            raise ValueError("Round 8 fixed spatial proxy requires compact valid sites")
        return physical_risk(
            members, target, weights, design="without_replacement",
            probability=members.shape[-1] / self.native_weights.numel())

    def after_step(self, row):
        if not row.get("update_accepted", True):
            raise FloatingPointError("Round 8 cannot advance a rejected update")
        if self.manages_queries:
            scale = self.difficulties["multiscale_spectrum"]
            shape = self.difficulties["topology"]
            values = {}
            for kind in ("self", "pair"):
                if set(scale[kind]) != set(shape[kind]):
                    raise ValueError("Scale/Shape query exposure differs")
                values[kind] = {i: (scale[kind][i], shape[kind][i]) for i in scale[kind]}
            self.sampler.finish(
                self_difficulties=values["self"], pair_difficulties=values["pair"],
                exposure_sink=lambda item: self._append("round8_query_exposure.jsonl", item))
        expected_step = self.step + 1
        if int(row["step"]) != expected_step:
            raise ValueError("post-update cursor mismatch")
        self.step = expected_step
        if self.step % self.steps_per_epoch == 0:
            self.sampler.refresh(self.epoch)
            if self.mode == "warmup" and self.epoch == 10:
                self.calibration.freeze(10)
                self._append("round8_calibration.jsonl", {
                    "event": "frozen", "epoch": 10,
                    "coefficients": self.calibration.coefficients,
                    "priorities": self.calibration.priorities, "units": self.calibration.units})
        row["round8"] = {"version": VERSION, "mode": self.mode,
                          "calibration_frozen": self.calibration.coefficients is not None,
                          "scores": self.family_scores,
                          "blocks_raw": {k: float(v.detach()) for k, v in self.raw_blocks.items() if v is not None},
                          "blocks_calibrated": {k: float(v.detach()) * (1.0 if k.startswith("state.") else self.coefficients[k]) for k, v in self.blocks.items() if v is not None},
                          "state_blocks_semantics": "chain-rule gradient contributions, not additive scores",
                          "coefficients": dict(self.coefficients)}

    def _append(self, filename, value):
        with (self.run_dir / "logs" / filename).open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")

    def _artifact_identity(self):
        """Content identity independent of torch.save archive IDs and map order."""
        digest = hashlib.sha256()
        def add(value):
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(b'T' + str(tensor.dtype).encode() + repr(tuple(tensor.shape)).encode() + b':')
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(value, Mapping):
                digest.update(b'{')
                for key in sorted(value, key=lambda x: repr(x)):
                    add(key); add(value[key])
                digest.update(b'}')
            elif isinstance(value, (list, tuple)):
                digest.update(b'[')
                for item in value: add(item)
                digest.update(b']')
            elif value is None: digest.update(b'N')
            elif isinstance(value, bool): digest.update(b'B1' if value else b'B0')
            elif isinstance(value, int): digest.update(b'I' + str(value).encode() + b';')
            elif isinstance(value, float):
                if not math.isfinite(value): raise ValueError("nonfinite artifact value")
                digest.update(b'F' + struct.pack('!d', value))
            elif isinstance(value, str):
                data = value.encode(); digest.update(b'S' + str(len(data)).encode() + b':' + data)
            else: raise TypeError(f"unsupported artifact value: {type(value).__name__}")
        add(self.artifacts)
        return digest.hexdigest()

    def state_dict(self):
        return {"version": VERSION, "mode": self.mode, "step": self.step,
                "artifact_identity": self._artifact_identity(),
                "calibration": self.calibration.state_dict(),
                "sampler": self.sampler.state_dict()}

    def load_state_dict(self, state):
        if state["version"] != VERSION or state["artifact_identity"] != self._artifact_identity():
            raise ValueError("Round 8 runtime implementation/artifact identity differs")
        # A fork must also originate from probes using this exact query protocol.
        from ..coherence.round8_queries import VERSION as query_version
        if state["sampler"].get("version") != query_version:
            raise ValueError("Round 8 checkpoint query protocol differs; fresh warmup required")
        # Mode changes are permitted only at the declared common-warmup fork.
        fork = state["mode"] == "warmup" and self.mode in {"control", "candidate"}
        if state["mode"] != self.mode and not (fork and state["step"] == 10 * self.steps_per_epoch):
            raise ValueError("Round 8 mode transition is outside the epoch-10 fork")
        self.calibration = InternalCalibration.from_state_dict(state["calibration"])
        if (self.calibration.priorities != dict(self.artifacts["priorities"]) or self.calibration.units != dict(self.artifacts["units"]) or self.calibration.active != dict(self.artifacts["active"])):
            raise ValueError("calibration state differs from frozen artifacts")
        if fork and self.calibration.coefficients is None:
            raise ValueError("warmup fork requires frozen calibration")
        if not fork:
            self.sampler = SharedQuerySampler.from_state_dict(state["sampler"])
        self.step = int(state["step"])
