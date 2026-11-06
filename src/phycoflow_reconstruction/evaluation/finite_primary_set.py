"""Fixed-query finite-primary Shape evaluation with a TRAIN-frozen artifact."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..coherence.registry import build_coherence_family
from ..data.training_batches import fixed_query_indices
from ..training.coherence_sampling import _coherence_query_seed, _load_checkpoint_family_state
from ..training.run_store import file_sha256, load_project_checkpoint


@dataclass
class FinitePrimarySetAccumulator:
    """Score each member separately; aggregate paired Shape costs over draws."""

    family: Any
    artifact_path: Path
    artifact_sha256: str
    query_point_count: int
    query_seed: int
    draw_count: int
    query_indices: torch.Tensor | None = None
    sample_ids: list[str] = field(default_factory=list)
    members: list[list[dict[str, float]]] = field(default_factory=list)
    representative: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @classmethod
    def build(cls, runtime, *, artifact_path=None):
        settings = runtime.config.get("coherence", {}).get("families", {}).get("topology")
        if not isinstance(settings, dict):
            raise TypeError("finite Shape evaluation needs its resolved family configuration")
        path = (
            Path(artifact_path).resolve()
            if artifact_path is not None
            else runtime.checkpoint_path.parent.parent / "artifacts" / "topology_family.pt"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"finite Shape evaluation needs a frozen TRAIN artifact: {path}"
            )
        digest = file_sha256(path)
        manifest_path = path.parent.parent / "run_manifest.json"
        manifest = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = manifest.get("coherence_family_state_sha256s", {}).get("topology")
            if expected is None or expected != digest:
                raise ValueError("finite Shape family artifact differs from its run manifest")
        family = build_coherence_family(
            "topology", settings, runtime.dataset.data_spec, runtime.dataset.normalizer
        ).to(runtime.device)
        if family.version != "4":
            raise ValueError("finite Shape accumulator requires finite_primary_v1")
        artifact = torch.load(path, map_location=runtime.device, weights_only=True)
        for name in ("normalization_offset", "normalization_scale"):
            if not torch.equal(
                getattr(family, name).detach().cpu(), artifact["state_dict"][name].detach().cpu()
            ):
                raise ValueError(
                    "finite Shape artifact normalizer differs from the evaluation dataset"
                )
        family.load_state_artifact(artifact)
        if (
            not family.geometry_sha256
            or not family.neighbor_indices.numel()
            or not family.spatial_objective.calibrated
        ):
            raise ValueError(
                "finite Shape evaluation requires frozen geometry and TRAIN/SOURCE scales"
            )
        if (
            path.parent.parent == runtime.checkpoint_path.parent.parent
            and runtime.checkpoint_path.is_file()
        ):
            checkpoint = load_project_checkpoint(runtime.checkpoint_path)
            state = checkpoint.get("family_states", {}).get("topology")
            if state is None:
                raise ValueError("checkpoint is missing frozen finite Shape family state")
            _load_checkpoint_family_state(
                family,
                state,
                source_checkpoint_sha256=manifest.get("source_hashes", {}).get("checkpoint"),
            )
        family.eval()
        return cls(
            family=family,
            artifact_path=path,
            artifact_sha256=digest,
            query_point_count=int(runtime.config["coherence"]["compute_budget"]["point_count"]),
            query_seed=_coherence_query_seed(runtime.config),
            draw_count=4 if runtime.model.capabilities.generative else 1,
        )

    @torch.no_grad()
    def update(self, prediction, target, coordinates, sample_id, *, conditional_draws=()):
        if len(conditional_draws) != self.draw_count - 1:
            raise ValueError(f"finite Shape evaluation needs {self.draw_count} member(s)")
        if self.query_indices is None:
            self.query_indices = fixed_query_indices(
                prediction.shape[1], self.query_point_count, seed=self.query_seed
            )
        indices = self.query_indices.to(prediction.device)
        selected_target, selected_coords = target[:, indices], coordinates[:, indices]
        selected_members = (
            prediction[:, indices],
            *(draw[:, indices] for draw in conditional_draws),
        )
        rows = []
        for member in selected_members:
            result = self.family(member, selected_target, coordinates=selected_coords)
            row = {
                name: float(value.scalar_loss.detach())
                for name, value in result.component_results.items()
            }
            row["family_total"] = float(result.scalar_loss.detach())
            if not all(np.isfinite(value) for value in row.values()):
                raise FloatingPointError("finite Shape evaluation produced non-finite values")
            rows.append(row)
        self.members.append(rows)
        self.sample_ids.append(str(sample_id))
        if self.representative is None:
            self.representative = tuple(
                x.detach().cpu() for x in (selected_members[0], selected_target, selected_coords)
            )

    def finalize(self, output_root, *, split, checkpoint_label, run_label, scale):
        del scale
        if not self.members or self.representative is None:
            raise ValueError("finite Shape evaluation has no snapshots")
        from .coherence_set import render_coherence_distribution
        from .finite_primary_visualization import (
            render_finite_primary_payload,
            save_finite_primary_payload,
        )

        names = tuple(self.members[0][0])
        if any(tuple(row) != names for members in self.members for row in members):
            raise ValueError("finite Shape leaf bank changed across members")
        values = np.asarray(
            [[[row[name] for name in names] for row in members] for members in self.members]
        )
        mean = values.mean(axis=1)
        destination = Path(output_root) / "coherence" / "topology"
        destination.mkdir(parents=True, exist_ok=True)
        metrics_path = destination / "metrics.npz"
        np.savez_compressed(
            metrics_path,
            sample_ids=np.asarray(self.sample_ids),
            component_names=np.asarray(names),
            member_component_values=values,
            mean_component_values=mean,
        )
        total_index = names.index("family_total")
        figure_path = render_coherence_distribution(
            mean[:, [total_index]],
            ["finite-primary Shape"],
            ["family_total"],
            destination / "finite_primary_distance.png",
            title=f"{run_label} — {split} Shape",
            subtitle=f"{checkpoint_label}; {len(mean)} conditions; mean of {self.draw_count} member(s)",
            ylabel="Paired finite-primary distance",
            scale="linear",
        )
        device = self.family.normalization_offset.device
        prediction, target, coordinates = (x.to(device) for x in self.representative)
        payload_path = save_finite_primary_payload(
            self.family,
            prediction,
            target,
            coordinates=coordinates,
            output_path=destination / "representative_payload.npz",
            sample_id=self.sample_ids[0],
            endpoint=checkpoint_label,
        )
        bank_figures = render_finite_primary_payload(payload_path, destination / "filtrations")
        report = {
            "family": "topology",
            "definition": "finite_primary_v1",
            "split": split,
            "sample_ids": self.sample_ids,
            "aggregation": "paired_per_member_then_condition_mean",
            "draw_count": self.draw_count,
            "query_policy": "frozen_training_fixed_shared",
            "query_indices_sha256": hashlib.sha256(
                self.query_indices.cpu().numpy().tobytes()
            ).hexdigest(),
            "geometry_sha256": self.family.geometry_sha256,
            "geometry": self.family.geometry_diagnostics,
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": self.artifact_sha256,
            "calibration": self.family.spatial_objective.calibration_record,
            "component_names": names,
            "member_component_values": values.tolist(),
            "mean_component_values": mean.tolist(),
            "representative": {
                "policy": "first_selected_condition_first_member",
                "sample_id": self.sample_ids[0],
            },
            "artifacts": {
                "metrics_payload": metrics_path.name,
                "figure": figure_path.name,
                "representative_payload": payload_path.name,
                "filtrations": [str(path.relative_to(destination)) for path in bank_figures],
            },
        }
        report_path = destination / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        return {
            "family": "topology",
            "definition": "finite_primary_v1",
            "directory": str(destination),
            "report": str(report_path),
            "metrics_payload": str(metrics_path),
            "figures": {"finite_primary_distance": str(figure_path)},
            "representative_payload": str(payload_path),
        }
