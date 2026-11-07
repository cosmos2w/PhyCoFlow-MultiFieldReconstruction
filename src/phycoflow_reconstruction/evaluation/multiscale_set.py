"""Evaluate Scale on a previously frozen TRAIN geometry and calibration."""

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
from ..training.coherence_sampling import (
    _coherence_query_seed,
    _load_checkpoint_family_state,
    _requires_frozen_checkpoint_state,
)
from ..training.run_store import file_sha256, load_project_checkpoint


@dataclass
class MultiscaleSetAccumulator:
    """Score independent conditional draws without fitting on the evaluation split."""

    family: Any
    artifact_path: Path
    artifact_sha256: str
    query_point_count: int
    query_seed: int
    draw_count: int
    sample_ids: list[str] = field(default_factory=list)
    rows: list[dict[str, float]] = field(default_factory=list)
    band_rows: list[dict[str, Any]] = field(default_factory=list)
    query_indices: torch.Tensor | None = None

    @classmethod
    def build(cls, runtime: Any, *, artifact_path: str | Path | None = None):
        configured = (
            runtime.config.get("coherence", {}).get("families", {}).get("multiscale_spectrum")
        )
        if not isinstance(configured, dict):
            raise TypeError("multiscale set evaluation needs its resolved family configuration")
        path = (
            Path(artifact_path).resolve()
            if artifact_path is not None
            else runtime.checkpoint_path.parent.parent
            / "artifacts"
            / "multiscale_spectrum_family.pt"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"multiscale set evaluation needs a frozen TRAIN artifact: {path}"
            )
        digest = file_sha256(path)
        manifest_path = path.parent.parent / "run_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = manifest.get("coherence_family_state_sha256s", {}).get("multiscale_spectrum")
            if expected is None or digest != expected:
                raise ValueError("multiscale family artifact differs from its run manifest")
        family = build_coherence_family(
            "multiscale_spectrum", configured, runtime.dataset.data_spec, runtime.dataset.normalizer
        ).to(runtime.device)
        artifact = torch.load(path, map_location=runtime.device, weights_only=True)
        for name in ("normalization_offset", "normalization_scale"):
            if not torch.equal(
                getattr(family, name).detach().cpu(), artifact["state_dict"][name].detach().cpu()
            ):
                raise ValueError(
                    "multiscale artifact normalizer differs from the evaluation dataset"
                )
        family.load_state_artifact(artifact)
        if not family.fitted or family.calibration_record.get("split") != "train":
            raise ValueError("multiscale set evaluation requires TRAIN-fitted geometry and scales")
        # A child checkpoint must agree with its run-owned frozen family; a
        # SOURCE comparison intentionally uses the explicitly supplied child artifact.
        if (
            path.parent.parent == runtime.checkpoint_path.parent.parent
            and runtime.checkpoint_path.is_file()
        ):
            checkpoint = load_project_checkpoint(runtime.checkpoint_path)
            state = checkpoint.get("family_states", {}).get("multiscale_spectrum")
            if state is not None:
                _load_checkpoint_family_state(family, state)
            elif _requires_frozen_checkpoint_state(family):
                raise ValueError("checkpoint is missing frozen multiscale family state")
        family.eval()
        stochastic = family.objective in {"signed_band_energy_score", "conditional_energy_score"}
        if stochastic and not runtime.model.capabilities.generative:
            raise ValueError("conditional Scale evaluation requires a generative model")
        return cls(
            family=family,
            artifact_path=path,
            artifact_sha256=digest,
            query_point_count=int(runtime.config["coherence"]["compute_budget"]["point_count"]),
            query_seed=_coherence_query_seed(runtime.config),
            draw_count=4 if stochastic else 1,
        )

    @torch.no_grad()
    def update(self, prediction, target, coordinates, sample_id, *, conditional_draws=()):
        if len(conditional_draws) != self.draw_count - 1:
            raise ValueError(
                f"multiscale evaluation needs exactly {self.draw_count} independent draws"
            )
        if self.query_indices is None:
            self.query_indices = fixed_query_indices(
                prediction.shape[1], self.query_point_count, seed=self.query_seed
            )
            digest = hashlib.sha256(self.query_indices.cpu().numpy().tobytes()).hexdigest()
            expected = self.family.calibration_record.get("query_indices_sha256")
            if expected is not None and digest != expected:
                raise ValueError(
                    "multiscale evaluation queries differ from the frozen TRAIN artifact"
                )
        indices = self.query_indices.to(prediction.device)
        context = {}
        if self.draw_count > 1:
            context = {
                "conditional_draws": tuple(draw[:, indices] for draw in conditional_draws),
                "score_all_conditional_draws": True,
            }
        result = self.family(
            prediction[:, indices],
            target[:, indices],
            coordinates=coordinates[:, indices],
            context=context,
        )
        row = {
            path: float(term.scalar_loss.detach())
            for path, term in result.component_results.items()
        }
        row["family_total"] = float(result.scalar_loss.detach())
        if any(not np.isfinite(value) for value in row.values()):
            raise FloatingPointError("multiscale set evaluation produced a non-finite value")
        self.rows.append(row)
        self.sample_ids.append(str(sample_id))
        signed = (
            result.diagnostics.get("signed_band")
            or result.diagnostics.get("paired_signed_band")
            or {}
        )
        primary = result.component_results.get(
            "multiscale_spectrum.signed_band_energy_score.conditional"
        )
        self.band_rows.append(
            {
                "score": signed.get("energy_score_per_band", signed.get("distance_per_band")),
                "attraction": None
                if primary is None
                else primary.diagnostics.get("attraction_per_band"),
                "repulsion": None
                if primary is None
                else primary.diagnostics.get("repulsion_per_band"),
                "joint": signed.get("joint_energy_score", signed.get("joint_distance")),
            }
        )

    def finalize(self, output_root, *, split, checkpoint_label, run_label, scale):
        del scale
        if not self.rows:
            raise ValueError("multiscale set evaluation has no snapshots")
        from .coherence_set import render_coherence_distribution

        names = tuple(self.rows[0])
        if any(tuple(row) != names for row in self.rows):
            raise ValueError("multiscale result components changed across samples")
        values = np.asarray([[row[name] for name in names] for row in self.rows], dtype=np.float64)
        destination = Path(output_root) / "coherence" / "multiscale_spectrum"
        destination.mkdir(parents=True, exist_ok=True)
        payload_path = destination / "metrics.npz"
        np.savez_compressed(
            payload_path,
            component_names=np.asarray(names),
            component_values=values,
            sample_ids=np.asarray(self.sample_ids),
            per_band_score=np.asarray([row["score"] for row in self.band_rows])
            if self.band_rows[0]["score"] is not None
            else np.empty((len(values), 0)),
            per_band_attraction=np.asarray([row["attraction"] for row in self.band_rows])
            if self.band_rows[0]["attraction"] is not None
            else np.empty((len(values), 0)),
            per_band_repulsion=np.asarray([row["repulsion"] for row in self.band_rows])
            if self.band_rows[0]["repulsion"] is not None
            else np.empty((len(values), 0)),
        )
        selected = [
            i
            for i, name in enumerate(names)
            if (
                name == "family_total"
                or "signed_band_energy_score" in name
                or "joint_spectral" in name
                or "paired_signed_band_distance" in name
                or "gram_self_energy_score" in name
                or "gram_mutual_energy_score" in name
            )
        ]
        if not selected:
            selected = [len(names) - 1]
        plot_values = values[:, selected]
        minimum, maximum = float(plot_values.min()), float(plot_values.max())
        lower = min(0.0, minimum * 1.1)
        upper = max(maximum * 1.1, np.finfo(np.float64).eps)
        figure_path = render_coherence_distribution(
            plot_values,
            [names[i].replace("multiscale_spectrum.", "") for i in selected],
            ["family_total" if names[i] == "family_total" else "detail" for i in selected],
            destination / "scale_distance.png",
            title=f"{run_label} — {split} Scale",
            subtitle=f"{checkpoint_label}; {len(values)} conditions; {self.draw_count} draw(s)",
            ylabel="Represented Scale discrepancy",
            scale="linear",
            value_limits=(lower, upper),
        )
        report = {
            "family": "multiscale_spectrum",
            "split": split,
            "sample_ids": self.sample_ids,
            "objective": self.family.objective,
            "score_definition_version": getattr(
                self.family, "score_definition_version", self.family.settings.score_definition_version
            ),
            "anchor_score_definition_version": self.family.settings.score_definition_version,
            "draw_count": self.draw_count,
            "query_policy": "frozen_training_fixed_shared",
            "query_indices_sha256": hashlib.sha256(
                self.query_indices.cpu().numpy().tobytes()
            ).hexdigest(),
            "geometry": self.family.geometry_record,
            "calibration": self.family.calibration_record,
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": self.artifact_sha256,
            "component_means": {
                name: float(values[:, index].mean()) for index, name in enumerate(names)
            },
            "per_sample_band_and_mean_diagnostics": self.band_rows,
            "artifacts": {"metrics_payload": payload_path.name, "figure": figure_path.name},
        }
        report_path = destination / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        return {
            "family": "multiscale_spectrum",
            "directory": str(destination),
            "report": str(report_path),
            "metrics_payload": str(payload_path),
            "figures": {"scale_distance": str(figure_path)},
        }
