"""Re-score saved post-training checkpoints on their run's exact validation panel.

The run's frozen coherence families, calibration scales, sensors, and comparison
queries are restored and verified against its manifest; only the requested
``evaluation`` overrides (for example ``generation_steps``) change. The source and each
checkpoint pass through the trainer's own evaluation and selection gate. The run
directory is never modified.

    python -m phycoflow_reconstruction.training.posttrain_rescore RUN_DIR \\
        --checkpoint epoch_050 --checkpoint last --steps 16 --output rescore.json
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..coherence import build_enabled_families
from ..config.load import load_config
from ..config.validate import validate_config
from ..data.factory import open_field_dataset
from ..data.manifest import SensorManifest, manifest_from_batch
from ..utils.reproducibility import seed_everything
from .common import sensor_protocol_from_config
from .model_lifecycle import load_training_aux_state
from .post_training import (
    _build_comparison_batch,
    _build_evaluation_batch,
    _evaluate,
    _fidelity_guard_report,
    _reference_bank,
)
from .run_store import file_sha256, load_model_state_strict, load_project_checkpoint
from .source import load_source_model, source_hashes
from .topology_selection import topology_selection_report


def _without_query_ids(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "query_ids"}


@dataclass
class RestoredPanel:
    """A post-training run's verified validation panel around its (mutable) source model."""

    run_dir: Path
    config: dict[str, Any]
    model: torch.nn.Module
    families: dict[str, Any]
    banks: dict[str, Any]
    family_scales: dict[str, float]
    evaluation_batch: Any
    comparison_batch: Any
    field_names: tuple[str, ...]
    normalizer: Any

    def score(self) -> dict[str, Any]:
        # The panel's query indices were verified on restore; omit them from every report.
        return _without_query_ids(
            _evaluate(
                self.model,
                self.evaluation_batch,
                self.comparison_batch,
                self.families,
                self.banks,
                self.field_names,
                self.config,
                normalizer=self.normalizer,
                family_scales=self.family_scales,
            )
        )

    def load_checkpoint(self, name: str) -> int:
        payload = load_project_checkpoint(self.run_dir / "checkpoints" / f"{name}.pt")
        load_model_state_strict(self.model, payload["model"])
        load_training_aux_state(self.model, payload)
        for family_name, family in self.families.items():
            if family_name in payload.get("family_states", {}):
                family.load_state_dict(payload["family_states"][family_name])
        return int(payload["global_step"])


def restore_evaluation_panel(
    run_dir: str | Path,
    *,
    evaluation_overrides: dict[str, Any],
    device_name: str | None = None,
) -> RestoredPanel:
    run_dir = Path(run_dir).resolve()
    config = deepcopy(load_config(run_dir / "resolved_config.yaml"))
    run_split = str(config.get("evaluation", {}).get("split", "validation"))
    overrides = dict(evaluation_overrides)
    requested_split = overrides.pop("split", None)
    config["evaluation"] = {**config.get("evaluation", {}), **overrides}
    if device_name is not None:
        config["runtime"]["device"] = device_name
    # Validate with the run's own split (selection rules refer to the validation panel); a held-out
    # split requested for offline scoring is applied afterwards and only changes which days are drawn.
    validate_config(config)
    if requested_split is not None:
        config["evaluation"]["split"] = str(requested_split)
    seed_everything(
        int(config["runtime"].get("seed", 42)), bool(config["runtime"].get("deterministic", True))
    )
    device = torch.device(config["runtime"].get("device", "cpu"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if source_hashes(config) != manifest["source_hashes"]:
        raise ValueError("source files differ from the run manifest")
    model, train_dataset, _ = load_source_model(config, device)
    families = build_enabled_families(
        config["coherence"], train_dataset.data_spec, train_dataset.normalizer
    )
    families = {name: family.to(device) for name, family in families.items()}
    banks = {}
    for name, family in families.items():
        path = run_dir / "artifacts" / f"{name}_family.pt"
        if file_sha256(path) != manifest["coherence_family_state_sha256s"][name]:
            raise ValueError(f"{name} family artifact differs from the run manifest")
        family.load_state_artifact(torch.load(path, map_location=device, weights_only=True))
        bank_name = (
            "coherence_reference.pt"
            if name == "global_distribution"
            else f"coherence_reference_{name}.pt"
        )
        banks[name] = _reference_bank(
            config, train_dataset, existing_path=run_dir / "artifacts" / bank_name, family_name=name
        )
    if (
        file_sha256(run_dir / "artifacts/coherence_calibration.json")
        != manifest["coherence_calibration_sha256"]
    ):
        raise ValueError("coherence calibration differs from the run manifest")
    calibration = json.loads(
        (run_dir / "artifacts" / "coherence_calibration.json").read_text(encoding="utf-8")
    )
    family_scales = {name: float(calibration["resolved_scales"][name]) for name in families}

    split = str(config["evaluation"].get("split", "validation"))
    evaluation_dataset = open_field_dataset(
        config["dataset"], split=split, normalizer=train_dataset.normalizer
    )
    evaluation_batch = _build_evaluation_batch(
        evaluation_dataset, config, sensor_protocol_from_config(config), device
    )
    comparison_batch = _build_comparison_batch(evaluation_batch, config)
    if split == run_split:
        # The run's own panel: it must reproduce the saved sensors and queries exactly.
        saved = SensorManifest.load(run_dir / "artifacts" / "evaluation_sensor_manifest.json")
        if manifest_from_batch(comparison_batch, evaluation_dataset.path, split).digest() != (
            saved.digest()
        ):
            raise ValueError("rebuilt evaluation sensors differ from the run's saved manifest")
        saved_queries = torch.load(
            run_dir / "artifacts" / "evaluation_query_indices.pt", weights_only=True
        )["query_indices"]
        if not torch.equal(
            saved_queries.cpu(), comparison_batch.metadata["query_indices"].detach().cpu()
        ):
            raise ValueError("rebuilt evaluation queries differ from the run's saved indices")
    else:
        # A held-out split requested through evaluation_overrides: a fresh panel built with the run's own
        # protocol (sample selection, seed, sensors, queries); there is no saved manifest to compare with.
        print(
            f"restore_evaluation_panel: fresh {split!r} panel (the run's panel is {run_split!r}); "
            "saved sensor/query manifests are not applicable"
        )
    evaluation_dataset.close()
    train_dataset.close()
    return RestoredPanel(
        run_dir=run_dir,
        config=config,
        model=model,
        families=families,
        banks=banks,
        family_scales=family_scales,
        evaluation_batch=evaluation_batch,
        comparison_batch=comparison_batch,
        field_names=tuple(train_dataset.field_names),
        normalizer=train_dataset.normalizer,
    )


def rescore_post_training_run(
    run_dir: str | Path,
    *,
    checkpoints: list[str],
    evaluation_overrides: dict[str, Any],
    device_name: str | None = None,
) -> dict[str, Any]:
    panel = restore_evaluation_panel(
        run_dir, evaluation_overrides=evaluation_overrides, device_name=device_name
    )
    config = panel.config
    selection_metric = config.get("checkpointing", {}).get("selection_metric")

    def gate(metrics: dict[str, Any]) -> dict[str, Any]:
        # The trainer's checkpoint gate, else its final source-relative fidelity report.
        if selection_metric == "topology_with_fidelity":
            return topology_selection_report(source, metrics, config["posttrain_fidelity"])
        return _fidelity_guard_report(source, metrics, config)

    source = panel.score()
    rows = []
    for name in checkpoints:
        global_step = panel.load_checkpoint(name)
        metrics = panel.score()
        report = gate(metrics)
        report.pop("metrics", None)
        rows.append(
            {
                "checkpoint": name,
                "global_step": global_step,
                "metrics": metrics,
                "selection": report,
            }
        )
    return {
        "run_dir": str(panel.run_dir),
        "evaluation_overrides": evaluation_overrides,
        "recorded_source": _without_query_ids(
            json.loads((panel.run_dir / "evaluation" / "before.json").read_text(encoding="utf-8"))
        ),
        "source": source,
        "checkpoints": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = rescore_post_training_run(
        args.run_dir,
        checkpoints=args.checkpoint,
        evaluation_overrides={} if args.steps is None else {"generation_steps": args.steps},
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Rescored {len(payload['checkpoints'])} checkpoints: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
