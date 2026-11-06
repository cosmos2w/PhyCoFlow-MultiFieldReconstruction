"""Normal Scale evaluation must use a frozen TRAIN artifact and real draws."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from phycoflow_reconstruction.coherence.families.multiscale_spectrum import MultiscaleSpectrumFamily
from phycoflow_reconstruction.contracts import DataSpec, FieldSample
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation.coherence_set import build_coherence_accumulators
from phycoflow_reconstruction.training.run_store import file_sha256


def _runtime(tmp_path: Path, *, point=False, generative=True):
    objective = {
        "kind": "paired_signed_band_mean_distance" if point else "signed_band_energy_score",
        "conditional_draws": 1 if point else 2,
        "score_definition_version": "paired_signed_spectral_mean_v1"
        if point
        else "signed_spectral_mean_augmented_v4",
    }
    if not point:
        objective.update(almost_fair_alpha=1.0, joint_energy_score_weight=0.25)
    config = {
        "fields": ["u"],
        "geometry": {"operator": "fem_p1", "coordinate_scaling": "none"},
        "bands": {"count": 2},
        "calibration": {"field_unit_floor": 0.01},
        "numerics": {"dtype": "float64"},
        "objective": objective,
    }
    spec = DataSpec(("u",), ("1",), 1, (16,))
    normalizer = FieldNormalizer.identity(1)
    # Sensor batches retain float32 query coordinates in the production path.
    coords = (torch.linspace(0, 1, 16, dtype=torch.float32)[:, None] ** 1.3).double()
    states = torch.randn(8, 16, 1, generator=torch.Generator().manual_seed(16), dtype=torch.float64)
    family = MultiscaleSpectrumFamily(config, spec, normalizer)
    query = torch.arange(16, dtype=torch.long)
    family.fit_reference_tensors(
        coords,
        states,
        sample_ids=list("abcdefgh"),
        provenance={
            "split": "train",
            "query_indices_sha256": hashlib.sha256(query.numpy().tobytes()).hexdigest(),
        },
    )
    artifact = tmp_path / "artifacts" / "multiscale_spectrum_family.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(family.state_artifact(), artifact)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {"coherence_family_state_sha256s": {"multiscale_spectrum": file_sha256(artifact)}}
        )
    )
    runtime = SimpleNamespace(
        config={
            "coherence": {
                "families": {"multiscale_spectrum": config},
                "compute_budget": {"point_count": 16, "query_seed": 7},
            },
            "observations": {},
            "runtime": {},
        },
        dataset=SimpleNamespace(field_names=("u",), data_spec=spec, normalizer=normalizer),
        model=SimpleNamespace(capabilities=SimpleNamespace(generative=generative)),
        device=torch.device("cpu"),
        checkpoint_path=tmp_path / "checkpoints" / "last.pt",
        seed=17,
    )
    return runtime, coords, states, artifact


def test_multiscale_set_uses_four_draws_and_frozen_train_artifact(tmp_path):
    runtime, coords, states, artifact = _runtime(tmp_path)
    accumulator = build_coherence_accumulators(["multiscale_spectrum"], runtime)[
        "multiscale_spectrum"
    ]
    assert accumulator.draw_count == 4 and accumulator.artifact_sha256 == file_sha256(artifact)
    with pytest.raises(ValueError, match="exactly 4"):
        accumulator.update(states[:1], states[1:2], coords[None], "a")
    accumulator.update(
        states[:1],
        states[1:2],
        coords[None],
        "a",
        conditional_draws=tuple(states[i : i + 1] for i in (2, 3, 4)),
    )
    output = accumulator.finalize(
        tmp_path / "out",
        split="validation",
        checkpoint_label="last",
        run_label="fixture",
        scale="log",
    )
    report = json.loads(Path(output["report"]).read_text())
    assert report["draw_count"] == 4 and report["calibration"]["split"] == "train"
    assert report["per_sample_band_and_mean_diagnostics"][0]["joint"]["conditional_draws"] == 4
    with np.load(output["metrics_payload"], allow_pickle=False) as payload:
        assert payload["per_band_attraction"].shape == payload["per_band_repulsion"].shape == (1, 2)
        assert np.isfinite(payload["component_values"]).all()
    figure = Path(output["figures"]["scale_distance"])
    with Image.open(figure) as image:
        image.verify()
    assert figure.with_suffix(".pdf").is_file()


def test_multiscale_set_requires_matching_artifact_and_model_capability(tmp_path):
    runtime, coords, states, artifact = _runtime(tmp_path, generative=False)
    with pytest.raises(ValueError, match="generative model"):
        build_coherence_accumulators(["multiscale_spectrum"], runtime)
    runtime.model.capabilities.generative = True
    runtime.dataset.normalizer = FieldNormalizer(offset=[1.0], scale=[1.0])
    with pytest.raises(ValueError, match="normalizer differs"):
        build_coherence_accumulators(["multiscale_spectrum"], runtime)
    runtime.dataset.normalizer = FieldNormalizer.identity(1)
    manifest = artifact.parent.parent / "run_manifest.json"
    manifest.write_text(
        json.dumps({"coherence_family_state_sha256s": {"multiscale_spectrum": "0" * 64}})
    )
    with pytest.raises(ValueError, match="run manifest"):
        build_coherence_accumulators(["multiscale_spectrum"], runtime)
    manifest.unlink()
    accumulator = build_coherence_accumulators(["multiscale_spectrum"], runtime)[
        "multiscale_spectrum"
    ]
    altered = coords.clone()
    altered[0] += 0.01
    with pytest.raises(ValueError, match="differ from the fitted geometry"):
        accumulator.update(
            states[:1],
            states[1:2],
            altered[None],
            "a",
            conditional_draws=tuple(states[i : i + 1] for i in (2, 3, 4)),
        )


def test_deterministic_multiscale_set_uses_one_paired_point(tmp_path):
    runtime, coords, states, _ = _runtime(tmp_path, point=True, generative=False)
    accumulator = build_coherence_accumulators(["multiscale_spectrum"], runtime)[
        "multiscale_spectrum"
    ]
    assert accumulator.draw_count == 1
    accumulator.update(states[:1], states[1:2], coords[None], "a")
    assert "multiscale_spectrum.paired_joint_spectral_mean_distance.paired" in accumulator.rows[0]


def test_normal_set_route_requests_four_target_free_draws(tmp_path, monkeypatch):
    from phycoflow_reconstruction.evaluation import reconstruction_set

    runtime, coords, states, artifact = _runtime(tmp_path)
    dataset_path = tmp_path / "dataset.bin"
    dataset_path.write_bytes(b"fixture")

    class Dataset:
        path = dataset_path
        field_names = runtime.dataset.field_names
        data_spec = runtime.dataset.data_spec
        normalizer = runtime.dataset.normalizer

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return FieldSample(
                values=states[0],
                coordinates=coords,
                coordinates_raw=coords,
                time=torch.tensor([0.0]),
                trajectory_id="fixture",
                time_index=0,
                conditions=torch.zeros(1),
                field_names=("u",),
                logical_shape=(16,),
                reconstruction_unit="snapshot",
            )

        def close(self):
            pass

    class Model(torch.nn.Module):
        capabilities = SimpleNamespace(generative=True)

        def __init__(self):
            super().__init__()
            self.seeds = []

        def reconstruct(self, batch, *, steps, generator):
            assert batch.target_fields is None
            self.seeds.append(generator.initial_seed())
            noise = torch.randn(
                batch.query_coords.shape[:2] + (1,),
                generator=generator,
                dtype=batch.query_coords.dtype,
                device=batch.query_coords.device,
            )
            return SimpleNamespace(prediction=0.2 * noise + batch.obs_values.mean())

    runtime.dataset = Dataset()
    runtime.model = Model()
    runtime.generation_steps = 2
    runtime.config["observations"] = {
        "protocol": "random_uniform",
        "seed": 7,
        "fields": {"u": {"count": 2}},
    }
    runtime.checkpoint_path.parent.mkdir(parents=True)
    frozen = torch.load(artifact, weights_only=True)
    torch.save(
        {"family_states": {"multiscale_spectrum": frozen["state_dict"]}}, runtime.checkpoint_path
    )
    (tmp_path / "resolved_config.yaml").write_text("fixture: true\n")
    monkeypatch.setattr(
        reconstruction_set, "load_evaluation_runtime", lambda *args, **kwargs: runtime
    )
    monkeypatch.setattr(
        reconstruction_set, "warn_if_cuda_memory_tight", lambda *args, **kwargs: False
    )
    result = reconstruction_set._evaluate_reconstruction_set_once(
        tmp_path,
        case_dir=tmp_path.parent,
        split="validation",
        checkpoint="last",
        coherence_families=["multiscale_spectrum"],
        max_samples=1,
    )
    report = json.loads(
        (result.output_dir / "coherence/multiscale_spectrum/report.json").read_text()
    )
    assert report["draw_count"] == 4
    # One warm-up plus four evaluation draws; the latter have distinct seeds.
    assert len(runtime.model.seeds) == 5
    assert len(set(runtime.model.seeds[1:])) == 4
    comparison = reconstruction_set._render_posttraining_comparison(
        result, result, coherence_families=("multiscale_spectrum",), statistic_scale="linear"
    )
    compared = json.loads(comparison.read_text())
    assert compared["matched_coherence_contracts"]["multiscale_spectrum"]["draw_count"] == 4
    base_report = result.output_dir / "coherence/multiscale_spectrum/report_base.json"
    assert base_report.is_file()
    assert json.loads(base_report.read_text())["artifacts"]["metrics_payload"] == "metrics_base.npz"


def test_cli_accepts_and_routes_multiscale_set_selection(tmp_path, monkeypatch):
    import sys

    from phycoflow_reconstruction.cli import run_case_cli
    from phycoflow_reconstruction.evaluation import reconstruction_set

    calls = []
    monkeypatch.setattr(
        reconstruction_set,
        "evaluate_reconstruction_set",
        lambda *args, **kwargs: calls.append(kwargs) or tmp_path / "figure.png",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "visualize-run",
            "--run",
            "example",
            "--eval-set",
            "validation",
            "--eval-coherence",
            "global_distribution",
            "multiscale_spectrum",
            "topology",
        ],
    )
    assert run_case_cli("case", tmp_path) == 0
    assert calls[0]["coherence_families"] == [
        "global_distribution",
        "multiscale_spectrum",
        "topology",
    ]
