"""Normal finite-primary Shape evaluation and artifact contracts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("gudhi")

from test_finite_primary_visualization import _fixture

from phycoflow_reconstruction.contracts import FieldSample
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation import reconstruction_set
from phycoflow_reconstruction.evaluation.coherence_set import build_coherence_accumulators
from phycoflow_reconstruction.training.run_store import file_sha256


def _runtime(tmp_path, *, generative=True):
    family, prediction, target, coords, _ = _fixture()
    artifact = tmp_path / "artifacts" / "topology_family.pt"
    artifact.parent.mkdir(parents=True)
    torch.save(family.state_artifact(), artifact)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"coherence_family_state_sha256s": {"topology": file_sha256(artifact)}})
    )
    runtime = SimpleNamespace(
        config={
            "coherence": {
                "families": {"topology": family.config},
                "compute_budget": {"point_count": 15, "query_seed": 7},
            },
            "observations": {
                "protocol": "random_uniform",
                "seed": 7,
                "fields": {"a": {"count": 2}},
            },
            "runtime": {"seed": 7},
        },
        dataset=SimpleNamespace(
            field_names=("a", "b"), data_spec=family_spec(), normalizer=FieldNormalizer.identity(2)
        ),
        model=SimpleNamespace(capabilities=SimpleNamespace(generative=generative)),
        device=torch.device("cpu"),
        checkpoint_path=tmp_path / "checkpoints" / "last.pt",
        generation_steps=2,
        seed=17,
    )
    return runtime, family, prediction, target, coords, artifact


def family_spec():
    from phycoflow_reconstruction.contracts import DataSpec

    return DataSpec(("a", "b"), ("1", "1"), 2, (15,))


def test_finite_primary_normal_set_scores_all_members_and_renders_all_filtrations(
    tmp_path, monkeypatch
):
    runtime, family, prediction, target, coords, _artifact = _runtime(tmp_path)
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
                values=target[0],
                coordinates=coords[0],
                coordinates_raw=coords[0],
                time=torch.tensor([0.0]),
                trajectory_id="fixture",
                time_index=0,
                conditions=torch.zeros(1),
                field_names=("a", "b"),
                logical_shape=(15,),
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
            noise = torch.randn(prediction.shape, generator=generator)
            return SimpleNamespace(prediction=prediction + 0.05 * noise)

    runtime.dataset = Dataset()
    runtime.model = Model()
    runtime.checkpoint_path.parent.mkdir(parents=True)
    torch.save({"family_states": {"topology": family.state_dict()}}, runtime.checkpoint_path)
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
        coherence_families=["topology"],
        max_samples=1,
    )
    directory = result.output_dir / "coherence/topology"
    report = json.loads((directory / "report.json").read_text())
    assert report["definition"] == "finite_primary_v1" and report["draw_count"] == 4
    assert len(runtime.model.seeds) == 5 and len(set(runtime.model.seeds[1:])) == 4
    assert report["aggregation"] == "paired_per_member_then_condition_mean"
    with np.load(directory / "metrics.npz", allow_pickle=False) as data:
        assert data["member_component_values"].shape[:2] == (1, 4)
        np.testing.assert_allclose(
            data["mean_component_values"], data["member_component_values"].mean(1)
        )
        assert sum(".h0.finite" in name for name in data["component_names"]) == 16
    with np.load(directory / "representative_payload.npz", allow_pickle=False) as data:
        assert len(data["unit_names"]) == 16
    assert len(list((directory / "filtrations").glob("*.png"))) == 16
    assert len(list((directory / "filtrations").glob("*.pdf"))) == 16
    comparison = reconstruction_set._render_posttraining_comparison(
        result, result, coherence_families=("topology",), statistic_scale="linear"
    )
    comparison_report = json.loads(comparison.read_text())
    assert comparison_report["matched_coherence_contracts"]["topology"]["draw_count"] == 4
    assert (directory / "report_base.json").is_file()
    assert len(list((directory / "filtrations_base").glob("*.png"))) == 16


def test_finite_primary_set_rejects_unfrozen_or_changed_artifact(tmp_path):
    runtime, _, prediction, target, coords, artifact = _runtime(tmp_path, generative=False)
    accumulator = build_coherence_accumulators(["topology"], runtime)["topology"]
    assert accumulator.draw_count == 1
    accumulator.update(prediction, target, coords, "a")
    assert any(".h0.finite" in name for name in accumulator.members[0][0])
    runtime.dataset.normalizer = FieldNormalizer(offset=[1.0, 0.0], scale=[1.0, 1.0])
    with pytest.raises(ValueError, match="normalizer differs"):
        build_coherence_accumulators(["topology"], runtime)
    runtime.dataset.normalizer = FieldNormalizer.identity(2)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"coherence_family_state_sha256s": {"topology": "0" * 64}})
    )
    with pytest.raises(ValueError, match="run manifest"):
        build_coherence_accumulators(["topology"], runtime)
    artifact.unlink()
    with pytest.raises(FileNotFoundError, match="frozen TRAIN artifact"):
        build_coherence_accumulators(["topology"], runtime)
