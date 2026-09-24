"""Recovery boundaries and run selection for the preemptable long-run launcher."""

import json

import pytest
import torch

from phycoflow_reconstruction.training import segmented as launcher
from phycoflow_reconstruction.training.run_store import RunStore


@pytest.mark.parametrize("step,expected", [(0, 3), (1, 2), (3, 9), (12, 12), (29, 1), (30, 0)])
def test_segment_boundaries_preserve_epoch_meaning(step, expected):
    assert launcher.segment_budget(step, 30, 3, 4) == expected


def test_launcher_counts_selected_training_snapshots(monkeypatch):
    class Dataset:
        count = 30
        closed = False

        def __len__(self):
            return self.count

        def close(self):
            self.closed = True

    dataset = Dataset()

    def open_dataset(config, *, split):
        assert split == "train"
        return dataset

    def select(dataset, settings):
        assert settings == {"fraction": 0.3}
        dataset.count = 9

    monkeypatch.setattr(launcher, "open_field_dataset", open_dataset)
    monkeypatch.setattr(launcher, "apply_training_subset", select)
    config = {
        "dataset": {},
        "optimization": {
            "batch_size": 2,
            "train_fraction": 1.0,
            "training_subset": {"fraction": 0.3},
        },
    }
    assert launcher.configured_epoch_steps(config) == 5
    assert dataset.closed
    config["optimization"].update(sampling="full_pass", steps_per_epoch=3)
    assert launcher.configured_epoch_steps(config) == 3
    assert launcher.segment_budget(4, 5 * 6, 5, 1) == 1
    assert launcher.segment_budget(5 * 6, 5 * 6, 5, 1) == 0


def _run(root, name, step, status="running"):
    run = root / name
    (run / "checkpoints").mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({"config_sha256": "recipe"}))
    (run / "status.json").write_text(json.dumps({"status": status, "global_step": step}))
    torch.save({"global_step": step}, run / "checkpoints/last.pt")
    return run


def test_terminal_checkpoint_is_selected_for_reporting_recovery(tmp_path):
    run = _run(tmp_path, "run", 30)
    assert launcher.select_run(tmp_path, "recipe") == (run, 30, False)


def test_completed_child_cannot_hide_a_second_resumable_child(tmp_path):
    _run(tmp_path, "a", 30, "completed")
    _run(tmp_path, "b", 9)
    with pytest.raises(RuntimeError, match="multiple resumable"):
        launcher.select_run(tmp_path, "recipe")


def test_setup_interruption_without_checkpoint_is_ignored(tmp_path):
    run = _run(tmp_path, "run", 0)
    (run / "checkpoints/last.pt").unlink()
    assert launcher.select_run(tmp_path, "recipe") == (None, 0, False)


def test_metric_recovery_drops_uncommitted_and_torn_writes(tmp_path):
    (tmp_path / "metrics").mkdir()
    committed = '{"step": 9, "value": 1}'
    for name in ("history", "validation_history", "topology_validation"):
        (tmp_path / f"metrics/{name}.jsonl").write_text(
            '{"step": 9, "value": 2}\n' + committed + '\n{"step": 18, "value": 3}\n{"step": 27,'
        )
    store = RunStore(tmp_path, {})
    store.recover_metric_histories(9)
    for path in (tmp_path / "metrics").glob("*.jsonl"):
        assert path.read_text() == committed + "\n"


def test_metric_recovery_does_not_hide_corruption_in_committed_history(tmp_path):
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics/history.jsonl").write_text('{broken}\n{"step": 9}\n')
    with pytest.raises(json.JSONDecodeError):
        RunStore(tmp_path, {}).recover_metric_histories(9)


def test_segmented_launcher_forwards_case_overrides_and_resume(tmp_path, monkeypatch):
    case = tmp_path / "custom_case"
    case.mkdir()
    (case / "run.py").write_text("# Synthetic entrypoint; subprocess is replaced below.\n")
    config_path = tmp_path / "posttrain.yaml"
    config_path.write_text("case: custom_case\n")
    overrides = ["source_run=/synthetic/source", "runtime.device=cpu"]
    config = {
        "stage": "post_training",
        "output": {"experiment_name": "test"},
        "optimization": {"epochs": 2},
    }

    def load_config(path, case_dir, case_name, values):
        assert (path, case_dir, case_name, values) == (config_path, case, "custom_case", overrides)
        return config

    monkeypatch.setattr(launcher, "_load_case_config", load_config)
    monkeypatch.setattr(launcher, "configured_epoch_steps", lambda config: 2)
    run = case / "runs/test/child"
    (run / "metrics").mkdir(parents=True)
    (run / "status.json").write_text(
        json.dumps(
            {
                "post_training_seconds": 0.1,
                "peak_cuda_memory_bytes": 0,
            }
        )
    )
    selections = iter(
        [(None, 0, False), (run, 2, False), (run, 2, False), (run, 4, True), (run, 4, True)]
    )
    monkeypatch.setattr(launcher, "select_run", lambda root, digest: next(selections))
    commands = []

    def launch(command, *, check):
        assert check
        commands.append(command)

    monkeypatch.setattr(launcher.subprocess, "run", launch)
    launcher.main(
        [
            "--case-dir",
            str(case),
            "--config",
            str(config_path),
            "--override",
            overrides[0],
            "--override",
            overrides[1],
        ]
    )
    assert len(commands) == 2
    for command in commands:
        assert command[2:5] == [str(case / "run.py"), "post-train", "--config"]
        assert command[command.index("--max-steps") + 1] == "2"
        assert command[8:12] == ["--override", overrides[0], "--override", overrides[1]]
    assert "--resume" not in commands[0]
    assert commands[1][-2:] == ["--resume", str(run)]
