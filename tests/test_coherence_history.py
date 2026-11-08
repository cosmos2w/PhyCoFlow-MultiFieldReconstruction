"""Generic coherence-component history extraction and rendering coverage."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from phycoflow_reconstruction.contracts import FamilyResult, TermResult
from phycoflow_reconstruction.training.coherence_history import (
    build_coherence_history_figure,
    extract_coherence_history,
    render_coherence_history,
)
from phycoflow_reconstruction.training.monitoring import TrainingMonitor
from phycoflow_reconstruction.training.post_training import _component_history_report


def _config() -> dict:
    return {
        "stage": "post_training",
        "model": {"name": "fixture"},
        "coherence": {
            "families": {
                "global_distribution": {
                    "enabled": True,
                    "weight": 3.0,
                    "components": {
                        "self": {"enabled": True, "weight": 2.0},
                    },
                },
                "cross_spectrum": {
                    "enabled": True,
                    "weight": 1.0,
                    "components": {
                        "self_spectrum": {"enabled": True, "weight": 1.0},
                        "same_frequency": {"enabled": True, "weight": 1.0},
                    },
                },
            }
        },
    }


def test_namespaced_component_history_is_grouped_and_partial_epochs_are_preserved() -> None:
    rows = [
        {
            "epoch": 1,
            "epoch_complete": True,
            "coherence_loss": 4.0,
            "coherence_family/cross_spectrum/weighted_contribution": 4.0,
            "coherence_component/cross_spectrum/self_spectrum.auto_spectrum/raw": 3.0,
            "coherence_component/cross_spectrum/self_spectrum.auto_spectrum/weighted_contribution": 3.0,
            "coherence_component/cross_spectrum/same_frequency.magnitude_squared/raw": 1.0e-3,
            "coherence_component/cross_spectrum/same_frequency.magnitude_squared/weighted_contribution": 1.0e-3,
        },
        {
            "epoch": 2,
            "epoch_complete": False,
            "coherence_loss": 2.0,
            "coherence_family/cross_spectrum/weighted_contribution": 2.0,
            "coherence_component/cross_spectrum/self_spectrum.auto_spectrum/raw": 1.5,
            "coherence_component/cross_spectrum/self_spectrum.auto_spectrum/weighted_contribution": 1.5,
            "coherence_component/cross_spectrum/same_frequency.magnitude_squared/raw": 5.0e-4,
            "coherence_component/cross_spectrum/same_frequency.magnitude_squared/weighted_contribution": 5.0e-4,
        },
    ]

    data = extract_coherence_history(rows, _config())

    assert [component.family for component in data.components] == [
        "cross_spectrum",
        "cross_spectrum",
    ]
    assert data.family_order == ("global_distribution", "cross_spectrum")
    assert data.total_epochs == (1.0, 2.0)
    assert all(component.partial_epochs == (2.0,) for component in data.components)


def test_legacy_flat_component_keys_recover_effective_weighted_contribution() -> None:
    rows = [
        {
            "epoch": 1,
            "global_distribution.self.marginal_w2": 0.5,
            "coherence_family/global_distribution/calibration_scale": 4.0,
            "coherence_family/global_distribution/weighted_contribution": 12.0,
            "coherence_loss": 12.0,
        }
    ]

    data = extract_coherence_history(rows, _config())

    assert len(data.components) == 1
    component = data.components[0]
    assert component.component == "self.marginal_w2"
    assert component.raw == (0.5,)
    assert component.weighted == (12.0,)  # 0.5 × inner 2 × outer 3 × calibration 4


def test_namespaced_history_extends_legacy_epochs_after_resume() -> None:
    rows = [
        {
            "epoch": 1,
            "cross_spectrum.self_spectrum.auto_spectrum": 3.0,
        },
        {
            "epoch": 2,
            "cross_spectrum.self_spectrum.auto_spectrum": 99.0,
            "coherence_component/cross_spectrum/self_spectrum.auto_spectrum/raw": 2.0,
            "coherence_component/cross_spectrum/self_spectrum.auto_spectrum/weighted_contribution": 2.0,
        },
    ]

    data = extract_coherence_history(rows, _config())

    component = data.components[0]
    assert component.epochs == (1.0, 2.0)
    assert component.raw == (3.0, 2.0)


def test_component_history_report_is_additive_across_inner_outer_and_calibration() -> None:
    result = FamilyResult(
        component_results={},
        per_sample_cost=None,
        scalar_loss=torch.tensor(1.0),
        diagnostics={
            "families": {
                "example": {
                    "weight": 3.0,
                    "calibration_scale": 4.0,
                    "components": {
                        "example.self.metric": {
                            "weight": 2.0,
                            "executed": True,
                            "raw_scalar_loss": 0.5,
                        },
                        "example.disabled.metric": {
                            "weight": 0.0,
                            "executed": False,
                            "raw_scalar_loss": None,
                        },
                    },
                }
            }
        },
    )

    report = _component_history_report(result)

    prefix = "coherence_component/example/self.metric"
    assert report[f"{prefix}/raw"] == 0.5
    assert report[f"{prefix}/weighted_contribution"] == 12.0
    assert not any("disabled" in key for key in report)


def test_component_history_report_supports_families_without_component_diagnostics() -> None:
    path = "topology.self.betti_curves"
    result = FamilyResult(
        component_results={path: TermResult(None, torch.tensor(0.25))},
        per_sample_cost=None,
        scalar_loss=torch.tensor(1.0),
        diagnostics={
            "families": {
                "topology": {
                    "weight": 2.0,
                    "calibration_scale": 3.0,
                }
            }
        },
    )
    families = {"topology": SimpleNamespace(component_weights={"self": 5.0})}

    report = _component_history_report(result, families)

    prefix = "coherence_component/topology/self.betti_curves"
    assert report[f"{prefix}/raw"] == 0.25
    assert report[f"{prefix}/inner_weight"] == 5.0
    assert report[f"{prefix}/weighted_contribution"] == 7.5


def test_adaptive_figure_uses_family_groups_and_independent_component_scales() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = [
        {
            "epoch": epoch,
            "coherence_loss": total,
            "cross_spectrum.self_spectrum.auto_spectrum": total,
            "cross_spectrum.same_frequency.magnitude_squared": pair,
            "cross_spectrum.cross_frequency.band_energy_coupling": pair * 0.8,
        }
        for epoch, total, pair in ((1, 20.0, 2.0e-3), (2, 30.0, 1.5e-3))
    ]
    data = extract_coherence_history(rows, _config())

    figure = build_coherence_history_figure(data, plt, description="post:fixture", detailed=True)

    assert len(figure.axes) == 5  # separate total/family summaries + three component panels
    assert figure.axes[0].get_title(loc="left") == "Total coherence"
    assert figure.axes[1].get_title(loc="left") == "Weighted family contributions"
    component_titles = [axis.get_title(loc="left") for axis in figure.axes[2:] if axis.axison]
    assert component_titles == [
        "Self spectrum · Auto spectrum",
        "Same frequency · Magnitude squared",
        "Cross frequency · Band energy coupling",
    ]
    assert all(axis.get_yscale() == "log" for axis in (figure.axes[0], *figure.axes[2:]))
    assert figure.axes[1].get_yscale() == "linear"  # legacy rows have no family totals
    assert len(figure.subfigs) == 2
    assert figure.subfigs[1]._suptitle.get_text() == "Cross spectrum"
    plt.close(figure)


def test_family_summary_displays_all_weighted_families_in_stable_colors() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = []
    for epoch, multiplier in ((1, 1.0), (2, 0.8)):
        rows.append(
            {
                "epoch": epoch,
                "coherence_loss": 0.006 * multiplier,
                "coherence_family/global_distribution/weighted_contribution": 0.002 * multiplier,
                "coherence_family/cross_spectrum/weighted_contribution": 0.001 * multiplier,
                "coherence_family/topology/weighted_contribution": 0.003 * multiplier,
                "coherence_component/global_distribution/self.marginal_w2/raw": 0.5 * multiplier,
                "coherence_component/global_distribution/self.marginal_w2/weighted_contribution": 0.2 * multiplier,
                "coherence_component/cross_spectrum/same_frequency.magnitude_squared/raw": 0.1 * multiplier,
                "coherence_component/cross_spectrum/same_frequency.magnitude_squared/weighted_contribution": 0.1 * multiplier,
                "coherence_component/topology/self.persistence.h0/raw": 0.3 * multiplier,
                "coherence_component/topology/self.persistence.h0/weighted_contribution": 0.3 * multiplier,
            }
        )
    data = extract_coherence_history(rows, _config())
    figure = build_coherence_history_figure(data, plt, description="post:fixture", detailed=True)

    total, families = figure.axes[:2]
    assert [line.get_label() for line in total.lines] == ["Total coherence"]
    assert [line.get_label() for line in families.lines] == [
        "Global distribution",
        "Cross spectrum",
        "Topology",
    ]
    family_colors = {line.get_label(): line.get_color() for line in families.lines}
    assert family_colors["Global distribution"] == "#D95F59"
    assert family_colors["Cross spectrum"] == "#3B6EA8"
    assert family_colors["Topology"] == "#B58900"
    topology_axis = next(axis for axis in figure.axes if axis.get_title(loc="left") == "Self · Persistence · H0")
    assert "raw 2.40e-01 · weighted 2.40e-01" in [text.get_text() for text in topology_axis.texts]
    assert figure.get_size_inches()[1] < 10.0
    figure.canvas.draw()
    plt.close(figure)


def test_renderer_replays_legacy_run_without_model_or_dataset_loading(tmp_path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (tmp_path / "resolved_config.yaml").write_text(
        """
stage: post_training
model: {name: fixture}
coherence:
  families:
    cross_spectrum:
      enabled: true
      weight: 1.0
      components:
        self_spectrum: {enabled: true, weight: 1.0}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (metrics / "history.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "epoch": epoch,
                    "coherence_loss": value,
                    "cross_spectrum.self_spectrum.auto_spectrum": value,
                }
            )
            for epoch, value in ((1, 3.0), (2, 2.0))
        )
        + "\n",
        encoding="utf-8",
    )

    path = render_coherence_history(tmp_path)

    assert path == tmp_path / "logs/coherence_history.png"
    assert path.stat().st_size > 0
    assert (tmp_path / "logs/coherence_history.pdf").read_bytes().startswith(b"%PDF")
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    detailed = render_coherence_history(tmp_path, detailed=True)
    assert detailed == tmp_path / "logs/coherence_history_detailed.png"
    assert detailed.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    explicit_pdf = tmp_path / "custom.pdf"
    assert render_coherence_history(tmp_path, output_path=explicit_pdf) == explicit_pdf
    assert explicit_pdf.read_bytes().startswith(b"%PDF")
    with Image.open(tmp_path / "custom.png") as image:
        image.verify()


def test_live_monitor_writes_loss_and_coherence_figures_from_one_epoch_row(tmp_path) -> None:
    (tmp_path / "metrics").mkdir()
    (tmp_path / "resolved_config.yaml").write_text(
        """
stage: post_training
model: {name: fixture}
coherence:
  families:
    cross_spectrum:
      enabled: true
      weight: 1.0
      components:
        self_spectrum: {enabled: true, weight: 1.0}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monitor = TrainingMonitor(
        tmp_path,
        start_step=0,
        final_step=1,
        configured_steps=1,
        steps_per_epoch=1,
        description="post:fixture",
        enabled=False,
        plot_every_steps=1,
    )

    monitor.record(
        {
            "step": 1,
            "coherence_loss": 3.0,
            "coherence_component/cross_spectrum/self_spectrum.auto_spectrum/raw": 3.0,
            "coherence_component/cross_spectrum/self_spectrum.auto_spectrum/weighted_contribution": 3.0,
        }
    )
    monitor.close()

    assert (tmp_path / "logs/loss_history.png").stat().st_size > 0
    assert (tmp_path / "logs/coherence_history.png").stat().st_size > 0
    assert (tmp_path / "logs/coherence_history.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize(
    ("definition", "marginal_title", "cross_title"),
    [
        (
            "marginal_mixed_mean_v2",
            "Equal-node marginal W2² mean",
            "Equal-node mixed-projection W2² mean",
        ),
        (
            "marginal_mixed_weighted_v3",
            "Measure-weighted marginal W2² mean",
            "Measure-weighted mixed-projection W2² mean",
        ),
        ("global_distribution_v1", "Self · Marginal W2", "Cross · Joint topk SWD"),
    ],
)
def test_state_definition_labels_keep_legacy_keys_and_scale_style(
    definition, marginal_title, cross_title
):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    config = {
        "coherence": {
            "families": {
                "global_distribution": {"definition": definition},
                "topology": {},
                "multiscale_spectrum": {},
            }
        }
    }
    rows = [
        {
            "epoch": 1,
            "coherence_loss": 5.0,
            "total_semantics": "parameter_direction_without_scalar_objective",
            "coherence_family/global_distribution/weighted_contribution": 2.0,
            "coherence_family/topology/weighted_contribution": 1.0,
            "coherence_family/multiscale_spectrum/weighted_contribution": 2.0,
            "coherence_component/global_distribution/self.marginal_w2/raw": 0.5,
            "coherence_component/global_distribution/cross.joint_topk_swd/raw": 1.0,
            "coherence_component/topology/self.example/raw": 1.0,
            "coherence_component/multiscale_spectrum/self.example/raw": 2.0,
        }
    ]
    data = extract_coherence_history(rows, config)
    figure = build_coherence_history_figure(data, plt, description="post:test", detailed=True)
    titles = [axis.get_title(loc="left") for axis in figure.axes]
    assert marginal_title in titles and cross_title in titles
    assert {component.component for component in data.components} >= {
        "self.marginal_w2",
        "cross.joint_topk_swd",
    }
    assert figure.axes[0].get_title(loc="left") == "Coherence reporting sum"
    assert "not optimized scalar" in figure.axes[0].get_ylabel()
    assert figure.axes[0].lines[0].get_marker() == "o"
    lines = {line.get_label(): line for line in figure.axes[1].lines}
    assert lines["Multiscale spectrum"].get_color() == "#356b9a"
    assert lines["Multiscale spectrum"].get_color() != lines["Topology"].get_color()
    assert lines["Multiscale spectrum"].get_marker() == "o"
    plt.close(figure)


def test_mixed_scalar_and_config_history_is_labeled_as_mixed():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = [
        {"epoch": 1, "coherence_loss": 2.0, "global_distribution.self.marginal_w2": 2.0},
        {
            "epoch": 2,
            "coherence_loss": 1.0,
            "global_distribution.self.marginal_w2": 1.0,
            "total_semantics": "parameter_direction_without_scalar_objective",
        },
    ]
    figure = build_coherence_history_figure(
        extract_coherence_history(rows, _config()), plt, description="post:test", detailed=True
    )
    assert figure.axes[0].get_title(loc="left") == "Coherence values · includes reporting sums"
    assert figure.axes[0].get_ylabel() == "Recorded value"
    plt.close(figure)


def test_compact_sss_preserves_raw_leaves_and_separates_paired_state():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    config = {"coherence": {"families": {
        "global_distribution": {"definition": "conditional_quantile_fair_v1"},
        "multiscale_spectrum": {}, "topology": {},
    }}}
    row = {"epoch": 1, "coherence_loss": 9.0,
           "total_semantics": "parameter_direction_without_scalar_objective",
           "coherence_family/global_distribution/weighted_contribution": 1.0,
           "coherence_family/multiscale_spectrum/weighted_contribution": 2.0,
           "coherence_family/topology/weighted_contribution": 3.0,
           "coherence_component/global_distribution/self.marginal_w2/raw": 4.0,
           "coherence_component/global_distribution/cross.joint_topk_swd/raw": 5.0,
           "coherence_component/global_distribution/conditional_quantile_fair/raw": 6.0,
           "coherence_component/multiscale_spectrum/gram_self_energy_score.conditional/raw": 7.0,
           "coherence_component/multiscale_spectrum/gram_mutual_energy_score.conditional/raw": 8.0,
           "coherence_component/topology/self.persistence/raw": 1.0,
           "coherence_component/topology/mutual.persistence/raw": 0.0,
           "coherence_component/topology/self.persistence.field.h0/raw": 12.0}
    data = extract_coherence_history([row], config)
    assert len(data.components) == 8
    figure = build_coherence_history_figure(data, plt, description="post:test")
    assert len(figure.axes) == 4
    assert "not a scalar ConFIG objective" in figure.axes[0].get_title(loc="left")
    assert [line.get_label() for line in figure.axes[0].lines] == ["State", "Scale", "Shape"]
    assert "paired squared, separate from joint fair objective" in figure.axes[1].get_title(loc="left")
    assert sorted(line.get_ydata()[0] for line in figure.axes[1].lines) == [4.0, 5.0]
    assert sorted(line.get_ydata()[0] for line in figure.axes[3].lines) == [0.0, 1.0]
    plt.close(figure)
    detailed = build_coherence_history_figure(data, plt, description="post:test", detailed=True)
    assert len(detailed.axes) > len(figure.axes)
    assert any("Field" in axis.get_title(loc="left") for axis in detailed.axes)
    plt.close(detailed)
