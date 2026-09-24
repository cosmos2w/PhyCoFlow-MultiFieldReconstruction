"""Re-render an automatic training preview through the canonical figure renderer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _renderer():
    """Import the project renderer from either an install or this checkout."""
    try:
        from phycoflow_reconstruction.training.preview import render_preview_payload
    except ModuleNotFoundError as error:
        if error.name != "phycoflow_reconstruction":
            raise
        project_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(project_root / "src"))
        from phycoflow_reconstruction.training.preview import render_preview_payload

    return render_preview_payload


def render(payload_path: Path, output_stem: Path, epoch: float) -> tuple[Path, ...]:
    """Render a saved preview payload without loading a model or checkpoint."""
    renderer = _renderer()
    return renderer(payload_path, output_stem=output_stem, epoch=epoch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path)
    parser.add_argument("--epoch", type=float)
    args = parser.parse_args()
    epoch = args.epoch
    if epoch is None:
        report_path = args.payload.with_name("latest_metrics.json")
        if not report_path.is_file():
            raise FileNotFoundError(
                "--epoch is required when latest_metrics.json is absent beside the payload"
            )
        epoch = float(json.loads(report_path.read_text())["training_epoch"])
    output_stem = args.output_stem or args.payload.with_suffix("")
    print("\n".join(str(path) for path in render(args.payload, output_stem, epoch)))


if __name__ == "__main__":
    main()
