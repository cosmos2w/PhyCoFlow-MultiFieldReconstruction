"""Reject duplicate report entrances and loose files in current reading libraries."""
import argparse
from pathlib import Path


def check_layout(root):
    root = Path(root)
    issues = []
    for path in sorted(root.iterdir()):
        loose_output = path.is_file() and path.suffix.lower() in {
            ".png", ".pdf", ".html", ".csv", ".jsonl", ".log", ".tmp",
            ".npy", ".npz", ".pt", ".pth", ".h5", ".hdf5", ".pkl", ".whl",
        }
        report_alias = path.is_symlink() and (
            path.name.lower().endswith(("_review", "_report"))
            or (path / "report.md").is_file() or (path / "report.pdf").is_file()
        )
        if (loose_output or report_alias or path.match("Coherence_Adaptive_Repair_Round*_Review")
                or path.name in {"report.md", "report.pdf", "report.html",
                                 "report_receipt.json", "page_previews"}
                or (path.name in {"figures", "reports", "LegacyCoherence"}
                    and path.is_symlink())):
            issues.append(f"{path.name}: retired/generated entrance belongs in its archive or campaign")
    for name in ("_Plans", "_Reports"):
        library = root / "UpdatePlans" / name
        if not library.is_dir():
            issues.append(f"UpdatePlans/{name}: reading library is missing")
            continue
        for path in sorted(library.iterdir()):
            if path.name == "README.md" and path.is_file() and not path.is_symlink():
                continue
            if path.is_dir() and not path.is_symlink():
                continue
            issues.append(f"{path.relative_to(root)}: use an ordinary campaign directory or _bk archive")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    issues = check_layout(args.root)
    for issue in issues:
        print(issue)
    print(f"Workspace navigation: {len(issues)} issues")
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
