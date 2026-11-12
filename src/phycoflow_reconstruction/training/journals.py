"""Recover append-only records against an atomic checkpoint cursor."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def _replace(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".journal-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def recover_journal(
    path: str | Path, completed_step: int, *, cursor_key: str = "step"
) -> Path | None:
    """Retain the last complete row per saved cursor; archive exact original bytes.

    Only a malformed final line is a recoverable torn write. Invalid earlier
    rows fail before any mutation. Records beyond the checkpoint are replayed
    by training; the immutable digest-named archive preserves their provenance.
    Returns that archive when recovery changed the journal, otherwise None.
    """
    if (
        isinstance(completed_step, bool)
        or not isinstance(completed_step, int)
        or completed_step < 0
    ):
        raise ValueError("checkpoint cursor must be a nonnegative integer")
    path = Path(path)
    if not path.exists():
        return None
    path = path.resolve()
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    rows: dict[int, bytes] = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == len(lines) - 1:
                break
            raise
        step = row.get(cursor_key) if isinstance(row, dict) else None
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"invalid journal cursor at row {index + 1}: {path}")
        if step <= completed_step:
            rows[step] = line.rstrip(b"\r\n") + b"\n"
    retained = b"".join(rows[step] for step in sorted(rows))
    if retained == original:
        return None
    digest = hashlib.sha256(original).hexdigest()
    archive_directory = path.parent / "recovery"
    archive_directory.mkdir(exist_ok=True)
    archive = archive_directory / f"{path.stem}.{digest}{path.suffix}"
    try:
        with archive.open("xb") as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if archive.read_bytes() != original:
            raise ValueError("existing journal recovery archive differs") from None
    _replace(path, retained)
    return archive
