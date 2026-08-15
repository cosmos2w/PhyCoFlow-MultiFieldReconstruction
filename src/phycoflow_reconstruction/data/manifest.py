"""Persist and reload deterministic sensor selections with dataset provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..contracts import FieldSample, ObservationBatch
from .sensor_protocols import SensorProtocol, build_observation_batch


@dataclass(frozen=True)
class SensorManifest:
    dataset_path: str
    dataset_fingerprint: str
    split: str
    protocol: dict[str, Any]
    indices: dict[str, list[list[int]]]
    version: str = "3"

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**asdict(self), "manifest_sha256": self.digest()}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> SensorManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        digest = payload.pop("manifest_sha256")
        manifest = cls(**payload)
        if manifest.digest() != digest:
            raise ValueError("sensor manifest checksum mismatch")
        return manifest


def dataset_fingerprint(path: str | Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    # Keep the identity portable across equivalent relinks/copies. This is a
    # fast structural fingerprint (size plus file boundary bytes), not a claim
    # of a full multi-gigabyte content checksum; catalog READMEs record full
    # checksums where they are available.
    digest = hashlib.sha256(f"v2:{stat.st_size}".encode())
    with path.open("rb") as stream:
        digest.update(stream.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            stream.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(stream.read(1024 * 1024))
    return digest.hexdigest()


def manifest_from_batch(
    batch: ObservationBatch, dataset_path: str | Path, split: str
) -> SensorManifest:
    if batch.obs_indices is None:
        raise ValueError("manifest creation requires observation indices")
    indices: dict[str, list[list[int]]] = {}
    for batch_index, sample_id in enumerate(batch.sample_ids):
        valid = batch.obs_valid_mask[batch_index]
        indices[sample_id] = [
            [int(point), int(field)]
            for point, field in zip(
                batch.obs_indices[batch_index, valid].tolist(),
                batch.obs_field_ids[batch_index, valid].tolist(),
            )
        ]
    return SensorManifest(
        # Catalog-relative identity keeps a frozen manifest portable across
        # equivalent local links; the content fingerprint remains authoritative.
        dataset_path=Path(dataset_path).name,
        dataset_fingerprint=dataset_fingerprint(dataset_path),
        split=split,
        protocol=dict(batch.metadata["protocol"]),
        indices=indices,
    )


def build_batch_from_manifest(
    samples: list[FieldSample],
    manifest: SensorManifest,
    dataset_path: str | Path,
    *,
    query_points: int | None = None,
) -> ObservationBatch:
    """Rebuild observations exactly and reject a manifest from other data."""
    if dataset_fingerprint(dataset_path) != manifest.dataset_fingerprint:
        raise ValueError("sensor manifest dataset fingerprint does not match the requested payload")
    protocol = SensorProtocol(**manifest.protocol)
    expected_ids = {
        f"{sample.trajectory_id}:{sample.time_index if sample.time_index is not None else 'all'}"
        for sample in samples
    }
    missing = sorted(expected_ids - manifest.indices.keys())
    if missing:
        raise KeyError(f"sensor manifest has no entries for samples {missing}")
    return build_observation_batch(
        samples,
        protocol,
        query_points=query_points,
        manifest_indices=manifest.indices,
    )
