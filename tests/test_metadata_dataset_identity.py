"""Split-locked operators must not read whole-file boundary payloads."""

import pytest

from phycoflow_reconstruction.data import manifest


def test_attested_identity_never_opens_data_and_rejects_mutation(tmp_path, monkeypatch):
    data = tmp_path / "fields.h5"
    data.write_bytes(b"train-and-heldout-payload")
    stat = data.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    manifest.register_dataset_identity(data, "a" * 64, identity)
    with monkeypatch.context() as scoped:
        scoped.setattr(type(data), "open", lambda *a, **kw: pytest.fail("dataset byte read"))
        assert manifest.dataset_fingerprint(data) == "a" * 64
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        manifest.dataset_fingerprint(data)


def test_registration_rejects_stale_stat_and_conflicting_binding(tmp_path):
    data = tmp_path / "fields.h5"
    data.write_bytes(b"payload")
    stat = data.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    with pytest.raises(ValueError, match="identity changed"):
        manifest.register_dataset_identity(data, "a" * 64, (0, 0, 0, 0))
    manifest.register_dataset_identity(data, "a" * 64, identity)
    with pytest.raises(ValueError, match="conflicting"):
        manifest.register_dataset_identity(data, "b" * 64, identity)
