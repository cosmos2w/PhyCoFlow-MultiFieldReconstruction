import json

import pytest

from phycoflow_reconstruction.training.journals import recover_journal


def test_checkpoint_replay_preserves_original_and_is_idempotent(tmp_path):
    path = tmp_path / "queries.jsonl"
    original = b'{"step":1,"x":0}\n{"step":2,"x":1}\n{"step":2,"x":2}\n{"step":3}\n{"step":'
    path.write_bytes(original)
    archive = recover_journal(path, 2)
    assert archive.read_bytes() == original
    assert [json.loads(x) for x in path.read_text().splitlines()] == [
        {"step": 1, "x": 0},
        {"step": 2, "x": 2},
    ]
    assert recover_journal(path, 2) is None
    assert len(list(tmp_path.glob("recovery/*"))) == 1


@pytest.mark.parametrize("raw", [b'{broken}\n{"step":2}\n', b'{"step":true}\n', b'{"step":1.5}\n'])
def test_invalid_committed_record_fails_before_mutation(tmp_path, raw):
    path = tmp_path / "records.jsonl"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        recover_journal(path, 1)
    assert path.read_bytes() == raw
    assert not list(tmp_path.glob("recovery/*"))


def test_named_cursor_and_missing_journal(tmp_path):
    path = tmp_path / "calibration.jsonl"
    assert recover_journal(path, 1) is None
    path.write_text('{"update":1}\n{"update":2}\n')
    assert recover_journal(path, 1, cursor_key="update").is_file()
    assert path.read_text() == '{"update":1}\n'
