"""Synthetic filesystem checks; no campaign tree is accessed."""
import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('relocation', Path(__file__).resolve().parents[1] / 'tools/storage/verify_campaign_relocation.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.fixture
def copied(tmp_path):
    src = tmp_path / 'source'; src.mkdir()
    (src / 'nested').mkdir()
    (src / 'nested' / 'data').write_bytes(b'content')
    os.link(src / 'nested' / 'data', src / 'hardlink')
    (src / 'link').symlink_to('nested/data')
    dst = tmp_path / 'target'
    subprocess.run(['cp', '-a', str(src), str(dst)], check=True)
    return src, dst


def test_valid_copy_and_alias_link_resolution(copied, tmp_path):
    src, dst = copied
    before, after = mod.inventory(src), mod.inventory(dst)
    assert mod.compare(before, after)['matched']
    assert after['entries']['hardlink']['hardlink_group'] == ['hardlink', 'nested/data']
    assert after['regular_file_logical_bytes_path_sum'] == 14
    alias = tmp_path / 'alias'; alias.symlink_to(dst, target_is_directory=True)
    links = mod.check_links(alias, after)
    assert links['all_resolved'] and links['links'][0]['target'] == 'nested/data'
    (dst / 'nested' / 'data').unlink()
    assert not mod.check_links(alias, after)['all_resolved']


@pytest.mark.parametrize('change', ['corrupt', 'missing', 'symlink', 'hardlink'])
def test_reject_changed_copy(copied, change):
    src, dst = copied
    baseline = mod.inventory(src)
    if change == 'corrupt':
        (dst / 'nested' / 'data').write_bytes(b'bad')
    elif change == 'missing':
        (dst / 'nested' / 'data').unlink()
    elif change == 'symlink':
        (dst / 'link').unlink(); (dst / 'link').symlink_to('hardlink')
    else:
        p = dst / 'hardlink'; contents = p.read_bytes(); st = p.stat()
        p.unlink(); p.write_bytes(contents); os.chmod(p, st.st_mode); os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert not mod.compare(baseline, mod.inventory(dst))['matched']


def test_detect_file_change_during_hash(tmp_path, monkeypatch):
    path = tmp_path / 'data'; path.write_bytes(b'initial')
    original = mod.os.read
    changed = False
    def changing(fd, size):
        nonlocal changed
        result = original(fd, size)
        if result and not changed:
            with path.open('ab') as out: out.write(b'changed')
            changed = True
        return result
    monkeypatch.setattr(mod.os, 'read', changing)
    with pytest.raises(RuntimeError, match='changed during hashing'):
        mod._hash_file(path, path.lstat())


def test_reject_receipt_inside_tree_and_nested_roots(copied):
    src, dst = copied
    with pytest.raises(ValueError, match='outside'):
        mod.validate_output(src / 'receipt.json', [src])
    with pytest.raises(ValueError, match='non-nested'):
        mod.reject_nested(src, src / 'nested')
    assert mod.validate_output(src.parent / 'receipt.json', [src, dst]) == src.parent / 'receipt.json'


def test_receipt_hash_detects_modification(copied, tmp_path):
    import json

    src, _ = copied
    value = mod.inventory(src)
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps(value))
    assert mod._read_receipt(receipt)['sha256'] == value['sha256']
    value['entries']['hardlink']['sha256'] = 'bad'
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='invalid receipt hash'):
        mod._read_receipt(receipt)
