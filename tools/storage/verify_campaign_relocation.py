"""Read-only inventory, comparison and post-alias link checks for quiescent trees.

This tool does not establish quiescence, copy, move, relink, or delete anything.
Receipts must be written outside all inventoried trees.
"""
import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


def _content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _seal(value):
    return {**value, 'sha256': _content_hash(value)}


def _read_receipt(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != _content_hash({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'invalid receipt hash: {path}')
    return value


def _signature(st):
    return (st.st_dev, st.st_ino, st.st_mode, st.st_uid, st.st_gid, st.st_size,
            st.st_mtime_ns, st.st_ctime_ns, st.st_nlink)


def _hash_file(path, before):
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    try:
        if _signature(os.fstat(fd)) != _signature(before):
            raise RuntimeError(f'file changed before hashing: {path}')
        digest = hashlib.sha256()
        while block := os.read(fd, 1024 * 1024):
            digest.update(block)
        if _signature(os.fstat(fd)) != _signature(before) or _signature(path.lstat()) != _signature(before):
            raise RuntimeError(f'file changed during hashing: {path}')
        return digest.hexdigest()
    finally:
        os.close(fd)


def inventory(root):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('root must be a directory')
    records, groups = {}, {}
    logical = allocated = 0

    def visit(path, relative):
        nonlocal logical, allocated
        before = path.lstat()
        key = relative.as_posix() if relative.parts else '.'
        mode = before.st_mode
        kind = ('file' if stat.S_ISREG(mode) else 'directory' if stat.S_ISDIR(mode)
                else 'symlink' if stat.S_ISLNK(mode) else 'unsupported')
        if kind == 'unsupported':
            raise ValueError(f'unsupported special file: {path}')
        record = {'type': kind, 'mode': stat.S_IMODE(mode), 'uid': before.st_uid,
                  'gid': before.st_gid, 'mtime_ns': before.st_mtime_ns, 'size': before.st_size}
        records[key] = record
        if kind == 'file':
            record['sha256'] = _hash_file(path, before)
            groups.setdefault((before.st_dev, before.st_ino), []).append((key, before.st_nlink))
            logical += before.st_size
            allocated += before.st_blocks * 512
        elif kind == 'symlink':
            groups.setdefault((before.st_dev, before.st_ino), []).append((key, before.st_nlink))
            record['target'] = os.readlink(path)
            if _signature(path.lstat()) != _signature(before):
                raise RuntimeError(f'symlink changed during inventory: {path}')
        else:
            with os.scandir(path) as entries:
                names = sorted(entry.name for entry in entries)
            for name in names:
                visit(path / name, relative / name)
            if _signature(path.lstat()) != _signature(before):
                raise RuntimeError(f'directory changed during inventory: {path}')

    visit(root, Path())
    # Physical allocation is counted once per inode, including external links.
    unique_allocated = 0
    for members in groups.values():
        paths = sorted(p for p, _ in members)
        nlink = members[0][1]
        external = nlink - len(paths)
        if external < 0 or any(n != nlink for _, n in members):
            raise RuntimeError('hardlink count changed during inventory')
        for p in paths:
            records[p]['hardlink_group'] = paths
            records[p]['external_hardlinks'] = external
        if records[paths[0]]['type'] == 'file':
            unique_allocated += (root / paths[0]).lstat().st_blocks * 512
    return _seal({'schema': 'campaign_tree_inventory_v1', 'root': str(root), 'entries': records,
            'regular_file_logical_bytes_path_sum': logical,
            'regular_file_allocated_bytes_path_sum': allocated,
            'regular_file_allocated_bytes_unique_inode': unique_allocated,
            'boundary': 'Read-only walk without following symlinks; caller must establish quiescence.'})


def reject_nested(a, b):
    a, b = Path(a).resolve(strict=True), Path(b).resolve(strict=True)
    if a == b or a.is_relative_to(b) or b.is_relative_to(a):
        raise ValueError('source and target must be distinct and non-nested')


def compare(source, target):
    reject_nested(source['root'], target['root'])
    left, right = source['entries'], target['entries']
    issues = []
    for path in sorted(left.keys() | right.keys()):
        if path not in left or path not in right:
            issues.append({'path': path, 'difference': 'missing from source' if path not in left else 'missing from target'})
        else:
            keys = left[path].keys() | right[path].keys()
            if left[path]['type'] == right[path]['type'] == 'directory':
                keys = keys - {'size'}  # Directory storage size is filesystem-dependent.
            diff = {k: {'source': left[path].get(k), 'target': right[path].get(k)}
                    for k in sorted(keys) if left[path].get(k) != right[path].get(k)}
            if diff:
                issues.append({'path': path, 'difference': diff})
    return {'schema': 'campaign_tree_comparison_v1', 'source': source['root'], 'target': target['root'],
            'matched': not issues, 'differences': issues,
            'boundary': 'Compares content, metadata and hardlink topology; directory size and allocation are filesystem-dependent and reported separately.'}


def check_links(alias, inv):
    alias = Path(alias)
    if not alias.is_dir() or alias.resolve(strict=True) != Path(inv['root']).resolve(strict=True):
        raise ValueError('alias must resolve to the inventoried target root')
    links = []
    for rel, record in inv['entries'].items():
        if record['type'] == 'symlink':
            path = alias / rel
            if not path.is_symlink() or os.readlink(path) != record['target']:
                raise ValueError(f'link changed after inventory: {rel}')
            try:
                resolved = str(path.resolve(strict=True)); error = None
            except (FileNotFoundError, RuntimeError, OSError) as exc:
                resolved = None; error = str(exc)
            links.append({'path': rel, 'target': record['target'], 'resolved': resolved, 'error': error})
    return {'schema': 'campaign_alias_links_v1', 'alias': str(alias.absolute()), 'target': inv['root'],
            'all_resolved': all(link['error'] is None for link in links), 'links': links,
            'boundary': 'Resolution alone does not prove an external target is the intended dependency.'}


def validate_output(output, roots):
    output = Path(output).resolve(strict=False)
    for root in roots:
        root = Path(root).resolve(strict=True)
        if output == root or output.is_relative_to(root):
            raise ValueError('receipt must be outside inventoried trees')
    if output.exists():
        raise FileExistsError('preserve existing receipt')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inv = sub.add_parser('inventory'); inv.add_argument('root', type=Path); inv.add_argument('--output', type=Path, required=True)
    comp = sub.add_parser('compare'); comp.add_argument('source', type=Path); comp.add_argument('target', type=Path); comp.add_argument('--output', type=Path, required=True)
    links = sub.add_parser('links'); links.add_argument('alias', type=Path); links.add_argument('inventory', type=Path); links.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'inventory':
        output = validate_output(args.output, [args.root]); result = inventory(args.root)
    elif args.command == 'compare':
        source, target = (_read_receipt(p) for p in (args.source, args.target))
        output = validate_output(args.output, [source['root'], target['root']]); result = compare(source, target)
        result.update(source_inventory_sha256=source['sha256'], target_inventory_sha256=target['sha256'])
    else:
        value = _read_receipt(args.inventory); output = validate_output(args.output, [value['root']]); result = check_links(args.alias, value)
        result['inventory_sha256'] = value['sha256']
    if args.command != 'inventory':
        result = _seal(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True); handle.write('\n')
    if not result.get('matched', result.get('all_resolved', True)):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
