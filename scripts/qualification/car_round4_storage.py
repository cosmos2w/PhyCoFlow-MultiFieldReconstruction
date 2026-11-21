#!/usr/bin/env python3
"""Time the actual full-size teacher recovery write, without model evaluation."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    args = p.parse_args()
    start, cpu = time.time(), time.process_time()
    import torch
    torch.set_num_threads(4)
    root = args.campaign.resolve(strict=True)
    manifest = json.loads((root/'configs/train_manifest.json').read_text())
    shape = (manifest['population'], 4, len(manifest['query_indices']), len(manifest['field_order']))
    output = root/'cache/SSS_Teacher_Recovery_IO_Qualification.pt'
    receipt_path = root/'operations/teacher_recovery_io.json'
    if output.exists() or receipt_path.exists():
        raise FileExistsError('recovery I/O qualification already exists')
    tensor = torch.zeros(shape, dtype=torch.float64)
    write_start = time.perf_counter()
    with output.open('xb') as stream:
        torch.save({'schema':'car_round4_recovery_io_qualification_v1', 'corrections':tensor}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    write_seconds = time.perf_counter()-write_start
    receipt = dict(schema='car_round4_teacher_recovery_io_v1',status='completed',
                   shape=list(shape),dtype='float64',array_bytes=tensor.numel()*tensor.element_size(),
                   file_bytes=output.stat().st_size,write_and_fsync_seconds=write_seconds,
                   wall_seconds=time.time()-start,process_cpu_seconds=time.process_time()-cpu,
                   start_unix=start,end_unix=time.time(),gpu_seconds=0.,
                   scope='CPU-only uncompressed full-size correction checkpoint write including fsync; no CUDA, model or scoring',
                   cache_file=str(output),teacher_checkpoint_writes=100)
    with receipt_path.open('x') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
