#!/usr/bin/env python3
"""Profile one actual Round 6-style trainer epoch; the caller owns the GPU lease.

The synchronized scopes are inclusive and perturb execution. The trainer's
epoch receipt supplies the enclosing epoch and callback times. This diagnostic
does not establish a production speedup or change the scientific recipe.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import wraps
import hashlib
import json
from pathlib import Path
import time

import torch
import yaml

import phycoflow_reconstruction.coherence.shape_compute as shape
import phycoflow_reconstruction.training.fidelity_budget as fidelity
import phycoflow_reconstruction.training.post_training as training
from phycoflow_reconstruction.training.multitask import OuterAggregator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    config = yaml.safe_load(args.config.read_text())
    config['runtime']['device'] = args.device
    config['output']['experiment_name'] = args.experiment_name
    steps = int(config['optimization']['steps_per_epoch'])
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('the complete-epoch profile requires an available CUDA device')
    timings = defaultdict(list)
    originals = []

    def instrument(owner, name, label):
        original = getattr(owner, name)
        originals.append((owner, name, original))

        @wraps(original)
        def measured(*positional, **keyword):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            try:
                return original(*positional, **keyword)
            finally:
                torch.cuda.synchronize(device)
                timings[label].append(time.perf_counter() - started)
        setattr(owner, name, measured)

    instrument(training, '_coherence_objective', 'coherence_including_rollout_and_shape')
    instrument(training, 'differentiable_reconstruction', 'rollout_forward')
    instrument(shape, 'cubical_diagrams', 'shape_pairing_and_transfers')
    instrument(fidelity, '_flat_gradient', 'family_and_constraint_backward')
    instrument(OuterAggregator, '_aggregate', 'config_solve_and_diagnostics')
    instrument(torch.linalg, 'pinv', 'wide_pseudoinverse')
    instrument(fidelity.ExternalPolicy, 'update_with_coherence_aggregation', 'complete_update')
    source_factory = training.build_training_batch_source
    originals.append((training, 'build_training_batch_source', source_factory))

    class TimedSource:
        def __init__(self, source):
            self.source = source

        def __getattr__(self, name):
            return getattr(self.source, name)

        def __iter__(self):
            iterator = iter(self.source)
            while True:
                torch.cuda.synchronize(device)
                before = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
                finally:
                    torch.cuda.synchronize(device)
                timings['batch_wait_and_assembly'].append(time.perf_counter() - before)
                yield batch

    def timed_source(*positional, **keyword):
        return TimedSource(source_factory(*positional, **keyword))

    training.build_training_batch_source = timed_source
    started = time.time()
    receipt = {
        'schema': 'car_round6_computation_profile_v1', 'status': 'failed',
        'config': str(args.config), 'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'device': args.device, 'device_name': torch.cuda.get_device_name(device),
        'steps_per_epoch': steps, 'started_unix': started,
        'measurement_limits': 'Synchronized inclusive scopes overlap and include checkpoint recomputation in backward. Shape scope includes CPU pairing and transfers together. Batch wait includes assembly and transfer. Setup may contribute to scope call counts. Synchronization perturbs timing.',
    }
    try:
        run = training.run_post_training(config, case_dir=args.case_dir, max_steps=steps)
        epoch_file = Path(run) / 'logs' / 'epoch_timing.jsonl'
        epochs = [json.loads(line) for line in epoch_file.read_text().splitlines() if line.strip()]
        if not epochs or epochs[-1]['attempted_updates'] != steps:
            raise RuntimeError('trainer did not record the requested complete epoch')
        receipt.update(status='completed', run_dir=str(run), epoch_timing=epochs[-1])
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
        receipt['wall_seconds'] = time.time() - started
        receipt['scopes'] = {name: {'calls': len(values), 'seconds': sum(values)} for name, values in timings.items()}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()
