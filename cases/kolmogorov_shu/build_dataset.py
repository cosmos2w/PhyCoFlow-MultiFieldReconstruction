"""Write the FlowReconBench Shu Kolmogorov file in this repository's canonical schema.

Fields, coordinates and time are copied unchanged: one stored trajectory of
3200 frames, [1, 3200, 65536, 1, 1, 1]. With one trajectory and no stored
splits, the loader's chronological 80/10/10 frame split gives
train = frames [0, 2560) = the 32 FlowReconBench training trajectories,
validation = frames [2560, 2880) and test = frames [2880, 3200), i.e. the
8 held-out trajectories in source order. Statistics are the FlowReconBench
training-split mean and std.
"""

import argparse
import json

import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--manifest", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

manifest = json.load(open(args.manifest))
stats = manifest["train_stats"]["vorticity"]
with h5py.File(args.source, "r") as src, h5py.File(args.out, "w-") as dst:
    fields = src["fields"]
    if fields.shape != (1, 3200, 65536, 1, 1, 1):
        raise ValueError(f"unexpected source layout {fields.shape}")
    out = dst.create_dataset("fields", shape=fields.shape, dtype=np.float32,
                             chunks=(1, 1, 65536, 1, 1, 1))
    for start in range(0, fields.shape[1], 200):
        out[0, start:start + 200] = fields[0, start:start + 200]
    dst["coordinates"] = src["coordinates"][:]
    dst["time"] = src["time"][:]
    dst["conditions"] = np.zeros((1, 1), np.float32)
    dst["trajectory_id"] = np.array([b"shu_stride4_all"])
    dst["statistics/train_mean"] = np.array([stats["mean"]], np.float32)
    dst["statistics/train_std"] = np.array([stats["std"]], np.float32)
    dst.attrs["field_names"] = json.dumps(["vorticity"])
    dst.attrs["field_units"] = json.dumps(["dimensionless"])
    dst.attrs["grid_shape"] = json.dumps([256, 256])
    dst.attrs["case_name"] = "kolmogorov_shu"
    dst.attrs["source"] = args.source
    dst.attrs["layout"] = src.attrs["layout"]
print("wrote", args.out)
