"""Expose the original, already stride-4 Shu NPY arrays in the B HDF5 schema.

Only metadata is written: HDF5 external storage reads original NPY payloads,
and virtual datasets supply the canonical field/coordinate ordering. The
original NPY files must remain at their absolute paths. Open this view read-only.
No vorticity or coordinate payloads are copied or temporally subsampled.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def external_npy(container, name, path, expected_shape, storage_shape=None):
    """Describe a validated NPY payload without reading or writing its data."""
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY header version {version}: {path}")
        offset = stream.tell()
    if shape != expected_shape or fortran or dtype != np.dtype(np.float32) or not dtype.isnative:
        raise ValueError(f"expected native float32 C-order NPY {expected_shape}: {path}")
    size = int(np.prod(shape)) * dtype.itemsize
    if path.stat().st_size != offset + size:
        raise ValueError(f"unexpected NPY payload length: {path}")
    target_shape = storage_shape or shape
    if int(np.prod(target_shape)) != int(np.prod(shape)):
        raise ValueError("external storage shape must preserve every scalar")
    return container.create_dataset(
        name, shape=target_shape, dtype=dtype, external=[(str(path), offset, size)]
    )


def prepare(source, output):
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("output must be outside the immutable source package")
    if output.exists():
        raise FileExistsError(output)

    # Verify the package before interpreting its metadata or writing any data.
    checksums = {}
    for line in (source / "MD5SUMS").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        item = (source / relative).resolve(strict=True)
        if source not in item.parents or digest(item, "md5") != expected:
            raise ValueError(f"invalid package checksum or path: {relative}")
        checksums[relative] = expected
    required = {
        "metadata.json",
        "vorticity.npy",
        "x.npy",
        "y.npy",
        "source_frame_index.npy",
        "provenance/kolmogorov2d_manifest.json",
    }
    if not required.issubset(checksums):
        raise ValueError("package checksum manifest is missing required inputs")

    metadata = json.loads((source / "metadata.json").read_text())
    manifest = json.loads((source / "provenance/kolmogorov2d_manifest.json").read_text())
    values = np.load(source / "vorticity.npy", mmap_mode="r", allow_pickle=False)
    if values.shape != (40, 80, 256, 256) or values.dtype != np.float32:
        raise ValueError("expected float32 vorticity [40,80,256,256]")
    if metadata["shape"] != list(values.shape) or metadata["axes"] != [
        "trajectory",
        "frame",
        "y",
        "x",
    ]:
        raise ValueError("metadata shape or axis convention differs from the package")
    train = metadata["split"]["train_trajs"]
    heldout = metadata["split"]["test_trajs"]
    expected_order = np.random.default_rng(0).permutation(40)
    if (
        train != sorted(expected_order[:32].tolist())
        or heldout != sorted(expected_order[32:].tolist())
        or metadata["split"]["split_seed"] != 0
    ):
        raise ValueError("unexpected trajectory holdout; B requires split seed 0")
    order = train + heldout
    expected_mapping = {str(trajectory): block for block, trajectory in enumerate(order)}
    if metadata["h5_mapping"]["h5_block_of_traj"] != expected_mapping:
        raise ValueError("HDF5 block mapping disagrees with trajectory split")
    if (
        manifest["train_trajs"] != train
        or manifest["test_trajs"] != heldout
        or manifest["n_train_frames"] != 2560
        or manifest["n_total_frames"] != 3200
        or manifest["frames_per_traj"] != 80
        or manifest["stride"] != 4
        or manifest["field_names"] != ["vorticity"]
    ):
        raise ValueError("package provenance disagrees with the B dataset contract")
    if metadata["temporal_stride"] != 4 or not np.array_equal(
        np.load(source / "source_frame_index.npy", allow_pickle=False), np.arange(0, 320, 4)
    ):
        raise ValueError("expected 80 already-strided frames; do not subsample again")

    axis = 2.0 * np.pi * np.arange(256, dtype=np.float32) / 256
    for name in ("x", "y"):
        supplied = np.load(source / f"{name}.npy", allow_pickle=False)
        if supplied.dtype != np.float32 or not np.array_equal(supplied, axis):
            raise ValueError(f"{name} coordinates differ from the original benchmark converter")
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    coordinates = np.stack((xx.ravel(), yy.ravel(), np.full(65536, 0.5, np.float32)), axis=-1)
    coordinates = coordinates.reshape(65536, 1, 1, 3)
    time = np.arange(3200, dtype=np.float32)
    stats = manifest["train_stats"]["vorticity"]
    if stats != metadata["train_only_normalization"]:
        raise ValueError("normalization differs between package metadata and provenance")
    total = squared = 0.0
    count = 0
    for trajectory in range(40):
        block = np.asarray(values[trajectory], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError(f"nonfinite vorticity in trajectory {trajectory}")
        if trajectory in train:
            total += float(block.sum())
            squared += float((block * block).sum())
            count += block.size
    mean = total / count
    std = float(np.sqrt(max(squared / count - mean * mean, 1e-12)))
    if not np.allclose([mean, std], [stats["mean"], stats["std"]], rtol=1e-12, atol=1e-14):
        raise ValueError("recomputed train-only statistics disagree with package provenance")

    output.parent.mkdir(parents=True, exist_ok=True)
    # External storage and VDS mappings describe the original bytes; no source
    # dataset receives a write. Exclusive creation protects existing run inputs.
    with h5py.File(output, "w-") as result:
        external_npy(
            result,
            "_source/vorticity",
            source / "vorticity.npy",
            (40, 80, 256, 256),
            (40, 80, 65536),
        )
        external_npy(result, "_source/x", source / "x.npy", (256,))
        external_npy(result, "_source/y", source / "y.npy", (256,))
        original = h5py.VirtualSource(".", "/_source/vorticity", shape=(40, 80, 65536))
        fields = h5py.VirtualLayout(shape=(1, 3200, 65536, 1, 1, 1), dtype=np.float32)
        for block, trajectory in enumerate(order):
            fields[0, block * 80 : (block + 1) * 80, :, 0, 0, 0] = original[trajectory, :, :]
        result.create_virtual_dataset("fields", fields)
        original_x = h5py.VirtualSource(".", "/_source/x", shape=(256,))
        original_y = h5py.VirtualSource(".", "/_source/y", shape=(256,))
        coordinate_view = h5py.VirtualLayout(shape=(65536, 1, 1, 3), dtype=np.float32)
        for index in range(256):
            coordinate_view[index * 256 : (index + 1) * 256, 0, 0, 0] = original_x[:]
            coordinate_view[index::256, 0, 0, 1] = original_y[:]
        result.create_virtual_dataset("coordinates", coordinate_view, fillvalue=0.5)
        result["time"] = time
        result["conditions"] = np.zeros((1, 1), np.float32)
        result["trajectory_id"] = np.array([b"shu_stride4_all"])
        result["statistics/train_mean"] = np.array([stats["mean"]], np.float32)
        result["statistics/train_std"] = np.array([stats["std"]], np.float32)
        result.attrs["field_names"] = json.dumps(["vorticity"])
        result.attrs["field_units"] = json.dumps(["dimensionless"])
        result.attrs["grid_shape"] = json.dumps([256, 256])
        result.attrs["case_name"] = "kolmogorov_shu"
        result.attrs["source"] = str(source)
        result.attrs["layout"] = (
            "frames [0,2560) = 32 train trajectories, frames [2560,3200) = "
            "8 held-out trajectories; block split gap 0 with train_ratio "
            "0.8000 reproduces the protocol"
        )
        result.attrs["source_metadata_sha256"] = digest(source / "metadata.json")
        result.attrs["source_vorticity_md5"] = checksums["vorticity.npy"]
        result.attrs["source_h5_block_of_traj"] = json.dumps(expected_mapping)
        result.attrs["source_temporal_stride"] = 4
        result.attrs["storage"] = "metadata-only virtual view of immutable external NPY arrays"
        result.attrs["read_only_source_contract"] = (
            "open HDF5 view read-only; retain original absolute NPY paths"
        )

    # Reopen and compare every stored frame, not only a preview or a sample.
    with h5py.File(output, "r") as result:
        for block, trajectory in enumerate(order):
            stored = result["fields"][0, block * 80 : (block + 1) * 80, :, 0, 0, 0]
            if not np.array_equal(stored, values[trajectory].reshape(80, 65536)):
                raise ValueError(f"roundtrip mismatch in trajectory {trajectory}")
        for name, expected in (
            ("coordinates", coordinates),
            ("time", time),
            ("statistics/train_mean", np.array([stats["mean"]], np.float32)),
            ("statistics/train_std", np.array([stats["std"]], np.float32)),
        ):
            if not np.array_equal(result[name][:], expected):
                raise ValueError(f"roundtrip mismatch in {name}")
    # Verify that constructing/reading the view did not alter its external inputs.
    for relative in ("vorticity.npy", "x.npy", "y.npy"):
        if digest(source / relative, "md5") != checksums[relative]:
            raise ValueError(f"external source changed while preparing view: {relative}")
    return {
        "output": str(output),
        "sha256": digest(output),
        "file_size_bytes": output.stat().st_size,
        "storage": "external original NPY arrays plus virtual mappings",
        "frames_verified": 3200,
        "source_vorticity_md5": checksums["vorticity.npy"],
        "train_trajectories": train,
        "native_validation_trajectories": heldout[:4],
        "native_test_trajectories": heldout[4:],
        "mean": mean,
        "std": std,
        "additional_temporal_subsampling": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="the supplied kolmogorov2d_shu package directory"
    )
    parser.add_argument(
        "--out", required=True, help="new metadata-only HDF5 view path; must not exist"
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.out), indent=2))


if __name__ == "__main__":
    main()
