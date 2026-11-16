"""One saved-SOURCE-row residual-head numerical qualification; no SOURCE rollout."""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

PROCESS_START_UNIX = time.time()
PROCESS_START_CPU = time.process_time()

import torch
import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT / "src"))

from phycoflow_reconstruction.contracts import ObservationBatch
from phycoflow_reconstruction.models.source_residual import FrozenSourceResidual
from phycoflow_reconstruction.training.source import load_source_model, source_checkpoint_path


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def select_source_member(members, expected_members, member_index):
    if members.ndim != 4 or members.shape[:3] != (expected_members, 1, 40300):
        raise ValueError("SOURCE row must have shape [J,1,40300,C] matching the bank receipt")
    if not 0 <= member_index < members.shape[0]:
        raise ValueError("member index is outside the genuine SOURCE members")
    return members[member_index:member_index + 1, 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--member-index", type=int, default=0)
    p.add_argument("--head-checkpoint", type=Path)
    args = p.parse_args()
    started_wall, started_cpu = PROCESS_START_UNIX, PROCESS_START_CPU
    torch.set_num_threads(4)
    config = yaml.safe_load(args.config.read_text())
    receipt_path = args.bank / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    expected_members = {"senseiver": 1, "ffm": 4}
    family = receipt.get("family")
    if (receipt.get("schema") != "car_round2_native_bank_v1" or receipt.get("status") != "completed"
            or receipt.get("kind") != "source" or receipt.get("split") != "train"
            or family not in expected_members or receipt.get("members") != expected_members[family]
            or receipt.get("native_points") != 40300):
        raise ValueError("requires a terminal native TRAIN SOURCE bank with the declared family member count")
    if not 0 <= args.row < len(receipt["files_sha256"]):
        raise ValueError("row index is outside the completed bank")
    row_path = args.bank / f"{args.row:04d}.pt"
    row_sha = file_hash(row_path)
    if row_sha != receipt["files_sha256"][args.row]:
        raise ValueError("SOURCE row does not match completed bank receipt")
    row = torch.load(row_path, map_location="cpu", weights_only=True)
    members = row["members_model"]
    values = select_source_member(members, expected_members[family], args.member_index)
    checkpoint_path = source_checkpoint_path(config)
    if file_hash(checkpoint_path) != receipt["checkpoint_sha256"]:
        raise ValueError("SOURCE checkpoint differs from bank receipt")
    source, dataset, identity = load_source_model(config, torch.device("cpu"))
    before = tensor_digest(source.state_dict())
    residual = config["residual"]
    torch.manual_seed(32061)
    wrapper = FrozenSourceResidual(source, coordinate_dim=residual["coordinate_dim"],
                                   field_scales=residual["field_scales"], width=residual["width"],
                                   max_fraction=residual["max_fraction"]).eval()
    if args.head_checkpoint:
        checkpoint = torch.load(args.head_checkpoint, map_location="cpu", weights_only=False)
        wrapper.load_state_dict(checkpoint["model"], strict=True)
        head_kind = "trained checkpoint"
    else:
        with torch.no_grad():
            wrapper.head[-1].weight.normal_(0, 0.03)
            wrapper.head[-1].bias.copy_(torch.linspace(-0.15, 0.15, len(residual["field_scales"])))
        head_kind = "synthetic nonzero head perturbation; no trained checkpoint"
    coordinates = row["coordinates_model"]
    count, channels = values.shape[1:]
    ids = torch.arange(count).unsqueeze(0)
    obs = row["obs_indices"].long()
    batch = ObservationBatch(obs_coords=coordinates.gather(1, obs[..., None].expand(-1, -1, coordinates.shape[-1])),
                             obs_values=row["obs_values"], obs_field_ids=row["obs_fields"].long(),
                             obs_valid_mask=torch.ones_like(obs, dtype=torch.bool), query_coords=coordinates,
                             query_valid_mask=torch.ones_like(ids, dtype=torch.bool), target_fields=None,
                             sample_ids=(row["sample_id"],), obs_indices=obs, metadata={"query_indices": ids})
    batch.validate()
    params = tuple(p for p in wrapper.parameters() if p.requires_grad)
    field_coeff = torch.linspace(0.5, 1.5, channels).reshape(1, 1, channels)
    site_coeff = (0.5 + (ids % 97).float() / 97).unsqueeze(-1)

    def subset(indices):
        return replace(batch, query_coords=batch.query_coords[:, indices],
                       query_valid_mask=batch.query_valid_mask[:, indices],
                       metadata={"query_indices": batch.metadata["query_indices"][:, indices]})

    def run(parts):
        wrapper.zero_grad(set_to_none=True)
        outputs = []
        for a, b in parts:
            out = wrapper.correction(subset(slice(a, b)), values[:, a:b])
            (out * field_coeff * site_coeff[:, a:b]).sum().div(count * channels).backward()
            outputs.append(out.detach())
        return torch.cat(outputs, 1), [p.grad.detach().clone() for p in params]

    full, full_grad = run([(0, count)])
    chunk_results = {}
    for chunk in (4096, 3071):
        actual, gradients = run([(a, min(a + chunk, count)) for a in range(0, count, chunk)])
        chunk_results[str(chunk)] = {"output_max_abs": float((actual - full).abs().max()),
                                     "gradient_max_abs": max(float((a-b).abs().max()) for a,b in zip(gradients, full_grad)),
                                     "output_allclose": torch.allclose(actual, full, rtol=1e-5, atol=1e-6),
                                     "gradients_allclose": all(torch.allclose(a, b, rtol=2e-4, atol=2e-6) for a,b in zip(gradients, full_grad))}
    selected = torch.arange(min(4096, count))
    with torch.no_grad():
        q4096 = wrapper.correction(subset(selected), values[:, selected])
        points = sorted({0, count // 2, count - 1, *obs[0, :8].tolist()})
        alone_error = max(float((wrapper.correction(subset(torch.tensor([i])), values[:, i:i+1]) - full[:, i:i+1]).abs().max()) for i in points)
        sensor = full[0, obs[0], row["obs_fields"][0].long()]
        sensor_max = float(sensor.abs().max())
        magnitude = float(full.abs().max())
    after = tensor_digest(source.state_dict())
    dataset.close()
    result = {"schema": "car_round2_native_head_chunk_microcase_v1", "scope": "one completed TRAIN condition, numerical execution microcase; no population performance claim or SOURCE rollout", "family": family, "member_index": args.member_index, "source_members": members.shape[0], "sample_id": row["sample_id"], "point_count": count, "field_order": receipt["field_order"], "units": "model field units", "device": "cpu", "dtype": "float32", "source_identity": identity,
              "bindings": {"row_sha256": row_sha, "bank_receipt_sha256": file_hash(receipt_path), "config_sha256": file_hash(args.config), "source_snapshot_receipt_sha256": file_hash(Path(config["source"]["snapshot_receipt"])), "source_checkpoint_sha256": file_hash(checkpoint_path), "head_checkpoint_sha256": file_hash(args.head_checkpoint) if args.head_checkpoint else None, "script_sha256": file_hash(Path(__file__))},
              "head": head_kind, "synthetic_head_seed": 32061 if not args.head_checkpoint else None, "head_state_sha256": tensor_digest({k:v for k,v in wrapper.state_dict().items() if not k.startswith("source.")}),
              "output_tolerance": {"rtol": 1e-5, "atol_model_units": 1e-6}, "gradient_tolerance": {"rtol": 2e-4, "atol": 2e-6}, "full_correction_max_abs": magnitude,
              "native_chunks": chunk_results, "q4096_max_abs": float((q4096 - full[:, selected]).abs().max()), "alone_max_abs": alone_error, "alone_indices": points, "sensor_null_max_abs": sensor_max, "source_state_sha256_before": before, "source_state_sha256_after": after,
              "timer": {"start_unix": started_wall, "end_unix": time.time(), "cpu_seconds": time.process_time() - started_cpu}}
    result["passed"] = (all(v["output_allclose"] and v["gradients_allclose"] for v in chunk_results.values()) and result["q4096_max_abs"] <= 1e-6 and alone_error <= 1e-6 and sensor_max == 0 and before == after and magnitude > 0)
    result.update(start_unix=started_wall, end_unix=result["timer"]["end_unix"],
                  wall_seconds=result["timer"]["end_unix"] - started_wall,
                  status="completed" if result["passed"] else "failed", independent_cpu=True,
                  timing_scope="CPU process from before torch and application imports through completed calculations; excludes interpreter startup and receipt serialization")
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    if not result["passed"]:
        raise SystemExit("head chunk contract failed; see saved result")


if __name__ == "__main__":
    main()
