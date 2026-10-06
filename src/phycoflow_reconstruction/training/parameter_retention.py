"""Default-off L2 retention toward immutable trainable LIVE SOURCE parameters."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import random

import numpy as np
import torch

DEFINITION = "source_parameter_l2_v1"
CALIBRATION = "source_train_gradient_at_reference_displacement"
RANDOM_POLICY = "matched_native_v1"


def trainable_parameter_identity(model):
    """Hash the declared named trainable layout and native tensor bytes once."""
    layout, digest = [], hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        item = {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
        layout.append(item)
        digest.update(json.dumps(item, sort_keys=True).encode())
        digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    if not layout:
        raise ValueError("parameter retention requires trainable parameters")
    return {"names": [item["name"] for item in layout], "layout": layout,
            "tensor_sha256": digest.hexdigest()}


class SourceParameterRetention:
    """Frozen TRAIN calibration and one device-resident source snapshot."""

    def __init__(self, model, artifact, *, source_checkpoint_sha256):
        self.identity = trainable_parameter_identity(model)
        if artifact.get("definition") != DEFINITION or artifact.get("calibration") != CALIBRATION:
            raise ValueError("invalid parameter retention definition/calibration")
        if artifact.get("split") != "train":
            raise ValueError("parameter retention calibration must use TRAIN only")
        if artifact.get("source_checkpoint_sha256") != source_checkpoint_sha256:
            raise ValueError("parameter retention source checkpoint mismatch")
        if artifact.get("source_parameter_identity") != self.identity:
            raise ValueError("parameter retention source trainable layout/tensor mismatch")
        for key in ("lambda_sp", "native_gradient_median", "reference_displacement"):
            value = float(artifact.get(key, float("nan")))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"parameter retention {key} must be finite and positive")
        epsilon = float(artifact.get("epsilon_displacement", 1e-12))
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("parameter retention epsilon_displacement must be positive")
        expected = float(artifact["native_gradient_median"]) / max(float(artifact["reference_displacement"]), epsilon)
        if not math.isclose(float(artifact["lambda_sp"]), expected, rel_tol=1e-12):
            raise ValueError("parameter retention coefficient differs from fixed TRAIN equation")
        probes = artifact.get("train_probes", [])
        if len(probes) != 8 or not artifact.get("reference_parameter_sha256"):
            raise ValueError("parameter retention requires eight declared TRAIN probes and reference identity")
        if any(not probe.get("sample_ids") or "native_seed" not in probe or not probe.get("query_sha256")
               for probe in probes):
            raise ValueError("parameter retention TRAIN probes must bind samples, native draws and queries")
        self.artifact = json.loads(json.dumps(artifact))
        self.coefficient = float(artifact["lambda_sp"])
        self.parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        self.source_parameters = tuple(parameter.detach().clone() for parameter in self.parameters)

    def penalty(self):
        # Native complex modulus: gradient is lambda*(theta-theta_SOURCE).
        omega = 0.5 * sum((parameter - source).abs().square().sum()
                          for parameter, source in zip(self.parameters, self.source_parameters))
        return omega, self.coefficient * omega

    def state_dict(self):
        return {"definition": DEFINITION, "lambda_sp": self.coefficient,
                "source_parameter_identity": self.identity,
                "calibration_sha256": hashlib.sha256(
                    json.dumps(self.artifact, sort_keys=True).encode()).hexdigest()}

    def verify_resume(self, state):
        if state != self.state_dict():
            raise ValueError("resume parameter retention source/coefficient/calibration mismatch")


def scalar_objective(native_loss, coherence_loss=None, *, data_weight=0.1,
                     coherence_weight=1.0, retention=None):
    """Declared N/F/S equation; the source penalty is included exactly once."""
    total = float(data_weight) * native_loss
    if coherence_loss is not None:
        total = total + float(coherence_weight) * coherence_loss
    if retention is not None:
        _, penalty = retention.penalty()
        total = total + penalty
    return total


def scalar_update(model, optimizer, native_loss, coherence_loss=None, *, data_weight=0.1,
                  coherence_weight=1.0, retention=None, grad_clip=None):
    """Reuse the existing one-backward optimizer, clipping and unused layout."""
    from .gradient_balance import _scalar_objective_update
    total = scalar_objective(native_loss, coherence_loss, data_weight=data_weight,
                             coherence_weight=coherence_weight)
    retention_report = {}
    if retention is not None:
        omega, penalty = retention.penalty()
        total = total + penalty
        retention_report = {"parameter_retention_omega": float(omega.detach()),
                            "parameter_retention_loss": float(penalty.detach()),
                            "parameter_retention_lambda_sp": retention.coefficient}
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    report = _scalar_objective_update(parameters, optimizer, total, grad_clip)
    report["loss/scalar_objective"] = float(total.detach())
    report.update(retention_report)
    return report


@contextmanager
def private_rng(seed, device):
    """A declared private stream leaves public Python/NumPy/Torch state intact."""
    cpu, python, numpy = torch.get_rng_state(), random.getstate(), np.random.get_state()
    cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    try:
        # torch.manual_seed also reseeds CUDA even for a CPU-only private call.
        # Touch only the CPU generator here; CUDA is separately saved/restored.
        torch.random.default_generator.manual_seed(int(seed))
        if cuda:
            torch.cuda.manual_seed_all(int(seed))
        random.seed(int(seed))
        np.random.seed(int(seed) % 2**32)
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda:
            torch.cuda.set_rng_state_all(cuda)
        random.setstate(python)
        np.random.set_state(numpy)


def private_call(function, *args, seed, device, **kwargs):
    with private_rng(seed, device):
        return function(*args, **kwargs)
