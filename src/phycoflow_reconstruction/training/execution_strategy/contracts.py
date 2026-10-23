"""Small, serializable contracts for cost-aware execution decisions.

The records in this module describe the frozen scientific task and measured
whole-plan alternatives. They do not construct losses or change training math.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("contract values must not contain NaN or infinity")
        return value
    raise TypeError(f"contract value {type(value).__name__} is not JSON serializable")


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class TaskContract:
    """Identity of the unchanged scientific objective and training protocol.

    The explicit identities keep important bindings visible. ``metadata`` is
    for additional already-resolved contract details, not runtime discoveries.
    """

    task_id: str
    source_identity: str
    case_id: str
    dataset_id: str
    split_identity: str
    adapter_kind: str
    reconstruction_identity: str
    normalization_identity: str
    objective_identity: str
    objective_rule: str
    component_weights: tuple[tuple[str, float], ...]
    aggregation_rule: str
    source_calibration_identity: str
    c0: float | None
    field_bounds: tuple[tuple[str, float, float], ...]
    query_geometry_identity: str
    sampling_identity: str
    numerical_precision: str
    optimizer_identity: str
    epoch_definition: str
    quality_contract_identity: str
    max_epochs: int
    calibration_status: str = "bound"
    provider_ids: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.max_epochs, bool) or int(self.max_epochs) != self.max_epochs:
            raise ValueError("max_epochs must be an integer")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        required = (
            "task_id",
            "source_identity",
            "case_id",
            "dataset_id",
            "split_identity",
            "adapter_kind",
            "reconstruction_identity",
            "normalization_identity",
            "objective_identity",
            "objective_rule",
            "aggregation_rule",
            "source_calibration_identity",
            "query_geometry_identity",
            "sampling_identity",
            "numerical_precision",
            "optimizer_identity",
            "epoch_definition",
            "quality_contract_identity",
        )
        if any(not str(getattr(self, name)).strip() for name in required):
            raise ValueError("TaskContract identity fields must be non-empty")
        if self.adapter_kind not in {"deterministic", "generative"}:
            raise ValueError("adapter_kind must be 'deterministic' or 'generative'")
        if self.calibration_status not in {"pending", "bound"}:
            raise ValueError("calibration_status must be 'pending' or 'bound'")
        if self.calibration_status == "bound":
            if self.source_calibration_identity == "pending":
                raise ValueError("bound calibration needs a resolved source calibration identity")
            if self.c0 is None or not math.isfinite(float(self.c0)) or self.c0 <= 0:
                raise ValueError("bound calibration requires finite positive c0")
            if not self.field_bounds:
                raise ValueError("bound calibration requires SOURCE-derived field bounds")
        elif (
            self.source_calibration_identity != "pending"
            or self.c0 is not None
            or self.field_bounds
        ):
            raise ValueError(
                "pending calibration must use a pending identity and contain no C0 or field bounds"
            )
        names = [str(name) for name, _ in self.component_weights]
        if not names or len(names) != len(set(names)):
            raise ValueError("component_weights must have unique, non-empty names")
        if any(not name for name in names):
            raise ValueError("component weight names cannot be empty")
        if any(not math.isfinite(float(weight)) for _, weight in self.component_weights):
            raise ValueError("component weights must be finite")
        fields = [str(name) for name, _, _ in self.field_bounds]
        if len(fields) != len(set(fields)):
            raise ValueError("field_bounds must have unique field names")
        for name, bound, scale in self.field_bounds:
            if not str(name).strip() or not math.isfinite(float(bound)):
                raise ValueError("field bounds need a name and finite bound")
            if not math.isfinite(float(scale)) or float(scale) <= 0:
                raise ValueError("field bound scales must be finite and positive")
        if len(set(self.provider_ids)) != len(self.provider_ids):
            raise ValueError("provider_ids must be unique")
        if len(set(self.features)) != len(self.features):
            raise ValueError("features must be unique")
        _jsonable(self.metadata)

    @property
    def task_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self._payload()).encode("utf-8")).hexdigest()

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "source_identity": self.source_identity,
            "case_id": self.case_id,
            "dataset_id": self.dataset_id,
            "split_identity": self.split_identity,
            "adapter_kind": self.adapter_kind,
            "reconstruction_identity": self.reconstruction_identity,
            "normalization_identity": self.normalization_identity,
            "objective_identity": self.objective_identity,
            "objective_rule": self.objective_rule,
            "component_weights": [[name, float(weight)] for name, weight in self.component_weights],
            "aggregation_rule": self.aggregation_rule,
            "source_calibration_identity": self.source_calibration_identity,
            "c0": None if self.c0 is None else float(self.c0),
            "field_bounds": [
                [name, float(bound), float(scale)] for name, bound, scale in self.field_bounds
            ],
            "query_geometry_identity": self.query_geometry_identity,
            "sampling_identity": self.sampling_identity,
            "numerical_precision": self.numerical_precision,
            "optimizer_identity": self.optimizer_identity,
            "epoch_definition": self.epoch_definition,
            "quality_contract_identity": self.quality_contract_identity,
            "max_epochs": int(self.max_epochs),
            "calibration_status": self.calibration_status,
            "provider_ids": list(self.provider_ids),
            "features": list(self.features),
            "metadata": _jsonable(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["task_hash"] = self.task_hash
        return payload

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> TaskContract:
        data = dict(mapping)
        schema_version = data.get("schema_version", 1)
        if isinstance(schema_version, bool) or schema_version != 1:
            raise ValueError(f"unsupported TaskContract schema version {schema_version!r}")
        expected_hash = data.get("task_hash")
        data.pop("schema_version", None)
        data.pop("task_hash", None)
        calibration_status = data.get("calibration_status")
        calibration_is_bound = (
            data.get("c0") is not None
            and bool(data.get("field_bounds"))
            and bool(data.get("source_calibration_identity"))
            and data.get("source_calibration_identity") != "pending"
        )
        if calibration_status is None:
            calibration_status = "bound" if calibration_is_bound else "pending"
        if calibration_status == "pending":
            data["source_calibration_identity"] = "pending"
            data["c0"] = None
            data["field_bounds"] = ()
        else:
            data.setdefault("source_calibration_identity", "pending")
            data.setdefault("c0", None)
            data.setdefault("field_bounds", ())
        data["calibration_status"] = calibration_status
        if isinstance(data.get("component_weights"), Mapping):
            data["component_weights"] = tuple(data["component_weights"].items())
        elif "component_weights" in data:
            data["component_weights"] = tuple(tuple(item) for item in data["component_weights"])
        if isinstance(data.get("field_bounds"), Mapping):
            data["field_bounds"] = tuple(
                (name, *tuple(bounds)) for name, bounds in data["field_bounds"].items()
            )
        elif "field_bounds" in data:
            data["field_bounds"] = tuple(tuple(item) for item in data["field_bounds"])
        for key in ("provider_ids", "features"):
            if key in data:
                data[key] = tuple(data[key])
        task = cls(**data)
        if expected_hash is not None and expected_hash != task.task_hash:
            raise ValueError("serialized TaskContract hash does not match its frozen fields")
        return task


@dataclass(frozen=True)
class ExecutionCapability:
    """One registered execution action and its applicability decision."""

    action_id: str
    provider_id: str
    applicable: bool
    preservation_level: str
    assumptions: tuple[str, ...] = ()
    failure_conditions: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    shared_work_keys: tuple[str, ...] = ()
    output_scope: str = ""
    differentiable: bool = True
    estimator_support: str = "exact"
    numerical_oracle: str = ""
    cache_identity: str = ""
    evidence_status: str = "unmeasured"
    validated_workload_scope: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.action_id or not self.provider_id:
            raise ValueError("capability action_id and provider_id are required")
        if self.preservation_level not in {
            "equivalent",
            "unbiased_scalar_estimator",
            "approximate_direction",
            "unsupported",
        }:
            raise ValueError(f"unsupported preservation level {self.preservation_level!r}")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class PlanMeasurement:
    """One bounded, measured whole-plan candidate in a cost profile."""

    plan_id: str
    mean_epoch_cost_s: float
    upper_epoch_cost_s: float
    actions: tuple[str, ...] = ()
    acquisition_overhead_s: float = 0.0
    refresh_overhead_s: float = 0.0
    audit_overhead_s: float = 0.0
    extra_variance: float = 0.0
    variance_method: str = "none"
    estimator_kind: str = "equivalent"
    family_probabilities: Mapping[str, float] = field(default_factory=dict)
    within_family: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    family_support: tuple[tuple[float, tuple[str, ...]], ...] = ()
    support_validated: bool = True
    support_nonempty: bool = True
    measured_split: str = "TRAIN"
    evidence_status: str = "measured"
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("plan_id cannot be empty")
        for name in (
            "mean_epoch_cost_s",
            "upper_epoch_cost_s",
            "acquisition_overhead_s",
            "refresh_overhead_s",
            "audit_overhead_s",
            "extra_variance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.upper_epoch_cost_s < self.mean_epoch_cost_s:
            raise ValueError("upper epoch cost must be at least the measured mean")
        if self.estimator_kind not in {
            "equivalent",
            "unbiased_scalar_estimator",
            "approximate_direction",
            "unsupported",
        }:
            raise ValueError(f"unsupported estimator kind {self.estimator_kind!r}")
        if self.measured_split.upper() != "TRAIN":
            raise ValueError("candidate cost/noise evidence must use TRAIN only")
        for family, probability in self.family_probabilities.items():
            if (
                not str(family).strip()
                or not math.isfinite(float(probability))
                or not 0 < probability <= 1
            ):
                raise ValueError("family inclusion probabilities must be in (0, 1]")
        for family, specification in self.within_family.items():
            probability = float(specification.get("conditional_inclusion_probability", 1.0))
            if not math.isfinite(probability) or not 0 < probability <= 1:
                raise ValueError(f"within-family probability for {family!r} must be in (0, 1]")
            if not specification.get("provider_id") or not specification.get("estimator"):
                raise ValueError("within-family declarations need provider_id and estimator")
        if "whole_family_sampling" in self.actions and not self.family_probabilities:
            raise ValueError("whole-family sampling requires explicit family probabilities")
        if self.family_probabilities and "whole_family_sampling" not in self.actions:
            raise ValueError("family probabilities require the whole-family sampling action")
        if "generalized_filtration_sampling" in self.actions and not self.within_family:
            raise ValueError("generalized filtration sampling requires a within-family provider")
        if self.within_family and "generalized_filtration_sampling" not in self.actions:
            raise ValueError("within-family probabilities require generalized filtration sampling")
        if self.estimator_kind == "unbiased_scalar_estimator":
            if self.variance_method not in {
                "exact_support",
                "exact_joint_support",
                "paired_train_probe",
                "fixed_train_probe",
            }:
                raise ValueError(
                    "unbiased stochastic estimators require exact-support or TRAIN-probe variance evidence"
                )
            if not self.support_validated or not self.support_nonempty:
                raise ValueError("unbiased estimators require validated, non-empty support")
        if self.evidence_status not in {
            "measured",
            "validated",
            "unmeasured",
            "failed",
            "quarantined",
        }:
            raise ValueError(f"unknown candidate evidence status {self.evidence_status!r}")
        if (
            self.family_probabilities
            and not any(probability >= 1.0 for probability in self.family_probabilities.values())
            and not self.family_support
        ):
            raise ValueError(
                "family plan needs an always-included term or explicit non-empty joint support"
            )
        if self.family_support:
            total = 0.0
            expected_marginals = {name: 0.0 for name in self.family_probabilities}
            for mass, included in self.family_support:
                if not math.isfinite(float(mass)) or mass <= 0 or mass > 1:
                    raise ValueError("joint family support masses must be in (0, 1]")
                if not included or not set(included).issubset(expected_marginals):
                    raise ValueError(
                        "each joint support point must include known objective families"
                    )
                total += float(mass)
                for name in included:
                    expected_marginals[name] += float(mass)
            if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError("joint family support masses must sum to one")
            for name, expected in expected_marginals.items():
                if not math.isclose(
                    expected, float(self.family_probabilities[name]), rel_tol=1e-7, abs_tol=1e-9
                ):
                    raise ValueError(f"joint support does not match marginal for family {name!r}")

    @property
    def max_joint_importance_multiplier(self) -> float:
        multipliers = [1.0]
        for family, specification in self.within_family.items():
            outer = float(self.family_probabilities.get(family, 1.0))
            inner = float(specification.get("conditional_inclusion_probability", 1.0))
            multipliers.append(1.0 / (outer * inner))
        for probability in self.family_probabilities.values():
            multipliers.append(1.0 / float(probability))
        return max(multipliers)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self)) | {
            "max_joint_importance_multiplier": self.max_joint_importance_multiplier
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> PlanMeasurement:
        data = dict(mapping)
        data.pop("max_joint_importance_multiplier", None)
        data["actions"] = tuple(data.get("actions", ()))
        data["family_support"] = tuple(
            (float(mass), tuple(included)) for mass, included in data.get("family_support", ())
        )
        return cls(**data)


@dataclass(frozen=True)
class CostProfile:
    """TRAIN-only whole-plan costs, uncertainty, and estimator-noise evidence."""

    task_hash: str
    hardware_fingerprint: str
    profiled_epoch: int
    max_epochs: int
    measurements: tuple[PlanMeasurement, ...]
    intrinsic_noise_variance: float
    noise_split: str = "TRAIN"
    profile_split: str = "TRAIN"
    profile_overhead_s: float = 0.0
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("unsupported CostProfile schema version")
        if not self.task_hash or not self.hardware_fingerprint:
            raise ValueError("cost profiles require task and hardware identities")
        if any(
            isinstance(value, bool) or int(value) != value
            for value in (self.profiled_epoch, self.max_epochs)
        ):
            raise ValueError("profiled_epoch and max_epochs must be integers")
        if self.profiled_epoch < 0 or self.max_epochs < 1 or self.profiled_epoch > self.max_epochs:
            raise ValueError("invalid profile epoch horizon")
        if self.profile_split.upper() != "TRAIN" or self.noise_split.upper() != "TRAIN":
            raise ValueError("adaptive cost and noise profiles must use TRAIN only")
        if (
            not math.isfinite(float(self.intrinsic_noise_variance))
            or self.intrinsic_noise_variance < 0
        ):
            raise ValueError("intrinsic noise variance must be finite and non-negative")
        if not math.isfinite(float(self.profile_overhead_s)) or self.profile_overhead_s < 0:
            raise ValueError("profile overhead must be finite and non-negative")
        if not 1 <= len(self.measurements) <= 6:
            raise ValueError("profile must contain between one and six measured plan candidates")
        ids = [item.plan_id for item in self.measurements]
        if len(ids) != len(set(ids)):
            raise ValueError("profile plan IDs must be unique")
        if "full" not in ids:
            raise ValueError("profile must measure the full/no-action plan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_hash": self.task_hash,
            "hardware_fingerprint": self.hardware_fingerprint,
            "profiled_epoch": self.profiled_epoch,
            "max_epochs": self.max_epochs,
            "measurements": [measurement.to_dict() for measurement in self.measurements],
            "intrinsic_noise_variance": float(self.intrinsic_noise_variance),
            "noise_split": self.noise_split,
            "profile_split": self.profile_split,
            "profile_overhead_s": float(self.profile_overhead_s),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> CostProfile:
        data = dict(mapping)
        data["measurements"] = tuple(
            item if isinstance(item, PlanMeasurement) else PlanMeasurement.from_mapping(item)
            for item in data["measurements"]
        )
        return cls(**data)


@dataclass
class ExecutionState:
    """Recovery binding and past-TRAIN summaries for policy refreshes."""

    task_hash: str
    plan_id: str
    rng_seed: int
    accepted_cursor: int = 0
    accepted_epoch: int = 0
    last_refresh_epoch: int | None = None
    last_full_audit_epoch: int | None = None
    charged_overhead_s: float = 0.0
    train_observations: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("unsupported ExecutionState schema version")
        if not self.task_hash or not self.plan_id:
            raise ValueError("execution state requires task and plan bindings")
        if (
            isinstance(self.rng_seed, bool)
            or int(self.rng_seed) != self.rng_seed
            or self.rng_seed < 0
        ):
            raise ValueError("rng_seed must be a non-negative integer")
        if any(
            isinstance(value, bool) or int(value) != value
            for value in (self.accepted_cursor, self.accepted_epoch)
        ):
            raise ValueError("accepted cursor and epoch must be integers")
        if self.accepted_cursor < 0 or self.accepted_epoch < 0:
            raise ValueError("accepted cursor and epoch must be non-negative")
        if any(
            value is not None and (isinstance(value, bool) or int(value) != value or value < 0)
            for value in (self.last_refresh_epoch, self.last_full_audit_epoch)
        ):
            raise ValueError("refresh and full-audit epochs must be non-negative integers")
        if not math.isfinite(float(self.charged_overhead_s)) or self.charged_overhead_s < 0:
            raise ValueError("charged overhead must be finite and non-negative")
        if any(str(item.get("split", "")).upper() != "TRAIN" for item in self.train_observations):
            raise ValueError("execution state may contain adaptive observations from TRAIN only")
        for item in self.train_observations:
            if int(item.get("epoch", -1)) < 0:
                raise ValueError("TRAIN execution observations need a non-negative epoch")
            cost = float(item.get("cost_s", -1))
            if not math.isfinite(cost) or cost < 0:
                raise ValueError(
                    "TRAIN execution cost observations must be finite and non-negative"
                )
            if "noise_variance" in item:
                variance = float(item["noise_variance"])
                if not math.isfinite(variance) or variance < 0:
                    raise ValueError(
                        "TRAIN execution noise observations must be finite and non-negative"
                    )

    @classmethod
    def for_plan(
        cls,
        task: TaskContract,
        plan: ExecutionPlan,
        *,
        rng_seed: int,
        accepted_cursor: int = 0,
        accepted_epoch: int = 0,
    ) -> ExecutionState:
        if plan.task_hash != task.task_hash:
            raise ValueError("cannot bind state to a plan for a different task")
        return cls(task.task_hash, plan.plan_id, int(rng_seed), accepted_cursor, accepted_epoch)

    @property
    def rng_identity(self) -> str:
        raw = f"execution-counter-v1:{self.task_hash}:{self.plan_id}:{int(self.rng_seed)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def counter_seed(self, stream: str, *, accepted_cursor: int | None = None) -> int:
        cursor = self.accepted_cursor if accepted_cursor is None else int(accepted_cursor)
        if (
            isinstance(accepted_cursor, bool)
            or (accepted_cursor is not None and int(accepted_cursor) != accepted_cursor)
            or cursor < 0
            or not isinstance(stream, str)
            or not stream
        ):
            raise ValueError("counter stream and non-negative cursor are required")
        raw = f"{self.rng_identity}:{cursor}:{stream}"
        return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:8], "big") % (2**63)

    def observe_train(
        self,
        *,
        epoch: int,
        cost_s: float,
        noise_variance: float | None = None,
        split: str = "TRAIN",
    ) -> None:
        if split.upper() != "TRAIN":
            raise ValueError(
                "held-out observations cannot drive adaptation; TRAIN observations are required"
            )
        if (
            isinstance(epoch, bool)
            or int(epoch) != epoch
            or epoch < 0
            or not math.isfinite(float(cost_s))
            or cost_s < 0
        ):
            raise ValueError("TRAIN observation needs a valid epoch and cost")
        record: dict[str, Any] = {"epoch": int(epoch), "cost_s": float(cost_s), "split": "TRAIN"}
        if noise_variance is not None:
            if not math.isfinite(float(noise_variance)) or noise_variance < 0:
                raise ValueError("noise variance must be finite and non-negative")
            record["noise_variance"] = float(noise_variance)
        self.train_observations.append(record)
        if len(self.train_observations) > 64:
            del self.train_observations[:-64]
        self.accepted_epoch = max(self.accepted_epoch, int(epoch))

    def commit_accepted_update(self, *, cursor_before_update: int) -> None:
        if (
            isinstance(cursor_before_update, bool)
            or int(cursor_before_update) != cursor_before_update
            or cursor_before_update != self.accepted_cursor
        ):
            raise ValueError("accepted update cursor does not match recovery state")
        self.accepted_cursor += 1

    def validate_resume(
        self,
        *,
        task_hash: str,
        plan_id: str,
        rng_identity: str | None = None,
        accepted_cursor: int | None = None,
    ) -> None:
        if task_hash != self.task_hash or plan_id != self.plan_id:
            raise ValueError("execution recovery task/plan binding mismatch")
        if rng_identity is not None and rng_identity != self.rng_identity:
            raise ValueError("execution RNG identity mismatch")
        if accepted_cursor is not None and (
            isinstance(accepted_cursor, bool)
            or int(accepted_cursor) != accepted_cursor
            or accepted_cursor != self.accepted_cursor
        ):
            raise ValueError("execution accepted cursor mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_hash": self.task_hash,
            "plan_id": self.plan_id,
            "rng_seed": int(self.rng_seed),
            "rng_identity": self.rng_identity,
            "accepted_cursor": int(self.accepted_cursor),
            "accepted_epoch": int(self.accepted_epoch),
            "last_refresh_epoch": self.last_refresh_epoch,
            "last_full_audit_epoch": self.last_full_audit_epoch,
            "charged_overhead_s": float(self.charged_overhead_s),
            "train_observations": _jsonable(self.train_observations),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ExecutionState:
        data = dict(mapping)
        data.pop("rng_identity", None)
        state = cls(**data)
        if mapping.get("rng_identity", state.rng_identity) != state.rng_identity:
            raise ValueError("serialized execution RNG identity is inconsistent")
        return state


@dataclass(frozen=True)
class ExecutionPlan:
    """Chosen measured plan, its cost/noise case, and explicit rejected options."""

    task_hash: str
    plan_id: str
    selected_actions: tuple[str, ...]
    family_probabilities: Mapping[str, float]
    within_family: Mapping[str, Mapping[str, Any]]
    family_support: tuple[tuple[float, tuple[str, ...]], ...]
    predicted_total_s: float
    predicted_full_total_s: float
    cost_savings_ratio: float
    extra_variance: float
    variance_ceiling: float
    max_joint_importance_multiplier: float
    no_action: bool
    selected_reason: str
    rejection_reasons: Mapping[str, str] = field(default_factory=dict)
    refresh_interval_epochs: int = 10
    audit_interval_epochs: int = 20
    accepted_cursor: int = 0
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("unsupported ExecutionPlan schema version")
        if not self.task_hash or not self.plan_id or not self.selected_reason:
            raise ValueError("plan requires task, id, and decision reason")
        values = (
            self.predicted_total_s,
            self.predicted_full_total_s,
            self.cost_savings_ratio,
            self.extra_variance,
            self.variance_ceiling,
            self.max_joint_importance_multiplier,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
            raise ValueError(
                "execution plan costs, variance, and ratios must be finite and non-negative"
            )
        if self.refresh_interval_epochs < 10 or self.audit_interval_epochs < 20:
            raise ValueError("policy refresh/audit intervals cannot be shorter than 10/20 epochs")
        if any(
            isinstance(value, bool) or int(value) != value
            for value in (
                self.refresh_interval_epochs,
                self.audit_interval_epochs,
                self.accepted_cursor,
            )
        ):
            raise ValueError("plan intervals and accepted cursor must be integers")
        if self.accepted_cursor < 0:
            raise ValueError("plan accepted cursor must be non-negative")
        if self.no_action and self.plan_id != "full":
            raise ValueError("no-action plans must use plan_id='full'")
        if not self.no_action and self.plan_id == "full":
            raise ValueError("accelerated plans must use a measured non-full plan_id")
        if self.max_joint_importance_multiplier > 4.0 + 1e-12:
            raise ValueError("nested importance multiplier exceeds the version-1 ceiling of four")
        if self.no_action and (
            self.selected_actions or self.family_probabilities or self.within_family
        ):
            raise ValueError("no-action plans cannot declare sampling actions or probabilities")
        if not self.no_action:
            if not self.selected_actions:
                raise ValueError("accelerated plans must select at least one registered action")
            if self.cost_savings_ratio + 1e-12 < 0.05:
                raise ValueError(
                    "selected accelerated plan does not meet the 5% net savings threshold"
                )
            if self.predicted_total_s >= self.predicted_full_total_s:
                raise ValueError("selected accelerated plan does not beat measured full execution")
            if self.extra_variance > self.variance_ceiling + 1e-12:
                raise ValueError("selected plan exceeds its serialized TRAIN-noise ceiling")
        for family, probability in self.family_probabilities.items():
            if (
                not str(family).strip()
                or not math.isfinite(float(probability))
                or not 0 < probability <= 1
            ):
                raise ValueError("execution-plan family probabilities must be in (0, 1]")
        for family, specification in self.within_family.items():
            probability = float(specification.get("conditional_inclusion_probability", 1.0))
            if not math.isfinite(probability) or not 0 < probability <= 1:
                raise ValueError(f"within-family probability for {family!r} must be in (0, 1]")
            if not specification.get("provider_id") or not specification.get("estimator"):
                raise ValueError("within-family plan entries need provider_id and estimator")
        if self.family_probabilities:
            if (
                not any(value >= 1.0 for value in self.family_probabilities.values())
                and not self.family_support
            ):
                raise ValueError(
                    "family plan needs an always-included term or explicit non-empty joint support"
                )
            if "whole_family_sampling" not in self.selected_actions:
                raise ValueError("family probabilities require the registered whole-family action")
        if self.within_family and "generalized_filtration_sampling" not in self.selected_actions:
            raise ValueError("within-family plan requires its registered provider action")
        if "whole_family_sampling" in self.selected_actions and not self.family_probabilities:
            raise ValueError("whole-family action requires explicit family probabilities")
        if "generalized_filtration_sampling" in self.selected_actions and not self.within_family:
            raise ValueError("generalized filtration action requires a within-family declaration")
        if self.family_support:
            total = 0.0
            marginals = {name: 0.0 for name in self.family_probabilities}
            for mass, included in self.family_support:
                if not math.isfinite(float(mass)) or mass <= 0 or mass > 1:
                    raise ValueError("plan joint support masses must be in (0, 1]")
                if not included or not set(included).issubset(marginals):
                    raise ValueError("plan joint support must have non-empty known-family outcomes")
                total += float(mass)
                for family in included:
                    marginals[family] += float(mass)
            if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError("plan joint support masses must sum to one")
            for family, probability in self.family_probabilities.items():
                if not math.isclose(
                    marginals[family], float(probability), rel_tol=1e-7, abs_tol=1e-9
                ):
                    raise ValueError(f"plan joint support disagrees with the {family!r} marginal")
        computed_multiplier = 1.0
        for family, probability in self.family_probabilities.items():
            computed_multiplier = max(computed_multiplier, 1.0 / float(probability))
            specification = self.within_family.get(family)
            if specification is not None:
                inner = float(specification.get("conditional_inclusion_probability", 1.0))
                computed_multiplier = max(computed_multiplier, 1.0 / (float(probability) * inner))
        for family, specification in self.within_family.items():
            if family not in self.family_probabilities:
                inner = float(specification.get("conditional_inclusion_probability", 1.0))
                computed_multiplier = max(computed_multiplier, 1.0 / inner)
        if not math.isclose(
            computed_multiplier,
            float(self.max_joint_importance_multiplier),
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError(
                "serialized max importance multiplier does not match the plan probabilities"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_hash": self.task_hash,
            "plan_id": self.plan_id,
            "selected_actions": list(self.selected_actions),
            "family_probabilities": _jsonable(self.family_probabilities),
            "within_family": _jsonable(self.within_family),
            "family_support": [[mass, list(names)] for mass, names in self.family_support],
            "predicted_total_s": float(self.predicted_total_s),
            "predicted_full_total_s": float(self.predicted_full_total_s),
            "cost_savings_ratio": float(self.cost_savings_ratio),
            "extra_variance": float(self.extra_variance),
            "variance_ceiling": float(self.variance_ceiling),
            "max_joint_importance_multiplier": float(self.max_joint_importance_multiplier),
            "no_action": bool(self.no_action),
            "selected_reason": self.selected_reason,
            "rejection_reasons": _jsonable(self.rejection_reasons),
            "refresh_interval_epochs": self.refresh_interval_epochs,
            "audit_interval_epochs": self.audit_interval_epochs,
            "accepted_cursor": self.accepted_cursor,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ExecutionPlan:
        data = dict(mapping)
        data["selected_actions"] = tuple(data.get("selected_actions", ()))
        data["family_support"] = tuple(
            (float(mass), tuple(names)) for mass, names in data.get("family_support", ())
        )
        return cls(**data)

    def validate_for_task(self, task: TaskContract) -> None:
        """Validate a deserialized plan against the matching declared registry."""

        if self.task_hash != task.task_hash:
            raise ValueError("execution plan is bound to a different task contract")
        from .providers import action_registry

        capabilities = {item.action_id: item for item in action_registry(task)}
        for action_id in self.selected_actions:
            capability = capabilities.get(action_id)
            if capability is None or not capability.applicable:
                reason = "unregistered" if capability is None else capability.reason
                raise ValueError(
                    f"execution plan selects ineligible action {action_id!r}: {reason}"
                )
        if task.calibration_status != "bound":
            raise ValueError("execution plan cannot bind to a task with pending SOURCE calibration")
        if self.within_family and task.objective_rule != "scalar_additive":
            raise ValueError("within-family sampling is incompatible with this objective rule")
        if self.family_probabilities and "whole_family_sampling" not in self.selected_actions:
            raise ValueError("execution plan family probabilities have no whole-family action")
        if "whole_family_sampling" in self.selected_actions and not self.family_probabilities:
            raise ValueError("execution plan whole-family action has no family probabilities")
        if self.within_family:
            from .providers import TOPOLOGY_FILTRATION_PROVIDER_ID

            for family, specification in self.within_family.items():
                if family not in self.family_probabilities and self.family_probabilities:
                    raise ValueError(
                        f"within-family entry {family!r} is not an included objective family"
                    )
                if specification.get("provider_id") != TOPOLOGY_FILTRATION_PROVIDER_ID:
                    raise ValueError("execution plan uses an unregistered filtration provider")
                if TOPOLOGY_FILTRATION_PROVIDER_ID not in task.provider_ids:
                    raise ValueError("task contract does not bind the selected filtration provider")
                if specification.get("estimator") != "polarity":
                    raise ValueError("execution plan uses an unsupported filtration estimator")
                probability = float(specification.get("conditional_inclusion_probability", 0.0))
                if not math.isclose(probability, 0.5, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError("the registered polarity estimator requires conditional p=0.5")

    def to_execution_mapping(
        self, *, accepted_cursor: int | None = None, rng_seed: int = 0
    ) -> dict[str, Any]:
        """Return the generic dispatch payload for one accepted-update cursor.

        Family decisions use a stateless counter draw, so a rejected update can
        retry the same inclusion outcome. The sampler gets its own independent
        counter seed and never consumes model/data RNG state.
        """
        cursor = self.accepted_cursor if accepted_cursor is None else int(accepted_cursor)
        if (
            isinstance(accepted_cursor, bool)
            or (accepted_cursor is not None and int(accepted_cursor) != accepted_cursor)
            or cursor < 0
        ):
            raise ValueError("accepted_cursor must be non-negative")
        if isinstance(rng_seed, bool) or int(rng_seed) != rng_seed or rng_seed < 0:
            raise ValueError("rng_seed must be a non-negative integer")
        family_inclusions = _draw_family_support(
            self.task_hash,
            self.plan_id,
            int(rng_seed),
            cursor,
            self.family_probabilities,
            self.family_support,
        )
        within: dict[str, dict[str, Any]] = {}
        for family, specification in self.within_family.items():
            provider_id = str(specification["provider_id"])
            estimator = str(specification["estimator"])
            seed = _counter_seed(self.task_hash, self.plan_id, int(rng_seed), cursor, provider_id)
            within[family] = {
                **_jsonable(specification),
                "provider_id": provider_id,
                "estimator": estimator,
                "seed": seed,
            }
        return {
            "task_hash": self.task_hash,
            "plan_id": self.plan_id,
            "selected_actions": list(self.selected_actions),
            "family_inclusions": family_inclusions,
            "family_probabilities": _jsonable(self.family_probabilities),
            "within_family": within,
            "accepted_cursor": cursor,
            "execution_rng_seed": int(rng_seed),
            "max_joint_importance_multiplier": float(self.max_joint_importance_multiplier),
            "dense_fidelity_required": True,
        }


def _counter_seed(task_hash: str, plan_id: str, rng_seed: int, cursor: int, stream: str) -> int:
    raw = f"execution-counter-v1:{task_hash}:{plan_id}:{int(rng_seed)}:{int(cursor)}:{stream}"
    return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:8], "big") % (2**63)


def _draw_family_support(
    task_hash: str,
    plan_id: str,
    rng_seed: int,
    cursor: int,
    probabilities: Mapping[str, float],
    support: tuple[tuple[float, tuple[str, ...]], ...],
) -> dict[str, bool]:
    if not probabilities:
        return {}
    if support:
        value = _counter_seed(task_hash, plan_id, rng_seed, cursor, "family-support") / float(2**63)
        running = 0.0
        selected = support[-1][1]
        for mass, included in support:
            running += float(mass)
            if value < running:
                selected = included
                break
        included_set = set(selected)
        return {family: family in included_set for family in probabilities}
    result: dict[str, bool] = {}
    for family, probability in probabilities.items():
        value = _counter_seed(task_hash, plan_id, rng_seed, cursor, f"family:{family}") / float(
            2**63
        )
        result[family] = value < float(probability)
    if result and not any(result.values()):
        # Independent marginals without a joint non-empty design are unsafe.
        # The planner prevents this case; keep a defensive error at dispatch.
        raise ValueError(
            "family draw produced an empty objective; provide a verified joint support"
        )
    return result


def policy_refresh_due(state: ExecutionState, epoch: int, *, interval_epochs: int = 10) -> bool:
    if interval_epochs < 10:
        raise ValueError("policy refresh interval must be at least 10 epochs")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    return state.last_refresh_epoch is None or epoch - state.last_refresh_epoch >= interval_epochs


def full_audit_due(state: ExecutionState, epoch: int, *, interval_epochs: int = 20) -> bool:
    if interval_epochs < 20:
        raise ValueError("full-gradient audit interval must be at least 20 epochs")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    return (
        state.last_full_audit_epoch is None
        or epoch - state.last_full_audit_epoch >= interval_epochs
    )
