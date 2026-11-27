from copy import deepcopy

import numpy as np
import pytest

from scripts.analysis.car_round8_child_relations import SCALE_KEYS, summarize, validate_identities
from scripts.analysis.car_round8_relation_parity import compare


def ref(name):
    return {"path": "/unused/" + name, "sha256": name}


def identity_fixture():
    ids = [f"frame-{i:03}" for i in range(100)]
    spec = {name: ref(name) for name in ("panel", "core", "core_config", "core_manifest", "expanded_bank",
                                        "prefit", "assessment_contract", "source_checkpoint")}
    assessment = {"schema": "car_round7_matched_assessment_v1", "status": "completed", "split": "validation",
                  "members": 4, "test_locked": True, "sample_ids": ids, "field_order": ["temperature", "pressure"],
                  "field_units": ["unknown", "unknown"], "physical_mask": "positive physical weights",
                  "source_native": ref("source-native"), "source_selected": ref("source-selected"),
                  "source_baseline": ref("baseline"), "child_selected": ref("child-selected"),
                  "spec": ref("spec"), "per_condition_scores": {"SOURCE": [{}] * 100, "candidate": [{}] * 100}}
    historical = deepcopy(assessment)
    cached = {"schema": "car_round8_support_readout_v1", "status": "completed", "split": "validation",
              "members": 4, "test_locked": True, "sample_ids": ids, "field_order": assessment["field_order"],
              "inputs": {"receipts": {"SOURCE": {"native": ref("source-native"), "selected": ref("source-selected")}}},
              "records": [{"sample_id": sid} for sid in ids]}
    decomposition = {"status": "completed", "split": "validation", "members": 4, "sample_ids": ids,
                     "field_order": assessment["field_order"], "spec": ref("spec"),
                     "source_receipt": ref("source-selected"), "child_receipt": ref("child-selected"),
                     "source_baseline": ref("baseline"), "records": [{"sample_id": sid} for sid in ids]}
    return assessment, cached, historical, spec, deepcopy(spec), decomposition, {"sample_ids": ids}


def test_identity_validation_refuses_a_different_observer_or_source_bank():
    fixture = identity_fixture()
    validate_identities(*fixture)
    changed = list(deepcopy(fixture))
    changed[3]["core"] = ref("different-core")
    with pytest.raises(ValueError, match="frozen observer"):
        validate_identities(*changed)
    changed = list(deepcopy(fixture))
    changed[0]["source_selected"] = ref("different-source")
    with pytest.raises(ValueError, match="SOURCE"):
        validate_identities(*changed)
    changed = list(deepcopy(fixture))
    changed[5]["records"][0]["sample_id"] = "unmatched"
    with pytest.raises(ValueError, match="row order"):
        validate_identities(*changed)


def test_paired_state_uses_raw_and_frozen_train_units():
    def scale(value):
        output = {name: [[value, value + 1]] for name in SCALE_KEYS}
        output.update(reference_power_over_floor=[[8., 9.]], reference_low_energy=[[0., 0.]])
        return output
    records = []
    for shift in (0., 3., 7.):
        records.append({"models": {
            "SOURCE": {"state": [4.+shift, 1.+shift, 3.], "scale": scale(2.+shift)},
            "candidate": {"state": [5.+shift, 1.5+shift, 3.5], "scale": scale(3.+shift)},
        }})
    result = summarize(records, normalizer=2., replicates=100)
    assert np.allclose(result["state"]["paired_delta_raw"]["mean"], [1., .5, .5])
    assert np.allclose(result["state"]["paired_delta_frozen_train_units"]["mean"], [.5, .25, .25])
    assert np.allclose(result["scale"]["rho_signed_error_mean_member"]["paired_delta"]["mean"], [[1., 1.]])
    with pytest.raises(ValueError, match="normalizer"):
        summarize(records, normalizer=0.)


def test_parity_detects_a_changed_context_value():
    scale = {key: [[1., 2.]] for key in SCALE_KEYS}
    scale.update(reference_power_over_floor=[[8., 9.]], reference_low_energy=[[0., 0.]],
                 field_pair_indices=[[0, 1]])
    model = {"state": [4., 1., 3.], "scale": scale}
    candidate = {"status": "completed", "sample_ids": ["frame-000"], "field_order": ["x", "y"],
                 "members": 4, "split": "validation", "inputs": {"source_readout": ref("source")},
                 "records": [{"sample_id": "frame-000", "models": {"SOURCE": deepcopy(model),
                                                                       "candidate": deepcopy(model)}}]}
    reference = deepcopy(candidate)
    reference["records"][0]["models"]["JointAnchor500"] = reference["records"][0]["models"].pop("candidate")
    assert max(compare(candidate, reference, reference_label="JointAnchor500").values()) == 0.
    reference["records"][0]["models"]["JointAnchor500"]["scale"]["rho_signed_error_mean_member"][0][0] += .1
    with pytest.raises(ValueError, match="readout parity failed"):
        compare(candidate, reference, reference_label="JointAnchor500")
