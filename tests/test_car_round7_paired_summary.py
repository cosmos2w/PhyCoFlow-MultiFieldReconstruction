import copy

import pytest

from scripts.analysis.car_round7_paired_summary import BLOCKS, summarize_scale


def test_paired_scale_summary_reconstructs_and_rejects_changed_components():
    weights = {"scale." + name: 1.0 for name in BLOCKS}
    def model(score):
        block = {"attraction": score + 2.0, "separation": 2.0, "score": score}
        return {"blocks": {name: dict(block) for name in BLOCKS},
                "definitions": {"Published_Core_Default": score, "Complete_Core_Prefit": score,
                                "Complete_Expanded_Prefit": 4.25 * score},
                "localization": {"joint_piece_field_attribution": {key: [val] for key, val in block.items()}}}
    doc = {"sample_ids": ["one", "two"], "split": "validation", "members": 4,
           "prefit_coefficients": weights,
           "records": [{"sample_id": name, "models": {"SOURCE": model(1), "candidate": model(2)}}
                       for name in ("one", "two")]}
    result = summarize_scale(doc, "candidate", 2, bootstrap=20)
    assert result["definitions"]["Complete_Expanded_Prefit"]["mean"] == pytest.approx(2.125)
    assert result["weighted_blocks"]["joint"]["score"]["mean"] == pytest.approx(0.125)
    assert result["raw_joint"]["score"]["mean"] == pytest.approx(0.5)
    bad = copy.deepcopy(doc)
    bad["records"][0]["models"]["candidate"]["blocks"]["joint"]["score"] = 5
    with pytest.raises(ValueError, match="A-R"):
        summarize_scale(bad, "candidate", 2, bootstrap=20)
    bad = copy.deepcopy(doc)
    bad["sample_ids"].reverse()
    with pytest.raises(ValueError, match="condition IDs"):
        summarize_scale(bad, "candidate", 2, bootstrap=20)
