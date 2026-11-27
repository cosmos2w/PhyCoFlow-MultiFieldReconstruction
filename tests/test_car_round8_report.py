import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.analysis.car_round8_reference_strata import reference_moments, summarize_strata
from scripts.analysis.report_car_round8 import (
    add_extra_figures, event_mask, figure_sections, physical_relative_l2, physical_rmse,
    rebase_section_figure_links,
)
from scripts.analysis.car_round8_cached_summary import ref
from scripts.analysis.car_round8_relation_intervals import paired_triplet, relation_intervals


def test_native_reference_moments_use_physical_masses_and_affine_units():
    truth=torch.tensor([[[1.],[3.]]]);mass=torch.tensor([1.,3.]);scale=torch.tensor([2.]);offset=torch.tensor([5.])
    mean,energy,variance=reference_moments(truth,mass,scale,offset)
    assert np.allclose(mean,[10.])
    assert np.allclose(energy,[(7**2+3*11**2)/4])
    assert np.allclose(variance,[3.])
    risk=[[1.],[2.],[3.],[4.]]
    result=summarize_strata(risk,[[2.],[3.],[4.],[5.]],[[1.],[2.],[3.],[4.]],list("abcd"),["field"])[0]
    assert result["source_member_mse_over_reference_spatial_variance"]==1
    assert [x["mean_member_risk"] for x in result["within_split_strata"]]==[1.5,3.5]


def test_error_annotation_uses_masked_physical_relative_l2():
    assert physical_relative_l2([2.,4.,900.],[1.,2.,0.],[1.,3.,0.])==pytest.approx(1.)
    assert physical_rmse([1.,.5,99.],[0.,1.,0.],[1.,3.,0.])==pytest.approx(np.sqrt(1.75/4))
    with pytest.raises(ValueError,match="zero reference"):
        physical_relative_l2([1.,2.],[0.,0.],[1.,1.])


def test_frozen_joint_event_uses_strict_thresholds_in_named_field_order():
    names=["CO","CH4","T"]
    definition={"T":{"threshold":3.},"CO":{"threshold":2.},"CH4":{"threshold":1.}}
    values=np.asarray([[3.,.5,4.],[2.,.5,4.],[3.,1.,4.],[3.,.5,3.]])
    assert event_mask(values,definition,names).tolist()==[True,False,False,False]


def test_paired_triplet_intervals_preserve_condition_pairing():
    source=[[5.,2.,3.],[10.,4.,6.],[20.,8.,12.]]
    candidate=[[6.,2.5,3.5],[11.,4.5,6.5],[21.,8.5,12.5]]
    result=paired_triplet(source,candidate,seed=10,replicates=100)
    assert np.allclose(result["mean"],[1.,.5,.5])
    assert np.allclose(result["interval_95"],[[1.,.5,.5],[1.,.5,.5]])
    with pytest.raises(ValueError,match="triplet"):
        paired_triplet(source,[[1.,2.,9.]]*3)


def test_component_changes_retain_raw_units_fixed_weights_and_paired_frames(tmp_path):
    ids=["frame-1","frame-2","frame-3"]
    source=np.asarray([[5.,2.,3.],[10.,4.,6.],[20.,8.,12.]])
    delta=np.asarray([[.1,.02,.08],[.3,.06,.24],[.9,.18,.72]])
    weights={"signed":.2,"joint":.125,"log_power_self":.3,"rho":.4,"third_order":.1}
    leaves={"self.finite":.1,"self.essential":.2,"mutual.finite":.3,"mutual.essential":.4}
    terms=("attraction","separation","score")
    records=[]
    scores={side:[] for side in ("SOURCE","candidate")}
    support={"sample_ids":ids,"records":[]}
    for i,sid in enumerate(ids):
        models={}
        for side,value in (("SOURCE",source[i]),("candidate",source[i]+delta[i])):
            models[side]={"blocks":{block:dict(zip(terms,(j+1)*value))
                                   for j,block in enumerate(weights)}}
            row={"shape":float(value[2])}
            for leaf,weight in leaves.items():
                row[f"shape.{leaf}.attraction"]=float(weight*value[0])
                row[f"shape.{leaf}.separation"]=float(weight*value[1])
            scores[side].append(row)
        records.append({"models":models})
        support["records"].append({"models":{"SOURCE":{"state":source[i]},
                                                       "JointAnchor500":{"state":source[i]+delta[i]}}})
    path=tmp_path/"decomposition.json"
    path.write_text(json.dumps({"sample_ids":ids,"records":records}))
    summary={"sample_ids":ids,"scale_weights":weights,"inputs":{"decomposition":ref(path)}}
    result=relation_intervals(summary,{"sample_ids":ids,"per_condition_scores":scores},support)
    raw=result["Scale_blocks"]["joint"]["raw"]
    weighted=result["Scale_blocks"]["joint"]["weighted"]
    np.testing.assert_allclose(raw["mean"],2*delta.mean(0))
    np.testing.assert_allclose(weighted["interval_95"],.125*np.asarray(raw["interval_95"]))
    np.testing.assert_allclose(sum(np.asarray(v["weighted"]["mean"]) for v in result["Scale_blocks"].values()),result["Scale"]["mean"])
    np.testing.assert_allclose(sum(np.asarray(v["mean"]) for v in result["Shape_components"].values()),result["Shape"]["mean"])
    np.testing.assert_allclose(result["Shape_components"]["mutual.essential"]["mean"],.4*delta.mean(0))


def test_extra_figures_require_matching_ordinary_pairs_and_unique_names(tmp_path):
    source=tmp_path/"source";destination=tmp_path/"figures";source.mkdir();destination.mkdir()
    (source/"training_shape.png").write_bytes(b"image")
    (source/"training_shape.pdf").write_bytes(b"master")
    manifest=tmp_path/"extra.json";manifest.write_text('{"training_shape.png":"TRAIN-only feature sensitivity."}')
    captions={};bindings=add_extra_figures(manifest,source,destination,captions)
    assert captions=={"training_shape":"TRAIN-only feature sensitivity."}
    assert (destination/"training_shape.pdf").read_bytes()==b"master"
    assert set(bindings["training_shape"])=={"png","pdf"}
    with pytest.raises(ValueError,match="unique"):
        add_extra_figures(manifest,source,destination,captions)
    manifest.write_text('{"../escape.png":"invalid"}')
    with pytest.raises(ValueError,match="plain filenames"):
        add_extra_figures(manifest,source,destination,{})


def test_mixed_case_figures_rebase_inline_links_and_pass_historical_anchor_validation(tmp_path, monkeypatch):
    from scripts.analysis import car_round7_render
    from scripts.analysis.assemble_car_round8_report import localize_links

    bundle=tmp_path/"assessment"/"bundle";source=bundle/"figures";source.mkdir(parents=True)
    output=tmp_path/"report";destination=output/"figures";destination.mkdir(parents=True)
    original="learning_representative_median_CO_T";exported=original.lower()
    Image.new("RGB",(3,3),"white").save(source/f"{original}.png")
    (source/f"{original}.pdf").write_bytes(b"synthetic PDF master")
    caption="SOURCE-selected CO and T fields."
    manifest=bundle/"figure_captions.json";manifest.write_text(json.dumps({original:caption}))
    captions={};bindings=add_extra_figures(manifest,source,destination,captions)
    assert captions=={exported:caption}
    for ext in ("png","pdf"):
        assert (destination/f"{exported}.{ext}").read_bytes()==(source/f"{original}.{ext}").read_bytes()
        assert bindings[exported][ext]==ref(source/f"{original}.{ext}")
        assert not (destination/f"{original}.{ext}").exists()
    section=f"Inline ![field](figures/{original}.png) and [PDF master](figures/{original}.pdf)."
    rebased=rebase_section_figure_links(section,captions)
    assert f"figures/{exported}.png" in rebased and f"figures/{exported}.pdf" in rebased
    assert f"figures/{original}.png" not in rebased and f"figures/{original}.pdf" not in rebased
    section_file=bundle/"section.md";section_file.write_text(section)
    localized=localize_links(section,section_file,tmp_path)
    assembled_section=rebase_section_figure_links(localized,captions)
    assert f"![field](figures/{exported}.png)" in assembled_section
    assert f"`assessment/bundle/figures/{original}.pdf`" in assembled_section
    (output/"report.md").write_text("# Coherence Adaptive Repair\n\n"+assembled_section+"\n\n"+"\n".join(figure_sections(captions)))

    class ReachedPDFConversion(Exception): pass
    def mock_subprocess(command,*args,**kwargs):
        if command[0]=="pdfinfo": return SimpleNamespace(stdout="Pages: 1\n")
        if command[0]=="libreoffice": raise ReachedPDFConversion
        pytest.fail(f"unexpected subprocess: {command}")
    monkeypatch.setattr(car_round7_render.subprocess,"run",mock_subprocess)
    with pytest.raises(ReachedPDFConversion):
        car_round7_render.render(output,captions)
    html=(output/"report.html").read_text()
    anchor="figure-"+exported.replace("_","-")
    assert f'id="{anchor}"' in html and f'href="#{anchor}"' in html


def test_extra_figure_casefold_collisions_are_rejected_before_copying(tmp_path):
    source=tmp_path/"source";destination=tmp_path/"figures";source.mkdir();destination.mkdir()
    for stem in ("Learning_Ratio","learning_ratio"):
        for ext in ("png","pdf"):
            (source/f"{stem}.{ext}").write_bytes(f"{stem}.{ext}".encode())
    manifest=tmp_path/"extra.json"
    manifest.write_text(json.dumps({"Learning_Ratio.png":"first","learning_ratio.pdf":"second"}))
    captions={}
    with pytest.raises(ValueError,match="unique"):
        add_extra_figures(manifest,source,destination,captions)
    assert captions=={} and list(destination.iterdir())==[]
    manifest.write_text(json.dumps({"Learning_Ratio":"first"}))
    with pytest.raises(ValueError,match="unique"):
        add_extra_figures(manifest,source,destination,{"learning_ratio":"existing"})
    (destination/"LEARNING_RATIO.PDF").write_bytes(b"existing")
    with pytest.raises(ValueError,match="unique"):
        add_extra_figures(manifest,source,destination,{})
