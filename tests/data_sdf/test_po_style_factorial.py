"""Synthetic factorial checks plus one synthetic rendered P--O target bundle."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from chemvector.experiments import po_style_factorial as design
from tools import diagnose_po_style_factorial as experiment


def cases():
    result=[]
    for mid in ("a","b","c","d"):
        for cell in design.profiles():
            probability=.95-.4*cell["bits"][1]
            bonds=[dict(bond_id=i,expected_type=kind,presence_probability=probability if kind=="SINGLE" else .99,
                stage="reachable_missing" if kind=="SINGLE" and probability<.85 else "reachable_correct",
                visible_fraction=.7-.1*cell["bits"][1],anchor_length_px=42+18*cell["bits"][2])
                for i,kind in enumerate(("DOUBLE","SINGLE","SINGLE"))]
            result.append(dict(molecule_id=mid,bits=cell["bits"],po_bonds=bonds))
    return result


def test_full_factorial_and_original_controls():
    profiles=design.profiles()
    assert len(profiles)==len({tuple(c["bits"]) for c in profiles})==16
    assert profiles[0]["profile"]["profile_id"]=="oa00"
    assert profiles[-1]["profile"]["profile_id"]=="oa22"
    for cell in profiles:
        assert cell["profile"]["rotation_degrees"]==0
        for name,levels,bit in zip(design.FACTORS,design.LEVELS,cell["bits"]):
            assert cell["profile"][name]==levels[bit]


def test_paired_effects_change_one_factor_and_order_is_stable():
    original=cases(); before=deepcopy(original)
    result=design.summarize(original)
    assert result==design.summarize(list(reversed(original))) and original==before
    assert result["single_targets"]==128 and result["double_targets"]==64
    for row in result["paired_factor_effects"]:
        assert row["paired_comparisons"]==row["both_reachable"]==64
        assert row["mean_probability_delta"]==pytest.approx(-.4 if row["factor"]=="font_size" else 0)
        assert row["lost_correct"]==(64 if row["factor"]=="font_size" else 0)


@pytest.mark.parametrize("mutation",["missing","duplicate","bond","cohort"])
def test_incomplete_factorial_never_imputes_or_passes(mutation):
    values=cases()
    if mutation=="missing": values.pop()
    elif mutation=="duplicate": values[-1]=values[0]
    elif mutation=="bond": values[0]["po_bonds"].pop()
    else: values=[r for r in values if r["molecule_id"]!="a"]
    with pytest.raises(ValueError): design.summarize(values)


def test_unreachable_is_not_probability_zero_or_dropped_from_denominator():
    values=cases()
    values[0]["po_bonds"][1].update(presence_probability=None,stage="raw_correspondence_gap")
    result=design.summarize(values)
    assert result["by_profile"][0]["single_targets"]==8
    assert result["by_profile"][0]["single_reachable"]==7
    assert all(r["paired_comparisons"]==64 and r["both_reachable"]==63 for r in result["paired_factor_effects"])


def test_fixed_prior_cohort_no_new_identities():
    registry=[dict(molecule_id=f"m{i}",canonical_smiles=smi,split="quarantine",training_authorized=False)
              for i,smi in enumerate(("O=P(O)(O)C","O=P(O)(O)CC","O=P(O)(O)CCC","O=P(O)(O)CCCC"))]
    remaining=[{**row,"sample_keys":[row["molecule_id"]+"__oa00"]} for row in registry]
    prior=[dict(molecule_id=r["molecule_id"],canonical_smiles=r["canonical_smiles"],sample_key=r["sample_keys"][0]) for r in remaining]
    assert design.select_entries(prior,registry,remaining)==registry
    with pytest.raises(ValueError): design.select_entries(prior,registry,remaining[:-1])


def test_rendered_single_molecule_targets_and_geometry(tmp_path,monkeypatch):
    from Data.sdf.case_diversity_v1.rendering import render
    from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile
    from chemvector.training.curriculum.data import load_sample,CurriculumCollator
    from tools.preflight_curriculum_v02_targets import validate_targets
    entry=dict(molecule_id="synthetic_po",canonical_smiles="O=P(O)(O)CCO",split="quarantine",
        training_authorized=False,external_overlap_status="unchecked",stages=["functional_groups"])
    profile=Profile(**design.profiles()[0]["profile"])
    files,meta=render(entry,profile)
    (tmp_path/"samples").mkdir()
    for suffix,payload in files.items(): (tmp_path/"samples"/(meta["sample_key"]+suffix)).write_bytes(payload)
    monkeypatch.setattr(experiment,"DATA",tmp_path)
    sample=load_sample(tmp_path,meta["sample_key"],purpose="diagnostic")
    collate=CurriculumCollator(purpose="diagnostic")
    assert validate_targets(sample,collate([sample]),collate.config)["atoms"]==7
    measured=dict(bonds=[dict(bond_id=b["bond_id"],expected_type=b["chemical_type"],candidates=[None,None],
        presence_probability=None,stage="raw_correspondence_gap") for b in sample["graph"]["bonds"]])
    focus=experiment.focus_bonds(sample,measured,{"candidates":[]})
    assert len(focus)==3 and all(0<r["visible_fraction"]<=1 and r["anchor_length_px"]>0 for r in focus)
    with pytest.raises(ValueError): load_sample(tmp_path,meta["sample_key"],purpose="training")


def test_cpu_guard_before_work(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    with pytest.raises(ValueError,match="CPU-only"): experiment.run()
