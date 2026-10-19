"""Re-score saved geometry/logits on CPU; no model instantiation or forward.

Reuses the unchanged matching contract and aromatic rule. Reconstruction is
metrics-only: no claim to restore the original hidden features or full logits.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.training.curriculum import data
from Data.sdf.chemvector_curriculum_v0_1.gallery import _font
from tools import diagnose_structure_expansion_bonds as diagnostic
from tools import review_structure_expansion_identity as fixed

REPORT_SHA = "b8ddab999dda04cfc86546348c2cbc8a213c7c16e54bc9e67895724ee2404f40"
OUTPUT = fixed.REPO / "outputs/audits/structure-expansion-bond-postcheck-20260923"


def restore_for_metrics(case, config, atom_threshold, pair_threshold):
    trace = case["candidates"]; rows = trace["candidates"]
    geometry = torch.zeros(1, config.maximum_atoms, 8)
    mask = torch.zeros(1, config.maximum_atoms, dtype=torch.bool)
    selected_mask = torch.zeros_like(mask)
    ids = [r["candidate"] for r in rows]
    fixed.require(len(ids) == len(set(ids)) == trace["raw_count"], "Duplicate/raw-count trace mismatch")
    for row in rows:
        slot = row["candidate"]
        fixed.require(type(slot) is int and 0 <= slot < config.maximum_atoms and len(row["geometry"]) == 8
            and type(row["selected"]) is bool and row["selected"] == (row["selection_logit"] >= atom_threshold),
            "Invalid saved slot/selection")
        geometry[0,slot] = torch.tensor(row["geometry"], dtype=torch.float32)
        mask[0,slot] = True
        selected_mask[0,slot] = row["selected"]
    fixed.require(int(selected_mask.sum()) == trace["selected_count"] and bool(torch.isfinite(geometry).all()),
                  "Invalid saved geometry/count")
    raw = dict(atom_geometry=geometry, atom_mask=mask, atom_indices=torch.arange(config.maximum_atoms)[None])
    selected = {**raw, "atom_mask":selected_mask}
    saved = case["pair_predictions"]
    pairs = torch.tensor(saved["pairs"], dtype=torch.long).reshape(-1,2)
    expected, _ = diagnostic.metrics.enumerate_unordered_atom_pairs(selected_mask)
    fixed.require(torch.equal(pairs,expected), "Saved pair enumeration changed")
    presence = torch.tensor(saved["presence_logits"], dtype=torch.float32)
    chemical = torch.tensor(saved["chemical_logits"], dtype=torch.float32).reshape(-1,4)
    fixed.require(presence.shape == (len(pairs),) and chemical.shape == (len(pairs),4)
                  and bool(torch.isfinite(presence).all()) and bool(torch.isfinite(chemical).all()),
                  "Invalid saved pair logits")
    promoted, _ = diagnostic.metrics.promote_complete_aromatic_rings(pairs, presence.sigmoid(), chemical,
        presence_threshold=pair_threshold, mean_threshold=.40, minimum_threshold=.05)
    fixed.require(promoted.tolist() == saved["promoted_chemical_ids"], "Saved aromatic promotion changed")
    return raw, selected, dict(pairs=pairs, output=dict(presence_logits=presence, chemical_logits=chemical), chemistry=promoted)


def draw_examples(examples):
    """GT error-bond overlays for human review only, never model inputs."""
    groups = ("raw_correspondence_gap", "selection_correspondence_gap", "reachable_missing", "reachable_wrong_type")
    representatives = []
    for stage in groups:
        eligible = [r for r in examples if r["stage"] == stage]
        if stage == "reachable_missing":
            subtype = Counter(r["element_pair"] for r in eligible).most_common(1)[0][0]
            eligible = [r for r in eligible if r["element_pair"] == subtype]
        elif stage == "reachable_wrong_type":
            subtype = Counter((r["expected_type"],r["predicted_type"]) for r in eligible).most_common(1)[0][0]
            eligible = [r for r in eligible if (r["expected_type"],r["predicted_type"]) == subtype]
        row = eligible[0]
        representatives.append(row)
    canvas=Image.new("RGB",(1120,1330),"white"); draw=ImageDraw.Draw(canvas)
    draw.text((12,12),"Saved diagnostic examples / GT error bonds in red / HUMAN REVIEW ONLY",font=_font(16),fill="black")
    draw.text((12,36),"Same structure, two original clean styles. Cropped thumbnails; model inputs unchanged.",font=_font(13),fill="#475569")
    selected = []
    for i, example in enumerate(representatives):
        for j, profile in enumerate(diagnostic.PROFILES):
            key = example["molecule_id"]+"__"+profile
            case = json.loads(fixed.read(diagnostic.OUTPUT/(key+".json")))
            layers = json.loads(fixed.read(fixed.DATASET/"samples"/(key+".layers.json")))
            atoms = {a["atom_id"]:a for a in layers["atoms"]}
            with Image.open(fixed.DATASET/"samples"/(key+".png")) as source:
                pic = source.convert("RGB")
            box=ImageChops.difference(pic,Image.new("RGB",pic.size,"white")).convert("L").point(lambda v:255 if v>12 else 0).getbbox()
            overlay=ImageDraw.Draw(pic)
            errors=[b for b in case["metrics"]["bonds"] if b["stage"] != "reachable_correct"]
            bonds={b["bond_id"]:b for b in layers["bonds"]}
            for error in errors:
                bond=bonds[error["bond_id"]]
                left=tuple(atoms[bond["begin_atom"]]["center"]); right=tuple(atoms[bond["end_atom"]]["center"])
                overlay.line((left,right),fill="#e03131",width=2)
                for x,y in (left,right): overlay.ellipse((x-4,y-4,x+4,y+4),outline="#e03131",width=2)
            if box:
                l,t,r,b=box; pic=pic.crop((max(0,l-16),max(0,t-16),min(768,r+16),min(768,b+16)))
            pic.thumbnail((530,222))
            x,y=j*560,72+i*310
            draw.rectangle((x,y,x+559,y+309),outline="#cbd5e1")
            draw.text((x+10,y+8),key,font=_font(13),fill="black")
            draw.text((x+10,y+28),example["stage"],font=_font(12),fill="#475569")
            canvas.paste(pic,(x+(560-pic.width)//2,y+53+(222-pic.height)//2))
            draw.text((x+10,y+283),f"local correct {case['metrics']['bond_stages']['reachable_correct']}/{case['metrics']['coverage']['target_bonds']}; error bonds {len(errors)}",font=_font(12),fill="#475569")
            selected.append(key)
    canvas.save(OUTPUT/"failure-examples.png")
    return selected


def run():
    torch.set_num_threads(2)
    fixed.require(not OUTPUT.exists(), "Fresh postcheck output required")
    root=diagnostic.OUTPUT
    fixed.require(fixed.digest(root/"report.json")==REPORT_SHA,"Saved diagnostic changed")
    report=json.loads(fixed.read(root/"report.json")); plan=json.loads(fixed.read(root/"plan.json"))
    fixed.require(report["complete"] is True and fixed.digest(root/"plan.json")==report["plan_sha256"],"Incomplete diagnostic")
    fixed.verify_bindings(plan["input_sha256"])
    config=PredictedProposalConfig(**report["config"])
    collator=data.CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,bond_queries=config.maximum_bonds),purpose="diagnostic")
    cases, failures, atom_gaps = [],[],[]
    for row in plan["samples"]:
        key=row["sample_key"]; path=root/(key+".json")
        fixed.require(fixed.digest(path)==report["case_sha256"][key],"Saved case changed")
        case=json.loads(fixed.read(path))
        fixed.require(all(case[k]==v for k,v in row.items()),"Case/plan selection mismatch")
        raw,selected,predicted=restore_for_metrics(case,config,plan["thresholds"]["atom_threshold"],plan["thresholds"]["pair_threshold"])
        sample=data.load_sample(fixed.DATASET,key,purpose="diagnostic")
        batch=collator([sample])
        measured=diagnostic.metrics.measure(raw,selected,predicted,batch,config,plan["thresholds"]["pair_threshold"])
        fixed.require(measured==case["metrics"],"Saved geometry/logit re-score differs")
        atoms={a["atom_id"]:a for a in sample["graph"]["atoms"]}
        bonds={b["bond_id"]:b for b in sample["graph"]["bonds"]}
        for bond in measured["bonds"]:
            if bond["stage"]=="reachable_correct": continue
            edge=bonds[bond["bond_id"]]
            elements=sorted(atoms[a]["element"] for a in (edge["begin_atom"],edge["end_atom"]))
            failures.append(dict(sample_key=key,molecule_id=row["molecule_id"],profile_id=row["profile_id"],
                element_pair="-".join(elements),**bond))
        for detail in measured["coverage"]["target_details"]:
            if detail["selected_assignment_matched"]: continue
            aid=batch["atom_ids"][0][detail["target"]]
            slots={c["candidate"]:c for c in case["candidates"]["candidates"]}
            atom_gaps.append(dict(sample_key=key,atom_id=aid,element=atoms[aid]["element"],**detail,
                near_raw_selector_logits=[slots[c]["selection_logit"] for c in detail["near_raw_ids"]]))
        cases.append(case)
    summary=diagnostic.metrics.summarize(cases)
    fixed.require(len(cases)==64 and summary==report["summary"],"Summary differs from saved re-score")
    for profile in diagnostic.PROFILES:
        fixed.require(diagnostic.metrics.summarize([c for c in cases if c["profile_id"]==profile])==report["by_profile"][profile],"Style summary differs")
    fixed.verify_bindings(plan["input_sha256"])
    missing_pairs=Counter(row["element_pair"] for row in failures if row["stage"]=="reachable_missing")
    confused=Counter((r["expected_type"],r["raw_type_argmax"],r["predicted_type"]) for r in failures if r["stage"]=="reachable_wrong_type")
    OUTPUT.mkdir(parents=True,exist_ok=False)
    fixed.write_new(OUTPUT/"bond-errors.json",failures)
    fixed.write_new(OUTPUT/"unmatched-atoms.json",atom_gaps)
    gallery=draw_examples(failures)
    result=dict(status="passed_saved_prediction_postcheck",diagnostic_report_sha256=REPORT_SHA,
        input_bindings_verified=len(plan["input_sha256"]),cases_rescored=len(cases),summary=summary,
        bond_error_count=len(failures),unmatched_target_atom_count=len(atom_gaps),
        raw_no_nearby_atoms=sum(not row["near_raw_ids"] for row in atom_gaps),
        raw_matched_then_no_selected_nearby=sum(row["raw_assignment_matched"] and not row["near_selected_ids"] for row in atom_gaps),
        unmatched_with_selected_nearby=sum(bool(row["near_selected_ids"]) for row in atom_gaps),
        missing_bonds_by_element_pair=dict(missing_pairs),
        chemical_confusions=[dict(expected=a,raw_argmax=b,after_promotion=c,count=n) for (a,b,c),n in sorted(confused.items())],
        gallery_selection="first frozen-panel case per stage; missing/type prioritize most frequent subtype; paired other style; illustrative only",
        gallery_sample_keys=gallery,model_loads=0,model_forwards=0,private_identity_queries=0,training_authorized=False,
        source_images_modified=False,gt_overlays_for_human_review_only=True,
        reconstruction_scope="saved active candidate geometry/masks and pair logits; not hidden features or original prediction digest",
        wrapper_sha256=fixed.digest(__file__),
        artifact_sha256={p.name:fixed.digest(p) for p in OUTPUT.iterdir() if p.is_file()})
    fixed.write_new(OUTPUT/"report.json",result)
    return result


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    result=run()
    print(json.dumps({k:result[k] for k in ("status","cases_rescored","bond_error_count","unmatched_target_atom_count","missing_bonds_by_element_pair","model_forwards")}))


if __name__ == "__main__": main()
