"""One bounded CPU factorial on the four already screened P--O structures.

New diagnostic images only; immutable historical code/data remain untouched.
The audit action re-scores saved geometry/logits, never reloads a model.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import torch

from Data.sdf.case_diversity_v1.rendering import render
from Data.sdf.chemvector_curriculum_v0_1.build import QUALITY_REJECTIONS
from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile
from Data.sdf.chemvector_curriculum_v0_1.typography import LINE
from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.experiments import po_style_factorial as design
from chemvector.training.curriculum import bundle, continuation, data, pixels, upstream_diagnostic
from chemvector.training.curriculum.runtime import ResourceBudget, run_budgeted
from tools import audit_structure_bond_diagnostic as saved_audit
from tools import diagnose_structure_expansion_bonds as prior
from tools import review_structure_expansion_identity as fixed
from tools.preflight_curriculum_v02_targets import validate_targets

OUTPUT = fixed.REPO / "outputs/experiments/chemvector-po-style-factorial-20260923"
DATA = OUTPUT / "dataset"
AUDIT = OUTPUT / "postcheck"
PRIOR_SHA = "b8ddab999dda04cfc86546348c2cbc8a213c7c16e54bc9e67895724ee2404f40"


def prepare_plan():
    fixed.require(fixed.digest(prior.OUTPUT/"report.json") == PRIOR_SHA,"Previous fixed diagnostic changed")
    previous = json.loads(fixed.read(prior.OUTPUT/"report.json"))
    plan = json.loads(fixed.read(prior.OUTPUT/"plan.json"))
    fixed.require(previous["complete"] is True and fixed.digest(prior.OUTPUT/"plan.json") == previous["plan_sha256"],
                  "Completed previous diagnostic required")
    bindings = dict(plan["input_sha256"])
    bindings.update({str(prior.OUTPUT/"report.json"):PRIOR_SHA,
                     str(prior.OUTPUT/"plan.json"):previous["plan_sha256"]})
    fixed.verify_bindings(bindings)
    entries = design.select_entries(plan["samples"],json.loads(fixed.read(fixed.DATASET/"molecules.json")),
        json.loads(fixed.read(prior.public.OUTPUT/"remaining-candidates.json")))
    controls = {}
    for entry in entries:
        for profile in ("oa00","oa22"):
            key=entry["molecule_id"]+"__"+profile
            controls[key]=previous["case_sha256"][key]
            bindings[str(prior.OUTPUT/(key+".json"))]=controls[key]
    for path in (Path(__file__).absolute(),Path(design.__file__).absolute(),Path(saved_audit.__file__).absolute(),
                 fixed.REPO/"tests/data_sdf/test_po_style_factorial.py"):
        bindings[str(path)]=fixed.digest(path)
    fixed.verify_bindings(bindings)
    return dict(schema_version="po-style-factorial-plan-1",entries=entries,profiles=design.profiles(),
        input_sha256=bindings,previous_cases_sha256=controls,planned_images=64,planned_controls=8,
        factors=list(design.FACTORS),planned_structure_forwards=64,planned_pair_forwards=64,
        thresholds=plan["thresholds"],provenance=plan["provenance"],initial_sha256=prior.metrics.INITIAL_SHA,
        expected_initial_state=prior.metrics.INITIAL_STATE,max_cpu_seconds=300,automatic_retries=0,
        new_molecule_identities=0,identity_queries=0,training_authorized=False,gpu_used=False,
        selection="all four prior-panel structures with two P--O singles; fixed 2^4 factorial, no replacements",
        quality_policy="any rejected/missing cell aborts before model loading; no imputation or relaxed checks",
        meaning="total effect of one renderer parameter; layout/text clearance may mediate effects")


def focus_bonds(sample, measured, trace):
    atoms={a["atom_id"]:a for a in sample["layers"]["atoms"]}
    edges={b["bond_id"]:b for b in sample["layers"]["bonds"]}
    svg=ET.fromstring(sample["svg"] if "svg" in sample else fixed.read(DATA/"samples"/(sample["key"]+".svg")))
    candidates={row["candidate"]:row for row in trace["candidates"]}
    result=[]
    for bond in measured["bonds"]:
        edge=edges[bond["bond_id"]]
        left,right=(atoms[edge[name]] for name in ("begin_atom","end_atom"))
        if {left["element"],right["element"]} != {"P","O"}: continue
        length=math.dist(left["center"],right["center"])
        segments=[]
        for index in edge["svg_node_indices"]:
            match=LINE.fullmatch(svg[index].attrib["d"])
            fixed.require(match is not None,"Expected ordinary straight P--O line segments")
            x1,y1,x2,y2=map(float,match.groups()); segments.append(math.dist((x1,y1),(x2,y2)))
        fixed.require(length > 0 and segments,"Missing P--O drawing geometry")
        source_ids=[edge["begin_atom"],edge["end_atom"]]
        candidate_errors=[]
        for slot,aid in zip(bond["candidates"],source_ids):
            candidate_errors.append(None if slot is None else math.dist(
                [v*767 for v in candidates[slot]["geometry"][:2]],atoms[aid]["center"]))
        result.append(dict(**bond,anchor_length_px=length,visible_segment_lengths_px=segments,
            visible_fraction=sum(segments)/len(segments)/length,predicted_endpoint_error_px=candidate_errors,
            atom_ids=source_ids,text_labels=[left["text"],right["text"]]))
    fixed.require(len(result)==3,"Expected exactly three P--O target edges")
    return result


def run():
    fixed.require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not torch.cuda.is_initialized(),"CPU-only factorial")
    fixed.require(not OUTPUT.exists() and OUTPUT.resolve()==OUTPUT,"Fresh factorial output required")
    torch.set_num_threads(2); torch.manual_seed(20260923)
    start=time.monotonic(); plan=prepare_plan()
    OUTPUT.mkdir(parents=True,exist_ok=False); DATA.mkdir(); (DATA/"samples").mkdir(); (OUTPUT/"cases").mkdir()
    fixed.write_new(OUTPUT/"plan.json",plan)
    report=dict(schema_version="po-style-factorial-result-1",status="incomplete",complete=False,
        plan_sha256=fixed.digest(OUTPUT/"plan.json"),training_authorized=False,training_performed=False,
        model_loads=0,structure_forwards=0,pair_forwards=0,gpu_used=False,identity_queries=0,
        optimizer_steps=0,automatic_retries=0,default_model_changed=False,default_thresholds_changed=False)
    rows,rejected,cases,hashes=[],[],[],{}
    pixel_controls,prediction_controls=0,0
    try:
        for entry in plan["entries"]:
            for cell in plan["profiles"]:
                profile=Profile(**cell["profile"])
                try:
                    files,meta=render(entry,profile)
                except ValueError as exc:
                    if str(exc) not in QUALITY_REJECTIONS: raise
                    rejected.append(dict(molecule_id=entry["molecule_id"],profile_id=profile.profile_id,
                                         bits=cell["bits"],reason=str(exc)))
                    continue
                key=meta["sample_key"]
                for suffix,payload in files.items():
                    name="samples/"+key+suffix
                    with (DATA/name).open("xb") as stream: stream.write(payload)
                    hashes[name]=sha256(payload).hexdigest()
                rows.append(dict(sample_key=key,molecule_id=entry["molecule_id"],profile_id=profile.profile_id,
                                 canonical_smiles=entry["canonical_smiles"],bits=cell["bits"],measured=meta["measured"]))
                if profile.profile_id in ("oa00","oa22"):
                    original=fixed.DATASET/"samples"/(key+".png")
                    fixed.require(meta["sha256"][".png"]==fixed.digest(original),"Control rerender changed original PNG bytes")
                    pixel_controls+=1
            print(json.dumps(dict(rendered=len(rows),rejected=len(rejected),planned=64)),flush=True)
        for name,value in (("samples.json",rows),("rejections.json",rejected),("sample-sha256.json",hashes)):
            fixed.write_new(DATA/name,value)
        report.update(rendered_images=len(rows),rejected_images=len(rejected),control_png_parity_passed=pixel_controls)
        fixed.require(len(rows)==64 and not rejected and pixel_controls==8,"Incomplete full factorial; no model run")
        fixed.verify_bindings({str(DATA/name):value for name,value in hashes.items()})
        collate=data.CurriculumCollator(AtomBondSceneConfig(),purpose="diagnostic")
        for row in rows:
            sample=data.load_sample(DATA,row["sample_key"],purpose="diagnostic")
            validate_targets(sample,collate([sample]),collate.config)
        report["target_preflight_passed"]=len(rows)
        fixed.verify_bindings(plan["input_sha256"])
        model,config,payload=bundle.load_bundle(prior.metrics.RUN/"initial.pt",expected_sha256=prior.metrics.INITIAL_SHA)
        report["model_loads"]=1
        fixed.require(payload["provenance"]==plan["provenance"] and payload["progress"]["optimizer_steps"]==0
            and prior.metrics.state_digest(model)==prior.metrics.INITIAL_STATE,"Unexpected initial model state")
        model.eval().requires_grad_(False)
        target_config=AtomBondSceneConfig(atom_queries=config.maximum_atoms,bond_queries=config.maximum_bonds)
        collate=data.CurriculumCollator(target_config,purpose="diagnostic")
        # Process the eight controls first, before any new factorial image inference.
        ordered=sorted(rows,key=lambda row:(row["profile_id"] not in ("oa00","oa22"),row["sample_key"]))
        with torch.inference_mode():
            for row in ordered:
                sample=data.load_sample(DATA,row["sample_key"],purpose="diagnostic")
                report["structure_forwards"]+=1
                forward=continuation.structure_forward(model,pixels.image_tensor(sample["image"]).unsqueeze(0),config,plan["thresholds"]["atom_threshold"])
                report["pair_forwards"]+=1
                predicted=prior.metrics.predict_pair(model,forward[0],forward[6],plan["thresholds"]["pair_threshold"])
                prediction_sha=prior.metrics.prediction_digest(forward[3],forward[6],predicted)
                measured=prior.metrics.measure(forward[3],forward[6],predicted,collate([sample]),config,plan["thresholds"]["pair_threshold"])
                fixed.require(prediction_sha==prior.metrics.prediction_digest(forward[3],forward[6],predicted),"Scoring changed predictions")
                trace=upstream_diagnostic.candidate_trace(forward[3],forward[6],forward[5])
                if row["profile_id"] in ("oa00","oa22"):
                    old=json.loads(fixed.read(prior.OUTPUT/(row["sample_key"]+".json")))
                    fixed.require(old["prediction_sha256"]==prediction_sha and old["metrics"]==measured
                                  and old["candidates"]==trace,"Original model prediction control parity failed")
                    prediction_controls+=1
                case={**row,"metrics":measured,"candidates":trace,"prediction_sha256":prediction_sha,
                    "pair_predictions":dict(pairs=predicted["pairs"].tolist(),
                        presence_logits=predicted["output"]["presence_logits"].tolist(),
                        chemical_logits=predicted["output"]["chemical_logits"].tolist(),
                        promoted_chemical_ids=predicted["chemistry"].tolist()),
                    "po_bonds":focus_bonds(sample,measured,trace)}
                fixed.write_new(OUTPUT/"cases"/(row["sample_key"]+".json"),case); cases.append(case)
                if len(cases)%8==0: print(json.dumps(dict(checked_images=len(cases),controls=prediction_controls)),flush=True)
        fixed.require(prediction_controls==8 and prior.metrics.state_digest(model)==prior.metrics.INITIAL_STATE
            and not torch.cuda.is_initialized() and all(not p.requires_grad and p.grad is None for p in model.parameters()),"Frozen state/device/control guard failed")
        fixed.verify_bindings(plan["input_sha256"])
        fixed.verify_bindings({str(DATA/name):value for name,value in hashes.items()})
        report.update(status="completed_diagnostic_only",complete=True,summary=design.summarize(cases),
            all_bond_summary=prior.metrics.summarize(cases),config=asdict(config),
            state_after=prior.metrics.INITIAL_STATE,control_prediction_parity_passed=prediction_controls,
            limitations=["Four error-enriched prior structures, not unseen validation or independent factorial replicates.",
                "Changing a renderer parameter includes mediated text clearance/layout changes; not a fixed-pixel neural intervention.",
                "Missing endpoints remain failures; probability effects only compare mutually reachable targets.",
                "Diagnostic image variants are not new identities or a training release; no OCR/stereo/final-SDF accuracy."])
    except Exception as exc:
        report.update(status="failed_or_incomplete",error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-start,checked_images=len(cases),
            control_prediction_parity_passed=prediction_controls,
            case_sha256={c["sample_key"]:fixed.digest(OUTPUT/"cases"/(c["sample_key"]+".json")) for c in cases},
            generated_manifest_sha256={name:fixed.digest(DATA/name) for name in
                ("samples.json","rejections.json","sample-sha256.json") if (DATA/name).is_file()})
        fixed.write_new(OUTPUT/"report.json",report)
    return report


def audit():
    fixed.require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not AUDIT.exists(),"Fresh CPU-only postcheck required")
    torch.set_num_threads(2)
    report=json.loads(fixed.read(OUTPUT/"report.json")); plan=json.loads(fixed.read(OUTPUT/"plan.json"))
    fixed.require(report["complete"] is True and fixed.digest(OUTPUT/"plan.json")==report["plan_sha256"],"Incomplete experiment")
    fixed.verify_bindings(plan["input_sha256"])
    fixed.verify_bindings({str(DATA/name):value for name,value in report["generated_manifest_sha256"].items()})
    hashes=json.loads(fixed.read(DATA/"sample-sha256.json"))
    fixed.verify_bindings({str(DATA/name):value for name,value in hashes.items()})
    config=PredictedProposalConfig(**report["config"])
    collate=data.CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,bond_queries=config.maximum_bonds),purpose="diagnostic")
    cases=[]
    for key,value in report["case_sha256"].items():
        path=OUTPUT/"cases"/(key+".json"); fixed.require(fixed.digest(path)==value,"Saved case changed")
        case=json.loads(fixed.read(path)); sample=data.load_sample(DATA,key,purpose="diagnostic")
        raw,selected,predicted=saved_audit.restore_for_metrics(case,config,plan["thresholds"]["atom_threshold"],plan["thresholds"]["pair_threshold"])
        batch=collate([sample]); validate_targets(sample,batch,collate.config)
        measured=prior.metrics.measure(raw,selected,predicted,batch,config,plan["thresholds"]["pair_threshold"])
        fixed.require(measured==case["metrics"] and focus_bonds(sample,measured,case["candidates"])==case["po_bonds"],"Saved metrics/geometry changed")
        cases.append(case)
    summary=design.summarize(cases)
    fixed.require(summary==report["summary"] and prior.metrics.summarize(cases)==report["all_bond_summary"],"Saved factorial aggregate mismatch")
    fixed.verify_bindings(plan["input_sha256"])
    AUDIT.mkdir()
    result=dict(status="passed_saved_factorial_postcheck",source_report_sha256=fixed.digest(OUTPUT/"report.json"),
        cases_rescored=len(cases),generated_files_verified=len(hashes),input_bindings_verified=len(plan["input_sha256"]),
        model_loads=0,model_forwards=0,identity_queries=0,training_authorized=False,
        reconstruction_scope="saved active candidate geometry/masks and pair logits only; not hidden features",summary=summary)
    fixed.write_new(AUDIT/"report.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("run","audit")); parser.add_argument("--worker",action="store_true")
    args=parser.parse_args(argv)
    if args.action=="audit":
        result=audit(); print(json.dumps({k:result[k] for k in ("status","cases_rescored","model_forwards")})); return 0
    if args.worker: return 0 if run()["complete"] else 1
    result=run_budgeted([sys.executable,"-m","tools.diagnose_po_style_factorial","run","--worker"],
        output=OUTPUT.with_name(OUTPUT.name+"-runtime"),budget=ResourceBudget(300))
    print(json.dumps(result,sort_keys=True)); return 0 if result["status"]=="completed" else 1


if __name__=="__main__": raise SystemExit(main())
