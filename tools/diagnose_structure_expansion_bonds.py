"""Frozen initial, 32 structures x two styles, at most 300 CPU wall seconds.

Labels select a deterministic size-spread coverage panel and score AFTER the
image-only predictions. No OCR, training, threshold search or deployment.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import sys
import time

from rdkit import Chem
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum import bundle, continuation, data, pixels, supervision_contract, upstream_diagnostic
from chemvector.training.curriculum.runtime import ResourceBudget, run_budgeted
from tools import diagnose_curriculum_v02_bond_reachability as metrics
from tools import review_structure_expansion_identity as fixed
from tools import review_structure_expansion_public as public

OUTPUT = fixed.REPO / "outputs/audits/structure-expansion-bond-diagnostic64-20260923"
PUBLIC_SHA = "04d48fc44a4b74f4335e21794e88a95ed1d07e82a6d7a8b45b7309a6a006b81c"
TARGETS = fixed.REPO / "outputs/audits/structure-expansion-all-style-targets-20260923/report.json"
TARGETS_SHA = "a8d742c76a99915850ad92135567c9387e05eedc8c9b69f62f74403d58ac4251"
PROFILES = ("oa00", "oa22")
MAX_CASE_BYTES = 256 * 1024**2


def select_panel(candidates, molecules, *, per_kind=8):
    fixed.require(type(per_kind) is int and per_kind > 0, "Invalid panel count")
    registry = {row["molecule_id"]:row for row in molecules}
    fixed.require(len(registry) == len(molecules), "Duplicate source molecule")
    features, by_id = {}, {}
    for row in candidates:
        mid = row["molecule_id"]
        fixed.require(mid in registry and mid not in by_id, "Foreign/duplicate candidate")
        source = registry[mid]
        fixed.require(row["source"] == "curriculum" and row["canonical_smiles"] == source["canonical_smiles"]
            and source["training_authorized"] is False and source["split"] == "quarantine"
            and row["image_count"] == len(set(row["sample_keys"]))
            and all(mid + "__" + profile in row["sample_keys"] for profile in PROFILES),
            "Candidate identity/quarantine or paired styles changed")
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        fixed.require(mol is not None, "Invalid canonical molecule")
        features[mid] = (mol.GetNumAtoms(), {str(b.GetBondType()) for b in mol.GetBonds()})
        by_id[mid] = row
    selected, used = [], set()
    for kind in metrics.KINDS:
        eligible = [mid for mid in by_id if mid not in used and kind in features[mid][1]]
        eligible.sort(key=lambda mid: (features[mid][0], sha256(f"20260923:{kind}:{mid}".encode()).hexdigest()))
        fixed.require(len(eligible) >= per_kind, "Insufficient fixed bond-type coverage")
        # Midpoints of equally sized rank intervals: spread size, no outcome selection.
        indices = [((2*i+1)*len(eligible)) // (2*per_kind) for i in range(per_kind)]
        fixed.require(len(set(indices)) == per_kind, "Duplicate panel rank")
        for index in indices:
            mid = eligible[index]
            for profile in PROFILES:
                selected.append(dict(molecule_id=mid, canonical_smiles=by_id[mid]["canonical_smiles"],
                    sample_key=mid + "__" + profile, profile_id=profile, coverage_bucket=kind,
                    atoms=features[mid][0], selected_rank=index, eligible_structures=len(eligible),
                    purpose="diagnostic", training_authorized=False))
            used.add(mid)
    return selected


def input_plan():
    fixed.require(fixed.digest(public.OUTPUT / "report.json") == PUBLIC_SHA
                  and fixed.digest(TARGETS) == TARGETS_SHA, "Completed screening/preflight changed")
    report = json.loads(fixed.read(public.OUTPUT / "report.json"))
    public_plan = json.loads(fixed.read(public.OUTPUT / "plan.json"))
    target_report = json.loads(fixed.read(TARGETS))
    fixed.require(report["complete"] is True and report["remaining_ambiguous_pairs"] == 0
                  and target_report["complete"] is True and target_report["failed_images"] == 0
                  and target_report["all_retained_styles_collated"] is True
                  and fixed.digest(public.OUTPUT / "plan.json") == report["plan_sha256"], "Incomplete source evidence")
    bindings = dict(public_plan["input_sha256"])
    for name, value in report["artifact_sha256"].items():
        fixed.require(Path(name).name == name, "Unsafe public artifact")
        bindings[str(public.OUTPUT / name)] = value
    bindings.update({str(public.OUTPUT / "report.json"):PUBLIC_SHA, str(TARGETS):TARGETS_SHA})
    fixed.verify_bindings(bindings)
    candidates = json.loads(fixed.read(public.OUTPUT / "remaining-candidates.json"))
    fixed.require(len(candidates) == report["remaining_molecules"]
                  and sum(row["image_count"] for row in candidates) == report["remaining_images"], "Screened count mismatch")
    panel = select_panel(candidates, json.loads(fixed.read(fixed.DATASET / "molecules.json")))
    hashes = json.loads(fixed.read(fixed.DATASET / "sample-sha256.json"))
    for row in panel:
        for suffix in (*data.PAYLOADS, ".meta.json"):
            name = "samples/" + row["sample_key"] + suffix
            bindings[str(fixed.DATASET / name)] = hashes[name]
    paths = {Path(__file__).absolute(), metrics.RUN / "initial.pt", metrics.RUN / "run.json", Path(metrics.__file__),
             fixed.REPO / "tests/data_sdf/test_structure_bond_diagnostic.py",
             *(Path(module.__file__) for module in (bundle, continuation, data, pixels, supervision_contract, upstream_diagnostic)),
             *(Path(inspect.getfile(fn)) for fn in (metrics.state_digest, metrics.tensor_tree_digest,
                metrics.match_atom_targets, metrics.enumerate_unordered_atom_pairs,
                metrics.promote_complete_aromatic_rings, AtomBondSceneConfig))}
    paths.update(path for folder in (fixed.REPO / "chemvector/components", fixed.REPO / "chemvector/core")
                 for path in folder.rglob("*.py"))
    for path in paths:
        value = fixed.digest(path)
        fixed.require(str(path) not in bindings or bindings[str(path)] == value, "Diagnostic source changed")
        bindings[str(path)] = value
    fixed.require(bindings[str(metrics.RUN / "initial.pt")] == metrics.INITIAL_SHA, "Initial checkpoint changed")
    fixed.verify_bindings(bindings)
    provenance = json.loads(fixed.read(metrics.RUN / "run.json"))["provenance"]
    return dict(schema_version="structure-expansion-bond-diagnostic-plan-1", samples=panel,
        planned_images=64, planned_structures=32, planned_structure_forwards=64, input_sha256=bindings,
        initial_sha256=metrics.INITIAL_SHA, expected_initial_state=metrics.INITIAL_STATE,
        selection="8 distinct size-rank interval midpoints per TRIPLE/DOUBLE/AROMATIC/SINGLE; paired oa00/oa22; no outcome selection",
        thresholds={key:value for key,value in provenance.items() if key.endswith("_threshold")},
        max_cpu_seconds=300, automatic_retries=0, gpu_used=False, optimizer_steps=0, training_authorized=False,
        private_identity_queries=0, ocr_loaded=False, whole_molecule_accuracy_measured=False,
        provenance=provenance)


def run():
    fixed.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "CPU-only diagnostic")
    fixed.require(not OUTPUT.exists() and OUTPUT.resolve() == OUTPUT, "Fresh diagnostic output required")
    torch.set_num_threads(2)
    torch.manual_seed(20260923)
    started = time.monotonic()
    plan = input_plan()
    panel, provenance = plan["samples"], plan["provenance"]
    fixed.require(len(panel) == 64 and len({r["molecule_id"] for r in panel}) == 32, "Fixed panel mismatch")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    fixed.write_new(OUTPUT / "plan.json", plan)  # Frozen before model loading.
    report = dict(schema_version="structure-expansion-bond-diagnostic-result-1", status="incomplete", complete=False,
        plan_sha256=fixed.digest(OUTPUT / "plan.json"), model_loads=0, structure_forwards=0, pair_forwards=0,
        training_authorized=False, training_performed=False, gpu_used=False, optimizer_steps=0,
        private_identity_queries=0, ocr_loaded=False, whole_molecule_accuracy_measured=False, automatic_retries=0)
    cases, case_bytes = [], 0
    try:
        model, config, payload = bundle.load_bundle(metrics.RUN / "initial.pt", expected_sha256=metrics.INITIAL_SHA)
        report["model_loads"] = 1
        fixed.require(payload["provenance"] == provenance and payload["progress"]["optimizer_steps"] == 0
                      and metrics.state_digest(model) == metrics.INITIAL_STATE, "Initial state/provenance mismatch")
        model.eval().requires_grad_(False)
        collate = data.CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
            bond_queries=config.maximum_bonds), purpose="diagnostic")
        with torch.inference_mode():
            for row in panel:
                sample = data.load_sample(fixed.DATASET, row["sample_key"], purpose="diagnostic")
                fixed.require(sample["graph"]["canonical_smiles"] == row["canonical_smiles"]
                    and sample["meta"]["molecule_id"] == row["molecule_id"], "Panel identity mismatch")
                report["structure_forwards"] += 1
                forward = continuation.structure_forward(model, pixels.image_tensor(sample["image"]).unsqueeze(0),
                                                         config, provenance["atom_threshold"])
                report["pair_forwards"] += 1
                predicted = metrics.predict_pair(model, forward[0], forward[6], provenance["pair_threshold"])
                before_score = metrics.prediction_digest(forward[3], forward[6], predicted)
                # Ground truth enters matching/metrics only after predictions are frozen.
                measured = metrics.measure(forward[3], forward[6], predicted, collate([sample]), config, provenance["pair_threshold"])
                fixed.require(metrics.prediction_digest(forward[3], forward[6], predicted) == before_score,
                              "Scoring mutated predictions")
                case = {**row, "metrics":measured, "prediction_sha256":before_score,
                    "candidates":upstream_diagnostic.candidate_trace(forward[3], forward[6], forward[5]),
                    "pair_predictions":dict(pairs=predicted["pairs"].tolist(),
                        presence_logits=predicted["output"]["presence_logits"].tolist(),
                        chemical_logits=predicted["output"]["chemical_logits"].tolist(),
                        promoted_chemical_ids=predicted["chemistry"].tolist())}
                case_bytes += len(fixed.encoded(case))
                fixed.require(case_bytes <= MAX_CASE_BYTES, "Diagnostic case artifact budget exceeded")
                fixed.write_new(OUTPUT / (row["sample_key"] + ".json"), case)
                cases.append(case)
                print(json.dumps({"checked":len(cases), "total":64, "stages":measured["bond_stages"]}), flush=True)
        after = metrics.state_digest(model)
        fixed.require(after == metrics.INITIAL_STATE and not torch.cuda.is_initialized()
            and all(not p.requires_grad and p.grad is None for p in model.parameters()), "Frozen state/device guard failed")
        fixed.verify_bindings(plan["input_sha256"])
        fixed.require(len(cases) == 64 and time.monotonic()-started < 300, "Incomplete panel or CPU deadline")
        report.update(complete=True, status="completed_diagnostic_only", summary=metrics.summarize(cases),
            by_profile={profile:metrics.summarize([c for c in cases if c["profile_id"] == profile]) for profile in PROFILES},
            states_before=metrics.INITIAL_STATE, states_after=after, config=asdict(config),
            limitations=["Targeted panel with repeated structures; not random/held-out accuracy or independent images.",
                "Profiles differ in several style factors; differences are not single-factor causal effects.",
                "Original Hungarian assignment and fixed radius; correspondence gaps need geometry inspection.",
                "No OCR/element/charge/isotope/stereo performance or final SDF compiler assessment.",
                "Unknown pair correspondences are not confirmed negatives; no training admission."])
    except Exception as exc:
        report.update(status="failed_or_incomplete", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, checked_images=len(cases), case_bytes=case_bytes,
            case_sha256={row["sample_key"]:fixed.digest(OUTPUT/(row["sample_key"]+".json")) for row in cases})
        fixed.write_new(OUTPUT / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        return 0 if run()["complete"] else 1
    result = run_budgeted([sys.executable,"-m","tools.diagnose_structure_expansion_bonds","--worker"],
        output=OUTPUT.with_name(OUTPUT.name+"-runtime"), budget=ResourceBudget(300))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
