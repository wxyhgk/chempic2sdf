"""Fixed 32-image, initial-only CPU bond diagnostic; no training or OCR.

Labels select a declared coverage panel and score completed predictions only.
They never provide model input atoms, graph edges, crops or repairs. This is a
targeted diagnostic, not a held-out whole-molecule accuracy measurement.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import time

from rdkit import Chem
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.bond_object import BOND_TYPES
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.direct_atom_pair_chemistry import promote_complete_aromatic_rings
from chemvector.train_direct_atom_pair_bond import match_atom_targets
from chemvector.training.curriculum import bundle, continuation, data, pixels, supervision_contract, upstream_diagnostic
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.interface_diagnostic import tensor_tree_digest
from chemvector.training.curriculum.release_evidence import verify_reviews
from tools import review_curriculum_v02_private_identity as fixed
from tools import review_curriculum_v02_public_identity as public

DATA = fixed.REPO / "outputs/datasets/curriculum-v0.2-expanded-20260919"
RUN = fixed.REPO / "outputs/experiments/chemvector-selected-matching-trial-20260915"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
INITIAL_STATE = "9d7f66f79329479e05c38f8b91d95118adb9f12c32a44e5bd9d83a458a8b452e"
OUTPUT = fixed.REPO / "outputs/audits/curriculum-v02-bond-diagnostic32-20260919"
KINDS = ("TRIPLE", "DOUBLE", "AROMATIC", "SINGLE")
STAGES = ("raw_correspondence_gap", "selection_correspondence_gap", "reachable_missing",
          "reachable_wrong_type", "reachable_correct")


def select_panel(candidates, molecules, *, per_kind=8):
    if type(per_kind) is not int or per_kind <= 0:
        raise ValueError("Invalid fixed coverage count")
    registry = {row["molecule_id"]: row for row in molecules}
    if len(registry) != len(molecules) or len({row["molecule_id"] for row in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate identity")
    features = {}
    for row in candidates:
        mid = row["molecule_id"]
        source = registry[mid]
        if (row["source"] != "curriculum" or row["canonical_smiles"] != source["canonical_smiles"]
                or source["training_authorized"] is not False or source["split"] != "quarantine"
                or row["image_count"] != len(set(row["sample_keys"])) or row["image_count"] < 1):
            raise ValueError("Candidate is not an exact quarantined source")
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is None:
            raise ValueError("Invalid candidate identity")
        features[mid] = (mol.GetNumAtoms(), {str(b.GetBondType()) for b in mol.GetBonds()})
    selected, used = [], set()
    for kind in KINDS:
        eligible = [row for row in candidates if row["molecule_id"] not in used and kind in features[row["molecule_id"]][1]]
        eligible.sort(key=lambda row: (features[row["molecule_id"]][0],
            sha256(f"20260919:{kind}:{row['molecule_id']}".encode()).hexdigest()))
        if len(eligible) < per_kind:
            raise ValueError(f"Insufficient fixed coverage for {kind}")
        for row in eligible[:per_kind]:
            mid = row["molecule_id"]
            profile_priority = {name: i for i, name in enumerate(("reference", "font_large", "font_small", "serif", "line_thin", "bond_short"))}
            for key in row["sample_keys"]:
                if not key.startswith(mid + "__") or key[len(mid) + 2:] not in profile_priority:
                    raise ValueError("Unexpected candidate profile/key")
            key = min(row["sample_keys"], key=lambda value: profile_priority[value[len(mid) + 2:]])
            selected.append(dict(molecule_id=mid, canonical_smiles=row["canonical_smiles"], sample_key=key,
                                 profile_id=key[len(mid) + 2:], coverage_bucket=kind,
                                 atoms=features[mid][0], purpose="diagnostic", training_authorized=False))
            used.add(mid)
    return selected


@torch.no_grad()
def predict_pair(model, features, selected, threshold):
    # Same pair enumeration and aromatic promotion as FrozenOCRPredictor's
    # from_predicted_atoms, without instantiating/invoking OCR or the compiler.
    pairs, rows = enumerate_unordered_atom_pairs(selected["atom_mask"])
    output = model.pair(features, selected["atom_features"], selected["atom_geometry"][..., :2],
                        selected["atom_scores"], pairs, rows)
    if any(not bool(torch.isfinite(output[key]).all()) for key in ("presence_logits", "chemical_logits")):
        raise ValueError("Nonfinite pair predictions")
    chemistry, _ = promote_complete_aromatic_rings(pairs, output["presence_logits"].sigmoid(),
        output["chemical_logits"], presence_threshold=threshold, mean_threshold=.40, minimum_threshold=.05)
    return dict(pairs=pairs, output=output, chemistry=chemistry)


@torch.no_grad()
def measure(raw, selected, predicted, batch, config, threshold):
    count = len(predicted["pairs"])
    if (not 0 < threshold < 1 or predicted["output"]["presence_logits"].shape != (count,)
            or predicted["output"]["chemical_logits"].shape != (count, len(BOND_TYPES))
            or predicted["chemistry"].shape != (count,)
            or any(not bool(torch.isfinite(predicted["output"][key]).all()) for key in ("presence_logits", "chemical_logits"))
            or bool(((predicted["chemistry"] < 0) | (predicted["chemistry"] >= len(BOND_TYPES))).any())):
        raise ValueError("Invalid fixed pair predictions/threshold")
    coverage = upstream_diagnostic.labeled_coverage(raw, selected, batch, config)
    if bool((selected["atom_mask"] & ~raw["atom_mask"]).any()):
        raise ValueError("Selector introduced nonpredicted atoms")
    _, raw_matched = match_atom_targets(raw, batch, config)
    mapping, _ = match_atom_targets(selected, batch, config)
    contract = supervision_contract.pair_targets_with_unknown(selected, mapping, batch)
    if not torch.equal(contract["pair_indices"], predicted["pairs"]):
        raise ValueError("Predicted pair enumeration changed")
    slots = {int(mapping[0, c]): c for c in selected["atom_mask"][0].nonzero().flatten().tolist()
             if int(mapping[0, c]) >= 0}
    pair_index = {tuple(pair): i for i, pair in enumerate(predicted["pairs"].tolist())}
    probabilities = predicted["output"]["presence_logits"].sigmoid()
    raw_types = predicted["output"]["chemical_logits"].argmax(-1)
    bonds = []
    for i in range(int(batch["bond_mask"].sum())):
        left, right = batch["bond_atom_refs"][0, i].tolist()
        index = None if left not in slots or right not in slots else pair_index[tuple(sorted((slots[left], slots[right])))]
        expected = BOND_TYPES[int(batch["bond_chemical"][0, i])]
        kind = None if index is None else BOND_TYPES[int(predicted["chemistry"][index])]
        present = index is not None and bool(probabilities[index] >= threshold)
        stage = ("selection_correspondence_gap" if bool(raw_matched[0, left] and raw_matched[0, right])
                 else "raw_correspondence_gap") if index is None else (
                     "reachable_missing" if not present else "reachable_correct" if kind == expected else "reachable_wrong_type")
        bonds.append(dict(bond_id=batch["bond_ids"][0][i], target_slots=[left, right], expected_type=expected,
                          candidates=[slots.get(left), slots.get(right)], stage=stage,
                          presence_probability=None if index is None else float(probabilities[index]),
                          predicted_type=kind, raw_type_argmax=None if index is None else BOND_TYPES[int(raw_types[index])]))
    present = probabilities >= threshold
    negatives = contract["known_mask"] & ~contract["legacy_targets"]["presence"]
    return dict(coverage=coverage, pair_contract=contract["coverage"], bonds=bonds,
                bond_stages={stage: sum(row["stage"] == stage for row in bonds) for stage in STAGES},
                confirmed_false_positive_pairs=int((present & negatives).sum()),
                unknown_present_pairs=int((present & contract["unknown_mask"]).sum()))


def summarize(cases):
    if not cases or len({row["sample_key"] for row in cases}) != len(cases):
        raise ValueError("Missing/duplicate diagnostic cases")
    bonds = [bond for case in cases for bond in case["metrics"]["bonds"]]
    by_kind = {}
    for kind in KINDS:
        rows = [bond for bond in bonds if bond["expected_type"] == kind]
        by_kind[kind] = dict(targets=len(rows), **{stage: sum(row["stage"] == stage for row in rows) for stage in STAGES})
    counts = {name: sum(case["metrics"]["coverage"][name] for case in cases) for name in (
        "target_atoms", "raw_matched", "selected_matched", "selected_atoms", "unmatched_selected", "target_bonds", "reachable_positive_pairs")}
    return dict(images=len(cases), coverage=counts, by_chemical_type=by_kind,
                bond_stages={stage: sum(row["stage"] == stage for row in bonds) for stage in STAGES},
                pair_contract=dict(sum((Counter(case["metrics"]["pair_contract"]) for case in cases), Counter())),
                confirmed_false_positive_pairs=sum(case["metrics"]["confirmed_false_positive_pairs"] for case in cases),
                unknown_present_pairs=sum(case["metrics"]["unknown_present_pairs"] for case in cases))


def prediction_digest(raw, selected, predicted):
    # The existing utility recursively supports dictionaries, not tuples/lists.
    return tensor_tree_digest({"raw": raw, "selected": selected, "predicted": predicted})


def run(output, *, max_seconds=300):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only diagnostic required")
    if type(max_seconds) is not int or not 0 < max_seconds <= 300:
        raise ValueError("At most 300 CPU wall seconds")
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise FileExistsError("Fresh diagnostic directory required")
    torch.set_num_threads(2)
    torch.manual_seed(20260919)
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError("Diagnostic CPU deadline reached")
    sources = {"request": fixed.REQUEST / "request.json", "request_candidates": fixed.REQUEST / "candidate-identities.json",
               "public_report": public.OUTPUT / "report.json", "remaining": public.OUTPUT / "remaining-candidates.json",
               "private_receipt": fixed.OUTPUT / "receipt.json"}
    raw = {name: fixed.read(path) for name, path in sources.items()}
    docs = {name: json.loads(payload) for name, payload in raw.items()}
    verify_reviews(raw, docs)  # Saved receipts only. No identity service calls.
    public_plan = json.loads(fixed.read(public.OUTPUT / "plan.json"))
    bindings = dict(public_plan["input_sha256"])
    if docs["public_report"]["plan_sha256"] != fixed.digest(public.OUTPUT / "plan.json"):
        raise ValueError("Public sidecar plan binding mismatch")
    for name, expected in docs["public_report"]["artifact_sha256"].items():
        if Path(name).name != name or fixed.digest(public.OUTPUT / name) != expected:
            raise ValueError("Public result artifact mismatch")
    molecules = json.loads(fixed.read(DATA / "molecules.json"))
    panel = select_panel(docs["remaining"], molecules)
    inventory = json.loads(fixed.read(DATA / "sample-sha256.json"))
    for row in panel:
        for suffix in (*data.PAYLOADS, ".meta.json"):
            name = f"samples/{row['sample_key']}{suffix}"
            bindings[str(DATA / name)] = inventory[name]
    paths = [*sources.values(), public.OUTPUT / "plan.json", Path(__file__), RUN / "run.json",
             RUN / "initial.pt", Path(fixed.__file__), Path(public.__file__),
             *(Path(module.__file__) for module in (bundle, continuation, data, pixels, supervision_contract, upstream_diagnostic)),
             *(Path(inspect.getfile(fn)) for fn in (state_digest, tensor_tree_digest, match_atom_targets,
                 enumerate_unordered_atom_pairs, promote_complete_aromatic_rings, AtomBondSceneConfig, verify_reviews))]
    # Bind the project implementations that define the loaded model classes.
    paths.extend(path for folder in (fixed.REPO / "chemvector/components", fixed.REPO / "chemvector/core")
                 if folder.is_dir() for path in folder.rglob("*.py"))
    for path in paths:
        value = fixed.digest(path)
        if str(path) in bindings and bindings[str(path)] != value:
            raise ValueError("Changed diagnostic dependency")
        bindings[str(path)] = value
    fixed.verify_bindings(bindings)
    if bindings[str(RUN / "initial.pt")] != INITIAL_SHA:
        raise ValueError("Initial checkpoint changed")
    provenance = json.loads(fixed.read(RUN / "run.json"))["provenance"]
    plan = dict(schema_version="chemvector-v02-bond-diagnostic-plan-0.1", samples=panel, planned_images=32,
                input_sha256=bindings, initial_sha256=INITIAL_SHA, expected_initial_state=INITIAL_STATE,
                selection="8 distinct smallest structures per TRIPLE/DOUBLE/AROMATIC/SINGLE, stable hash tie-break; surviving reference preferred",
                thresholds={key: value for key, value in provenance.items() if key.endswith("_threshold")},
                max_cpu_seconds=max_seconds, planned_structure_forwards=32, automatic_retries=0,
                gpu_used=False, optimizer_steps=0, training_authorized=False, private_identity_queries=0,
                ocr_loaded=False, whole_molecule_accuracy_measured=False)
    output.mkdir(parents=True, exist_ok=False)
    public.write(output / "plan.json", plan)
    cases = []
    report = dict(schema_version="chemvector-v02-bond-diagnostic-result-0.1", status="incomplete", complete=False,
                  plan_sha256=fixed.digest(output / "plan.json"), training_authorized=False, training_performed=False,
                  gpu_used=False, optimizer_steps=0, private_identity_queries=0, ocr_loaded=False,
                  whole_molecule_accuracy_measured=False, automatic_retries=0)
    try:
        deadline()
        model, config, payload = bundle.load_bundle(RUN / "initial.pt", expected_sha256=INITIAL_SHA)
        if payload["provenance"] != provenance or payload["progress"]["optimizer_steps"] != 0 or state_digest(model) != INITIAL_STATE:
            raise ValueError("Unexpected initial provenance/state")
        model.eval().requires_grad_(False)
        before = state_digest(model)
        collator = data.CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                    bond_queries=config.maximum_bonds), purpose="diagnostic")
        with torch.inference_mode():
            for row in panel:
                deadline()
                sample = data.load_sample(DATA, row["sample_key"], purpose="diagnostic")
                meta = sample["meta"]
                if (meta["profile_id"] != row["profile_id"] or meta["molecule_id"] != row["molecule_id"]
                        or sample["graph"]["canonical_smiles"] != row["canonical_smiles"]
                        or (meta["split"], meta["training_authorized"], meta["external_overlap_status"]) != ("quarantine", False, "unchecked")):
                    raise ValueError("Panel sample scope/identity mismatch")
                forward = continuation.structure_forward(model, pixels.image_tensor(sample["image"]).unsqueeze(0),
                                                         config, provenance["atom_threshold"])
                predicted = predict_pair(model, forward[0], forward[6], provenance["pair_threshold"])
                saved = prediction_digest(forward[3], forward[6], predicted)
                batch = collator([sample])  # Truth first enters AFTER image-only predictions.
                metrics = measure(forward[3], forward[6], predicted, batch, config, provenance["pair_threshold"])
                if prediction_digest(forward[3], forward[6], predicted) != saved:
                    raise ValueError("Scoring changed predictions")
                case = {**row, "metrics": metrics, "candidates": upstream_diagnostic.candidate_trace(forward[3], forward[6], forward[5])}
                public.write(output / (row["sample_key"] + ".json"), case)
                cases.append(case)
                print(json.dumps({"checked": len(cases), "total": 32, "bond_stages": metrics["bond_stages"]}), flush=True)
        after = state_digest(model)
        if before != after or torch.cuda.is_initialized() or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("Frozen model/device guard failed")
        fixed.verify_bindings(bindings)
        deadline()
        if len(cases) != 32:
            raise ValueError("Incomplete fixed panel")
        report.update(status="completed_diagnostic_only", complete=True, summary=summarize(cases),
                      states_before=before, states_after=after, config=asdict(config),
                      limitations=["Targeted small-structure panel, not random/held-out accuracy or a training release.",
                                   "Matching uses original Hungarian assignment and fixed radius; gaps are not proof that no visual atom exists.",
                                   "Chemical prediction is before final compiler consistency, not whole-molecule identity or stereo correctness.",
                                   "No OCR/element, charge, isotope or stereochemistry performance assessed.",
                                   "Unknown correspondences remain unknown, never counted as confirmed negatives."])
    except Exception as exc:
        report.update(status="failed_or_incomplete", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, checked_images=len(cases),
                      case_sha256={row["sample_key"]: fixed.digest(output / (row["sample_key"] + ".json")) for row in cases})
        public.write(output / "report.json", report)
    return report


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    result = run(OUTPUT)
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
