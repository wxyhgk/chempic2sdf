"""CPU-only replay of fixed saved proposals, not a model or training runner.

Original assignments/metrics remain authoritative. Radius-constrained matching
is an explicitly labelled target-contract counterfactual, not new predictions.
Scalar derivatives use detached cached logits only, never model parameters.
"""

import argparse
from collections import Counter
from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.predicted_incidence import PredictedProposalConfig, associate_predicted_proposals
from chemvector.proposal_object_set import _focal_presence_loss, _set_ranking_loss
from chemvector.train_direct_atom_pair_bond import match_atom_targets
from chemvector.training.curriculum import data, supervision_contract, upstream_diagnostic

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "outputs/audits/curriculum-v02-bond-diagnostic32-corrected-20260919"
DATA = REPO / "outputs/datasets/curriculum-v0.2-expanded-20260919"
OUTPUT = REPO / "outputs/audits/curriculum-v02-selector-matching-replay-20260919"
PLAN_SHA = "5a312bf999a31fbbac328bec5ec60965505dff2db6693050707e8c5382082998"
REPORT_SHA = "064cc28d5d2d5729a06b08e5127cfa6e10771498a1fe8ce5522dc2def54bf3a8"


def checked_path(path):
    path = Path(path).absolute()
    if "sealed_test" in path.parts or path.resolve() != path or not path.is_file():
        raise ValueError("Expected explicit nonsealed nonsymlink saved input")
    return path


def digest(path):
    result = sha256()
    with checked_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_bound(path, expected):
    payload = checked_path(path).read_bytes()
    if sha256(payload).hexdigest() != expected:
        raise ValueError("Saved artifact SHA mismatch")
    return json.loads(payload)


def verify_bindings(bindings, deadline):
    for path, expected in bindings.items():
        deadline()
        if digest(path) != expected:
            raise ValueError("Replay input changed: " + str(path))


def restore_candidates(trace, capacity, threshold):
    """Slot IDs are shape-only IDs, NOT recovered original peak-grid indices."""
    if type(capacity) is not int or capacity < 1 or not math.isfinite(threshold):
        raise ValueError("Invalid replay configuration")
    geometry = torch.zeros(1, capacity, 8)
    mask = torch.zeros(1, capacity, dtype=torch.bool)
    selected = mask.clone()
    scores, logits = torch.zeros(1, capacity), torch.zeros(1, capacity)
    for row in trace["candidates"]:
        slot = row["candidate"]
        values = row["geometry"]
        if (type(slot) is not int or not 0 <= slot < capacity or mask[0, slot]
                or len(values) != 8 or not all(math.isfinite(v) for v in values)
                or not all(0 <= v <= 1 for v in values[:2])
                or not math.isfinite(row["selection_logit"])
                or not 0 <= row["proposal_score"] <= 1 or type(row["selected"]) is not bool):
            raise ValueError("Invalid/duplicate saved candidate")
        geometry[0, slot] = torch.tensor(values, dtype=torch.float32)
        mask[0, slot] = True
        selected[0, slot] = row["selected"]
        logits[0, slot] = row["selection_logit"]
        scores[0, slot] = row["proposal_score"]
    if (int(mask.sum()) != trace["raw_count"] or int(selected.sum()) != trace["selected_count"]
            or not torch.equal(selected, mask & (logits >= threshold))):
        raise ValueError("Saved counts/selector decisions disagree")
    raw = dict(atom_geometry=geometry, atom_mask=mask, atom_scores=scores,
               atom_indices=torch.arange(capacity).unsqueeze(0))
    return raw, {**raw, "atom_mask": selected}, logits


def radius_assignment(cost, radius):
    """Maximize valid cardinality first, then minimize in-radius distance.

    Unmatched dummy columns cost k+1, while at most k real edges each cost <=1.
    Thus one extra valid edge always outweighs any possible distance tradeoff.
    This standalone offline comparator is deliberately NOT used by training.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if (cost.ndim != 2 or not np.isfinite(cost).all() or (cost < 0).any()
            or not math.isfinite(radius) or radius <= 0):
        raise ValueError("Invalid matching distances/radius")
    n, m = cost.shape
    if not n or not m:
        return []
    penalty = min(n, m) + 1.0
    augmented = np.full((n, m + n), penalty)
    augmented[:, :m] = np.where(cost <= radius, cost / radius, np.inf)
    rows, columns = linear_sum_assignment(augmented)
    return [(int(r), int(c)) for r, c in zip(rows, columns) if c < m]


def atom_loss_probe(logits, labels, mask):
    """Only d(atom focal + .5 ranking + .1 count)/d(cached logit).

    Per-image, eval-cache diagnostic; not the full set loss, model gradient,
    training-mode supervision history, or an optimizer step.
    """
    values = logits.detach().clone().requires_grad_(True)
    focal = _focal_presence_loss(values, labels, mask)
    ranking = _set_ranking_loss(values, labels, mask)
    count = F.smooth_l1_loss((values.sigmoid() * mask).sum(-1), labels.sum(-1).float())
    losses = dict(focal=focal, weighted_ranking=.5 * ranking, weighted_count=.1 * count)
    losses["atom_only_total"] = sum(losses.values())
    gradients = {key: torch.autograd.grad(value, values, retain_graph=True)[0].detach()
                 for key, value in losses.items()}
    return dict(losses={key: float(value.detach()) for key, value in losses.items()},
                target_count=int(labels.sum()), sigmoid_count=float((values.sigmoid() * mask).sum().detach())), gradients


def compare_assignment(atoms, batch, config):
    ids = atoms["atom_mask"][0].nonzero().flatten()
    target_count = int(batch["atom_mask"].sum())
    cost = (torch.cdist(atoms["atom_geometry"][0, ids, :2].float(),
                        batch["atom_anchors"][0, :target_count].float()) * config.image_size).numpy()
    rows, columns = linear_sum_assignment(cost)
    original = [dict(candidate=int(ids[r]), target=int(c), distance_px=float(cost[r, c]),
                     accepted=bool(cost[r, c] <= config.atom_match_radius_px)) for r, c in zip(rows, columns)]
    alternative = [dict(candidate=int(ids[r]), target=c, distance_px=float(cost[r, c]))
                   for r, c in radius_assignment(cost, config.atom_match_radius_px)]
    old = {(r["candidate"], r["target"]) for r in original if r["accepted"]}
    new = {(r["candidate"], r["target"]) for r in alternative}
    result = dict(original_matched=len(old), radius_first_matched=len(new), mapping_changed=old != new,
                  original_assignment_before_gate=original, radius_first_assignment=alternative)
    if old != new:
        result.update(candidate_slots=ids.tolist(), distance_matrix_px=cost.tolist())
    return result


def analyze_case(case, batch, config, threshold):
    raw, selected, logits = restore_candidates(case["candidates"], config.maximum_atoms, threshold)
    coverage = upstream_diagnostic.labeled_coverage(raw, selected, batch, config)
    if coverage != case["metrics"]["coverage"]:
        raise ValueError("Original saved coverage did not reproduce")
    raw_mapping, raw_matched = match_atom_targets(raw, batch, config)
    mapping, matched = match_atom_targets(selected, batch, config)
    # Empty bonds replay ONLY the actual associate_predicted_proposals atom
    # branch: cached bond proposals/incidence features do not exist here.
    no_bonds = dict(bond_mask=torch.zeros(1, 0, dtype=torch.bool), bond_indices=torch.empty(1, 0, dtype=torch.long))
    labels = associate_predicted_proposals(raw, no_bonds, batch, config)
    if (not torch.equal(labels["atom_target_indices"], raw_mapping)
            or not torch.equal(labels["atom_target_matched"], raw_matched)
            or not torch.equal(labels["atom_presence_targets"], raw_mapping >= 0)):
        raise ValueError("Raw selector supervision diverges from matcher replay")
    contract = supervision_contract.pair_targets_with_unknown(selected, mapping, batch)
    if contract["coverage"] != case["metrics"]["pair_contract"]:
        raise ValueError("Saved unknown/known pair contract did not reproduce")
    target_to_slot = {int(mapping[0, i]): i for i in selected["atom_mask"][0].nonzero().flatten().tolist()
                      if int(mapping[0, i]) >= 0}
    for row in case["metrics"]["bonds"]:
        if row["candidates"] != [target_to_slot.get(t) for t in row["target_slots"]]:
            raise ValueError("Saved bond endpoint assignment did not reproduce")
    positive = labels["atom_presence_targets"]
    rejected = positive & ~selected["atom_mask"]
    retained_negative = selected["atom_mask"] & ~positive
    probe, gradients = atom_loss_probe(logits, positive, raw["atom_mask"])
    positive_values, negative_values = logits[positive], logits[raw["atom_mask"] & ~positive]
    rank_inversion = bool(len(positive_values) and len(negative_values) and negative_values.max() > positive_values.min())
    counts = dict(raw_candidates=int(raw["atom_mask"].sum()), raw_positive=int(positive.sum()),
                  raw_positive_rejected=int(rejected.sum()), raw_negative_selected=int(retained_negative.sum()),
                  rejected_positive_atom_only_gradient_up=int((rejected & (gradients["atom_only_total"] < 0)).sum()),
                  rejected_positive_atom_only_gradient_down=int((rejected & (gradients["atom_only_total"] > 0)).sum()),
                  images_with_rank_inversion=int(rank_inversion), lost_targets_no_nearby=0,
                  lost_targets_with_nearby=0, recovered_targets=0, targets_with_nearby_duplicates=0)
    details = []
    relevant = set(selected["atom_mask"][0].nonzero().flatten().tolist())
    for row in coverage["target_details"]:
        t = row["target"]
        raw_slots = (raw_mapping[0] == t).nonzero().flatten().tolist()
        selected_slots = (mapping[0] == t).nonzero().flatten().tolist()
        lost = row["raw_assignment_matched"] and not row["selected_assignment_matched"]
        counts["lost_targets_with_nearby" if row["near_selected_ids"] else "lost_targets_no_nearby"] += int(lost)
        counts["recovered_targets"] += int(not row["raw_assignment_matched"] and row["selected_assignment_matched"])
        counts["targets_with_nearby_duplicates"] += int(len(row["near_raw_ids"]) > 1)
        relevant.update(row["near_raw_ids"])
        details.append({**row, "raw_assigned_candidate": raw_slots[0] if raw_slots else None,
                        "selected_assigned_candidate": selected_slots[0] if selected_slots else None})
    candidates = []
    for i in sorted(relevant):
        distances = torch.linalg.vector_norm(raw["atom_geometry"][0, i, :2] -
            batch["atom_anchors"][0, :coverage["target_atoms"]], dim=-1) * config.image_size
        candidates.append(dict(candidate=i, selector_logit=float(logits[0, i]), selected=bool(selected["atom_mask"][0, i]),
            raw_positive=bool(positive[0, i]), raw_target=int(raw_mapping[0, i]), selected_target=int(mapping[0, i]),
            nearest_target=int(distances.argmin()), nearest_distance_px=float(distances.min()),
            atom_only_logit_derivatives={key: float(value[0, i]) for key, value in gradients.items()}))
    comparisons = {"raw": compare_assignment(raw, batch, config), "selected": compare_assignment(selected, batch, config)}
    for key, expected in (("raw", coverage["raw_matched"]), ("selected", coverage["selected_matched"])):
        if comparisons[key]["original_matched"] != expected:
            raise ValueError("Assignment trace disagrees with actual matcher")
    new_targets = {r["target"] for r in comparisons["selected"]["radius_first_assignment"]}
    counterfactual_bonds = sum(all(t in new_targets for t in row["target_slots"]) for row in case["metrics"]["bonds"])
    return dict(sample_key=case["sample_key"], canonical_smiles=case["canonical_smiles"],
                original_coverage=coverage, original_pair_contract=contract["coverage"], counts=counts,
                target_details=details, relevant_candidates=candidates, atom_only_loss_probe=probe,
                counterfactual_matching=comparisons, counterfactual_reachable_bonds=counterfactual_bonds)


def run(output=OUTPUT, *, max_seconds=60):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only saved replay required")
    if type(max_seconds) is not int or not 0 < max_seconds <= 60:
        raise ValueError("Replay limit must be at most 60 CPU wall seconds")
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise FileExistsError("Fresh nonsymlink audit directory required")
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError("Saved replay deadline exceeded")
    torch.set_num_threads(2)
    plan = read_bound(SOURCE / "plan.json", PLAN_SHA)
    original = read_bound(SOURCE / "report.json", REPORT_SHA)
    if (not original["complete"] or original["plan_sha256"] != PLAN_SHA
            or plan["planned_images"] != 32 or len(plan["samples"]) != 32
            or len({r["sample_key"] for r in plan["samples"]}) != 32
            or set(original["case_sha256"]) != {r["sample_key"] for r in plan["samples"]}):
        raise ValueError("Incomplete/changed fixed panel")
    bindings = {**plan["input_sha256"], str(SOURCE / "plan.json"): PLAN_SHA,
                str(SOURCE / "report.json"): REPORT_SHA}
    paths = [Path(__file__), REPO / "tests/data_sdf/test_curriculum_v2_selector_matching.py",
             *(Path(inspect.getfile(f)) for f in (associate_predicted_proposals, _focal_presence_loss, _set_ranking_loss,
                match_atom_targets, data.CurriculumCollator, upstream_diagnostic.labeled_coverage,
                supervision_contract.pair_targets_with_unknown))]
    for path in paths:
        name, value = str(path.absolute()), digest(path)
        if name in bindings and bindings[name] != value:
            raise ValueError("Changed original dependency")
        bindings[name] = value
    for key, value in original["case_sha256"].items():
        if not data.SAFE_KEY.fullmatch(key):
            raise ValueError("Unsafe saved sample key")
        bindings[str(SOURCE / (key + ".json"))] = value
    verify_bindings(bindings, deadline)
    config = PredictedProposalConfig(**original["config"])
    collator = data.CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
        bond_queries=config.maximum_bonds), purpose="diagnostic")
    cases = []
    for row in plan["samples"]:
        deadline()
        key = row["sample_key"]
        case = read_bound(SOURCE / (key + ".json"), original["case_sha256"][key])
        if any(case[k] != v for k, v in row.items()):
            raise ValueError("Saved sample differs from fixed panel")
        sample = data.load_sample(DATA, key, purpose="diagnostic")
        if sample["graph"]["canonical_smiles"] != case["canonical_smiles"]:
            raise ValueError("Sample identity differs from saved trace")
        cases.append(analyze_case(case, collator([sample]), config, plan["thresholds"]["atom_threshold"]))
    coverage = dict(sum((Counter({k: c["original_coverage"][k] for k in original["summary"]["coverage"]})
                         for c in cases), Counter()))
    if coverage != original["summary"]["coverage"]:
        raise ValueError("Original aggregate denominator did not reproduce")
    summary = dict(images=len(cases), original_coverage=coverage,
        counts=dict(sum((Counter(c["counts"]) for c in cases), Counter())),
        radius_first_raw_matched=sum(c["counterfactual_matching"]["raw"]["radius_first_matched"] for c in cases),
        radius_first_selected_matched=sum(c["counterfactual_matching"]["selected"]["radius_first_matched"] for c in cases),
        radius_first_reachable_bonds=sum(c["counterfactual_reachable_bonds"] for c in cases),
        raw_assignment_changed_images=[c["sample_key"] for c in cases if c["counterfactual_matching"]["raw"]["mapping_changed"]],
        selected_assignment_changed_images=[c["sample_key"] for c in cases if c["counterfactual_matching"]["selected"]["mapping_changed"]])
    verify_bindings(bindings, deadline)
    if torch.cuda.is_initialized():
        raise ValueError("Replay initialized CUDA")
    report = dict(schema_version="chemvector-v02-saved-selector-matching-0.1", complete=True,
        source_directory=str(SOURCE), input_sha256=bindings, original_input_bindings_verified=len(plan["input_sha256"]),
        scope="fixed eval-mode cached candidates; atom-only supervision replay and diagnostic scalar derivatives",
        gpu_used=False, model_loaded=False, model_forwards=0, optimizer_steps=0, training_authorized=False,
        private_identity_queries=0, thresholds_changed=False, original_predictions_modified=False,
        thresholds=plan["thresholds"], atom_match_radius_px=config.atom_match_radius_px,
        summary=summary, cases=cases, elapsed_seconds=time.monotonic() - started,
        limitations=["Targeted 32-image diagnostic; no held-out accuracy or generalization claim.",
            "Raw atom labels use the real association atom branch; missing cached bonds prevent full supervision replay.",
            "Slot IDs only reconstruct indexing shapes, not original peak-grid IDs or features.",
            "Derivatives are per-image atom-only cached-logit derivatives, not model/full-loss gradients or historic training labels.",
            "Radius-first comparator changes target correspondence only, not candidates, logits, thresholds or predictions.",
            "No pair predictions re-scored under counterfactual targets; original unknown pairs remain unknown."])
    output.mkdir(parents=True, exist_ok=False)
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    report = run()
    print(json.dumps({"summary": report["summary"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
