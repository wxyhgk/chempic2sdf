"""Four fixed CPU witnesses: additive selector loss-gradient attribution.

No optimizer, parameter update, new loss, threshold search or data release.
Contributions use ONE denominator (full-set gradient norm), so they add to
the previous full-objective direction. Positive/negative source splits are
chain-rule bookkeeping, NOT replacement training objectives.
"""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.predicted_incidence import associate_predicted_proposals
from chemvector.proposal_object_set import proposal_object_set_loss
from chemvector.training.curriculum import bundle, continuation, data, pixels, runtime, upstream_diagnostic
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.interface_diagnostic import tensor_tree_digest
from tools import diagnose_curriculum_v02_selector_gradients as previous

replay = previous.replay
PRIOR_SHA = "0b230bd34f12de1076e7c16d184440cb63ed3f6f43f81eb058a6f1ca335a7f70"
OUTPUT = replay.REPO / "outputs/audits/curriculum-v02-selector-loss-terms4-20260919"
WITNESSES = ("C#C", "C#[13CH]", "C#CC", "CCC#N")
WEIGHTS = {"atom_presence_loss": 1., "atom_ranking_loss": .5, "atom_count_loss": .1,
           "bond_presence_loss": 1., "bond_ranking_loss": .5, "bond_count_loss": .1, "incidence_loss": 1.}
WALL_SECONDS = 60


def select_witnesses(source_plan, saved_cases, label_cases):
    by_smiles = {r["canonical_smiles"]: r for r in source_plan["samples"]}
    if len(by_smiles) != len(source_plan["samples"]):
        raise ValueError("Duplicate fixed-panel identity")
    chosen = []
    for smiles in WITNESSES:
        if smiles not in by_smiles:
            raise ValueError("Missing fixed witness; never substitute")
        row = by_smiles[smiles]
        key = row["sample_key"]
        positive = {d["raw_assigned_candidate"] for d in label_cases[key]["target_details"]
                    if d["raw_assigned_candidate"] is not None}
        negative = [r for r in saved_cases[key]["candidates"]["candidates"] if r["candidate"] not in positive]
        if not positive or not negative:
            raise ValueError("Witness needs both positive and negative candidates")
        hardest = min(negative, key=lambda r: (-r["selection_logit"], r["candidate"]))
        chosen.append({**row, "positive_slots": sorted(positive), "hardest_negative": hardest["candidate"],
            "hardest_negative_ties": sorted(r["candidate"] for r in negative
                if r["selection_logit"] == hardest["selection_logit"]),
            "probe_slots": sorted(positive | {hardest["candidate"]})})
    return chosen


def split_gradient_sources(losses, logits, positive, mask, parameters):
    if (positive.shape != logits.shape or mask.shape != logits.shape or positive.dtype != torch.bool
            or mask.dtype != torch.bool or bool((positive & ~mask).any())):
        raise ValueError("Malformed selector source masks")
    weighted = {name.removesuffix("_loss"): weight * losses[name] for name, weight in WEIGHTS.items()}
    reconstructed_loss = sum(weighted.values())
    if not torch.allclose(reconstructed_loss.detach(), losses["loss"].detach(), rtol=1e-6, atol=1e-6):
        raise ValueError("Weighted losses do not reconstruct original full objective")
    gradients, logit_sources = {}, {}
    for name, value in weighted.items():
        if name.startswith("atom_"):
            derivative = torch.autograd.grad(value, logits, retain_graph=True)[0].detach()
            for suffix, source_mask in (("positive", positive & mask), ("negative", ~positive & mask)):
                contribution = derivative * source_mask
                # Its value is NOT a loss attribution; its first derivative is
                # exactly J_logits^T times the selected source derivative.
                bookkeeping = (logits * contribution).sum()
                key = name + "_" + suffix
                gradients[key] = previous.flat_gradient(bookkeeping, parameters)
                logit_sources[key] = contribution
        else:
            gradients[name] = previous.flat_gradient(value, parameters)
    full = previous.flat_gradient(losses["loss"], parameters)
    norm = float(full.norm())
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("No finite nonzero full-objective reference direction")
    error = float((sum(gradients.values()) - full).norm()) / norm
    if error > 1e-3:
        raise ValueError("Source parameter gradients do not reconstruct total")
    return weighted, gradients, logit_sources, full, error


def contributions(jacobian, gradients, full):
    norm = float(full.norm())
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("No finite nonzero full-objective reference direction")
    parts = {name: -float(jacobian @ gradient) / norm for name, gradient in gradients.items()}
    total = -float(jacobian @ full) / norm
    error = abs(sum(parts.values()) - total)
    # Cancellation can make relative-to-total errors misleading. Bound relative
    # to the sum of magnitudes, while recording the absolute residual too.
    if error > 1e-4 * max(sum(abs(v) for v in parts.values()), 1.):
        raise ValueError("Additive directional attribution failed")
    return dict(components=parts, total=total, sum_components=sum(parts.values()), absolute_residual=error)


def decompose(output, raw, bonds, labels, parameters, witness, old_case, thresholds):
    if any(p.grad is not None or not p.requires_grad for p in parameters):
        raise ValueError("Selector-only diagnostic leaves required")
    logits, positive = output["atom_presence_logits"], labels["atom_presence_targets"]
    if positive[0].nonzero().flatten().tolist() != witness["positive_slots"]:
        raise ValueError("Fixed witness labels changed")
    losses = proposal_object_set_loss(output, positive, labels["bond_presence_targets"], raw["atom_mask"],
        bonds["bond_mask"], labels["target_atom_refs"], labels["bond_supervision_mask"],
        atom_decision_threshold=thresholds["atom_threshold"], bond_decision_threshold=thresholds["bond_threshold"])
    weighted, gradients, sources, full, error = split_gradient_sources(losses, logits, positive, raw["atom_mask"], parameters)
    prototype = full - gradients["atom_count_positive"]
    old_rows = {r["candidate"]: r for r in old_case["candidates"]}
    jacobians, rows = {}, []
    for i in witness["probe_slots"]:
        jacobians[i] = previous.flat_gradient(logits[0, i], parameters)
        attribution = contributions(jacobians[i], gradients, full)
        proto_drift = -float(jacobians[i] @ prototype) / float(prototype.norm())
        if i in old_rows:
            for actual, arm in ((attribution["total"], "original_full_set"),
                                (proto_drift, "positive_count_detached_full_set")):
                expected = old_rows[i]["directions"][arm]["unit_negative_gradient_derivative"]
                if not math.isclose(actual, expected, rel_tol=1e-4, abs_tol=1e-4):
                    raise ValueError("Previous full/prototype parameter direction did not reproduce")
        rows.append(dict(candidate=i, positive=bool(positive[0, i]),
            logit=float(logits[0, i].detach()), selected=bool(logits[0, i].detach() >= thresholds["atom_threshold"]),
            raw_target=int(labels["atom_target_indices"][0, i]), retention=attribution,
            direct_logit_source_derivatives={key: float(value[0, i]) for key, value in sources.items()},
            previous_prototype_derivative_reproduced=proto_drift))
    hard = witness["hardest_negative"]
    margins = [dict(positive=i, negative=hard, current_margin=float((logits[0, i] - logits[0, hard]).detach()),
                   attribution=contributions(jacobians[i] - jacobians[hard], gradients, full))
               for i in witness["positive_slots"]]
    matrix = torch.stack([jacobians[i] for i in witness["probe_slots"]])
    gram = matrix @ matrix.T
    norms = gram.diag().clamp_min(0).sqrt()
    cosine = [[float(gram[i, j] / (norms[i] * norms[j])) if norms[i] * norms[j] > 0 else None
               for j in range(len(norms))] for i in range(len(norms))]
    if any(p.grad is not None for p in parameters):
        raise ValueError("Diagnostic accumulated parameter gradients")
    return dict(sample_key=witness["sample_key"], canonical_smiles=witness["canonical_smiles"],
        weighted_losses={key: float(value.detach()) for key, value in weighted.items()}, full_loss=float(losses["loss"].detach()),
        gradient_l2={key: float(value.norm()) for key, value in gradients.items()}, full_gradient_l2=float(full.norm()),
        gradient_sum_relative_error=error, rows=rows, margins=margins,
        jacobian_candidate_order=witness["probe_slots"], jacobian_gram=gram.tolist(), jacobian_cosine=cosine,
        normalization="Every contribution divided by ORIGINAL full-set gradient norm; never per-component normalization.",
        no_new_objectives=True)


def audit(output=OUTPUT):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only decomposition required")
    output = Path(output)
    if not output.is_dir() or (output / "plan.json").exists() or (output / "report.json").exists():
        raise ValueError("Fresh budgeted directory required")
    torch.set_num_threads(2)
    torch.manual_seed(20260919)
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started >= WALL_SECONDS:
            raise TimeoutError("Fixed decomposition CPU deadline exceeded")
    prior = replay.read_bound(previous.OUTPUT / "report.json", PRIOR_SHA)
    prior_plan = replay.read_bound(previous.OUTPUT / "plan.json", prior["plan_sha256"])
    original = replay.read_bound(replay.SOURCE / "report.json", replay.REPORT_SHA)
    original_plan = replay.read_bound(replay.SOURCE / "plan.json", replay.PLAN_SHA)
    labels = replay.read_bound(previous.PRIOR, previous.PRIOR_SHA)
    labels = {r["sample_key"]: r for r in labels["cases"]}
    saved, old_cases = {}, {}
    bindings = {**prior_plan["input_sha256"], str(previous.OUTPUT / "report.json"): PRIOR_SHA,
                str(previous.OUTPUT / "plan.json"): prior["plan_sha256"]}
    if not prior["complete"] or prior["checked_images"] != 32:
        raise ValueError("Incomplete prior diagnostic")
    for row in prior_plan["samples"]:
        if row["canonical_smiles"] in WITNESSES:
            key = row["sample_key"]
            saved[key] = replay.read_bound(replay.SOURCE / (key + ".json"), original["case_sha256"][key])
            path = previous.OUTPUT / (key + ".json")
            old_cases[key] = replay.read_bound(path, prior["case_sha256"][key])
            bindings[str(path)] = prior["case_sha256"][key]
    witnesses = select_witnesses(prior_plan, saved, labels)
    for path in (Path(__file__), replay.REPO / "tests/data_sdf/test_curriculum_v2_selector_loss_terms.py"):
        bindings[str(path.absolute())] = replay.digest(path)
    replay.verify_bindings(bindings, deadline)
    plan = dict(schema_version="chemvector-selector-loss-terms-plan-0.1", input_sha256=bindings,
        witnesses=witnesses, selection="Four named prior witnesses; hardest raw negative chosen from saved logits, lowest slot breaks ties",
        cpu_wall_seconds=WALL_SECONDS, automatic_retries=0, planned_structure_forwards=4, planned_selector_forwards=8,
        gpu_used=False, optimizer_created=False, optimizer_steps=0, training_authorized=False, private_identity_queries=0,
        thresholds=prior_plan["thresholds"], weighted_loss_terms=WEIGHTS, mode="eval", parameter_scope="set_head only",
        purpose="Decompose same objective; no new prototype or optimizer direction search")
    previous.write(output / "plan.json", plan)
    cases = []
    report = dict(complete=False, plan_sha256=replay.digest(output / "plan.json"),
        gpu_used=False, optimizer_created=False, optimizer_steps=0, training_authorized=False, training_performed=False,
        private_identity_queries=0, ocr_loaded=False, actual_structure_forwards=0, actual_selector_forwards=0, actual_pair_forwards=0)
    try:
        model, config, payload = bundle.load_bundle(previous.INITIAL, expected_sha256=original_plan["initial_sha256"])
        model.eval().requires_grad_(False)
        model.set_head.requires_grad_(True)
        before = state_digest(model)
        if (before != original_plan["expected_initial_state"] or asdict(config) != original["config"]
                or payload["progress"]["optimizer_steps"] != 0
                or any(p.requires_grad and not name.startswith("set_head.") for name, p in model.named_parameters())):
            raise ValueError("Initial/config/gradient scope changed")
        parameters = tuple(model.set_head.parameters())
        collator = data.CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
            bond_queries=config.maximum_bonds), purpose="diagnostic")
        for witness in witnesses:
            deadline()
            key = witness["sample_key"]
            print(json.dumps({"event": "case_started", "key": key}), flush=True)
            sample = data.load_sample(replay.DATA, key, purpose="diagnostic")
            with torch.no_grad():
                front = continuation.structure_forward(model, pixels.image_tensor(sample["image"]).unsqueeze(0),
                    config, plan["thresholds"]["atom_threshold"])
            report["actual_structure_forwards"] += 1
            report["actual_selector_forwards"] += 1
            raw, bonds = front[3], front[4]
            if upstream_diagnostic.candidate_trace(raw, front[6], front[5]) != saved[key]["candidates"]:
                raise ValueError("Saved image-only predictions changed")
            before_cache = tensor_tree_digest({"raw": raw, "bonds": bonds})
            result = previous.selector_call(model.set_head, raw, bonds)
            report["actual_selector_forwards"] += 1
            if any(not torch.equal(result[k].detach(), front[5][k]) for k in ("atom_presence_logits", "bond_presence_logits")):
                raise ValueError("Selector gradient replay changed predictions")
            targets = collator([sample])
            associated = associate_predicted_proposals(raw, bonds, targets, config)
            case = decompose(result, raw, bonds, associated, parameters, witness, old_cases[key], plan["thresholds"])
            if tensor_tree_digest({"raw": raw, "bonds": bonds}) != before_cache:
                raise ValueError("Gradient decomposition changed candidates")
            previous.write(output / (key + ".json"), case)
            cases.append(case)
            print(json.dumps({"event": "case_completed", "images": len(cases)}), flush=True)
            del front, raw, bonds, result, targets, associated, sample
        replay.verify_bindings(bindings, deadline)
        if state_digest(model) != before or any(p.grad is not None for p in model.parameters()) or torch.cuda.is_initialized():
            raise ValueError("Model state/gradient/device invariant failed")
        report.update(complete=True, states_before=before, states_after=state_digest(model),
            parameter_count=sum(p.numel() for p in parameters),
            max_gradient_sum_relative_error=max(c["gradient_sum_relative_error"] for c in cases),
            positive_candidates=sum(sum(r["positive"] for r in c["rows"]) for c in cases),
            limitations=["Four named diagnostic witnesses; no population or whole-molecule accuracy claim.",
                "Eval, frozen front, set-head loss only; no full multi-task or mixed-batch attribution.",
                "First-order Euclidean gradient reference; not AdamW, finite updates or long-term training.",
                "Positive/negative splits are chain-rule derivative bookkeeping, not new scalar objectives."])
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report.update(checked_images=len(cases), elapsed_seconds=time.monotonic() - started,
            case_sha256={c["sample_key"]: replay.digest(output / (c["sample_key"] + ".json")) for c in cases})
        previous.write(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        report = audit()
        print(json.dumps({"complete": report["complete"], "images": report["checked_images"]}), flush=True)
        return 0
    receipt = runtime.run_budgeted([sys.executable, "-m", "tools.diagnose_curriculum_v02_selector_loss_terms", "--worker"],
        output=OUTPUT, budget=runtime.ResourceBudget(WALL_SECONDS))
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
