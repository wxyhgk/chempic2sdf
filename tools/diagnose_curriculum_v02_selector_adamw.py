"""One fixed CPU60 six-image cold-AdamW check; no real optimizer or updates.

Four existing witnesses supply independent first-step directions. Two old,
pre-fixed training images measure those SAME directions (J_old @ delta_new).
This is local linearization, not a prediction, training or regression result.
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
from chemvector.training.curriculum import bundle, continuation, data, gradient_diagnostic, pixels, runtime, upstream_diagnostic
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.interface_diagnostic import tensor_tree_digest
from tools import diagnose_curriculum_v02_selector_loss_terms as terms

previous, replay = terms.previous, terms.replay
OUTPUT = replay.REPO / "outputs/audits/curriculum-v02-selector-cold-adamw6-20260919"
TERMS_SHA = "a47b66a9946ec7bca3cce733127ecaa136e976d905f59485268b42b962bc7fc9"
OLD_REPORT = replay.REPO / "outputs/audits/chemvector-loss-gradients-20260915-v2/report.json"
OLD_SHA = "00440c76be6a87ce5f72e1543586c7dc7ff96bd8ac01c67dee9b01e78df07866"
RELEASE = replay.REPO / "outputs/datasets/chemvector-expanded-release-20260915"
OLD_KEYS = ("full_molecule_expansion:curr_d1248897c226b214__serif",
            "full_molecule_expansion:curr_b7a85b1f6804b7df__font_small")
WALL_SECONDS, LEARNING_RATE = 60, 5e-6


def reference_direction(gradient):
    if gradient.ndim != 1 or not bool(torch.isfinite(gradient).all()):
        raise ValueError("Expected finite flat selector gradient")
    norm = float(gradient.norm())
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Expected nonzero selector gradient")
    clip = min(1., 1. / (norm + 1e-6))
    delta = gradient_diagnostic.cold_adamw_direction(gradient, LEARNING_RATE, clip)
    return delta, dict(selector_gradient_l2=norm, selector_clip_coefficient=clip,
                      cold_adamw_delta_l2=float(delta.norm()), learning_rate=LEARNING_RATE)


def probe_selection(logits, positive, mask, threshold):
    """Old controls: ALL positives/retained negatives, hardest and nearest negative.

    Rule fixed before model load, ties always lowest candidate slot. No
    selection using derivatives, no replacement when a category is absent.
    """
    if (logits.ndim != 1 or positive.shape != logits.shape or mask.shape != logits.shape
            or positive.dtype != torch.bool or mask.dtype != torch.bool
            or bool((positive & ~mask).any()) or not bool(torch.isfinite(logits[mask]).all())):
        raise ValueError("Malformed candidate probe inputs")
    positives = positive.nonzero().flatten().tolist()
    negatives = (mask & ~positive).nonzero().flatten().tolist()
    retained = [i for i in negatives if float(logits[i]) >= threshold]
    hardest = min(negatives, key=lambda i: (-float(logits[i]), i)) if negatives else None
    nearest = min(negatives, key=lambda i: (abs(float(logits[i]) - threshold), i)) if negatives else None
    return dict(positive_slots=positives, retained_negative_slots=retained,
                hardest_negative=hardest, nearest_threshold_negative=nearest,
                probe_slots=sorted(set(positives + retained + [i for i in (hardest, nearest) if i is not None])))


def score_effect(logit, change, positive, threshold):
    if not all(math.isfinite(x) for x in (logit, change, threshold)):
        raise ValueError("Nonfinite directional score")
    before, after = logit - threshold, logit + change - threshold
    sign = 1 if positive else -1
    return dict(logit_change=change, linearized_logit=logit + change,
                threshold_offset_before=before, threshold_offset_linearized=after,
                correct_side_headroom_before=sign * before,
                correct_side_headroom_linearized=sign * after,
                correct_side_headroom_change=sign * change,
                selected_before=logit >= threshold,
                selected_linearized=logit + change >= threshold,
                linearized_correct_to_wrong=(logit >= threshold) == positive
                    and (logit + change >= threshold) != positive,
                linearized_wrong_to_correct=(logit >= threshold) != positive
                    and (logit + change >= threshold) == positive)


def measure_probes(logits, labels, parameters, selection, directions, threshold, *, full_gradient=None, prior=None):
    rows = []
    prior_rows = {r["candidate"]: r for r in prior["rows"]} if prior else {}
    for i in selection["probe_slots"]:
        jacobian = previous.flat_gradient(logits[0, i], parameters)
        logit, positive = float(logits[0, i].detach()), bool(labels["atom_presence_targets"][0, i])
        row = dict(candidate=i, positive=positive, logit=logit,
                   raw_target=int(labels["atom_target_indices"][0, i]),
                   effects={key: score_effect(logit, float(jacobian @ delta), positive, threshold)
                            for key, delta in directions.items()})
        if full_gradient is not None:
            value = -float(jacobian @ full_gradient) / float(full_gradient.norm())
            row["unit_sgd_derivative"] = value
            old = prior_rows[i]
            if (positive != old["positive"] or logit != old["logit"]
                    or not math.isclose(value, old["retention"]["total"], rel_tol=1e-4, abs_tol=1e-4)):
                raise ValueError("Previous witness labels/logits/unit-SGD direction did not reproduce")
        rows.append(row)
    return rows


def prepare_plan(deadline):
    prior = replay.read_bound(terms.OUTPUT / "report.json", TERMS_SHA)
    prior_plan = replay.read_bound(terms.OUTPUT / "plan.json", prior["plan_sha256"])
    original = replay.read_bound(replay.SOURCE / "report.json", replay.REPORT_SHA)
    original_plan = replay.read_bound(replay.SOURCE / "plan.json", replay.PLAN_SHA)
    old = replay.read_bound(OLD_REPORT, OLD_SHA)
    if (not prior["complete"] or prior["checked_images"] != 4
            or tuple(old["plan"]["training_keys"][:2]) != OLD_KEYS):
        raise ValueError("Fixed source panels changed")
    bindings = dict(prior_plan["input_sha256"])
    def bind(path, expected):
        key = str(Path(path).absolute())
        if key in bindings and bindings[key] != expected:
            raise ValueError("Conflicting evidence binding")
        bindings[key] = expected
    bind(terms.OUTPUT / "report.json", TERMS_SHA)
    bind(terms.OUTPUT / "plan.json", prior["plan_sha256"])
    bind(OLD_REPORT, OLD_SHA)
    manifest_path, admission_path = RELEASE / "manifest.json", RELEASE / "admission.json"
    for path in (manifest_path, admission_path):
        bind(path, old["plan"]["input_sha256"][str(path)])
    manifest = replay.read_bound(manifest_path, bindings[str(manifest_path)])
    admission = replay.read_bound(admission_path, bindings[str(admission_path)])
    if (admission["approved_manifest_sha256"] != bindings[str(manifest_path)]
            or not admission["training_authorized"]):
        raise ValueError("Old training control release invalid")
    old_samples = []
    for key in OLD_KEYS:
        matches = [r for r in manifest["samples"] if r["source"] + ":" + r["sample_key"] == key]
        if len(matches) != 1 or matches[0]["split"] != "train" or matches[0]["domain"] != "full_molecule":
            raise ValueError("Old control must remain unique full-molecule TRAIN sample")
        row = matches[0]
        root = Path(manifest["sources"][row["source"]])
        meta_path = root / "samples" / (row["sample_key"] + ".meta.json")
        if old["plan"]["input_sha256"][str(meta_path)] != row["meta_sha256"]:
            raise ValueError("Historical old control metadata changed")
        bind(meta_path, row["meta_sha256"])
        meta = replay.read_bound(meta_path, row["meta_sha256"])
        for suffix in data.PAYLOADS:
            bind(root / "samples" / (row["sample_key"] + suffix), meta["sha256"][suffix])
        old_samples.append({**row, "root": str(root), "fixed_key": key, "purpose": "diagnostic"})
    saved, prior_cases = {}, {}
    for witness in prior_plan["witnesses"]:
        key = witness["sample_key"]
        path = terms.OUTPUT / (key + ".json")
        bind(path, prior["case_sha256"][key])
        prior_cases[key] = replay.read_bound(path, prior["case_sha256"][key])
        saved[key] = replay.read_bound(replay.SOURCE / (key + ".json"), original["case_sha256"][key])
    for path in (Path(__file__), Path(gradient_diagnostic.__file__), Path(runtime.__file__),
                 replay.REPO / "tests/data_sdf/test_curriculum_v2_selector_adamw.py"):
        bind(path, replay.digest(path))
    replay.verify_bindings(bindings, deadline)
    plan = dict(schema_version="chemvector-selector-cold-adamw-plan-0.1", input_sha256=bindings,
        witnesses=prior_plan["witnesses"], old_controls=old_samples, thresholds=prior_plan["thresholds"],
        old_selection="First two training keys of fixed September15 four-image gradient audit; no substitutions",
        old_probe_rule="All raw positives and retained negatives; highest-logit and nearest-threshold negative; lowest-slot ties",
        direction_rule="Independent cold AdamW first step for EACH new witness; not accumulated or averaged",
        learning_rate=LEARNING_RATE, eps=1e-8, betas=[.9, .999], weight_decay=0,
        clip_norm=1., clip_scope="selector parameters only, not historical joint model",
        numeric_reference="FP32 autograd gradients flattened to float64; float64 formula and J dot delta, no FP32 parameter writes",
        mode="eval", batch_size=1, parameter_scope="set_head only, upstream frozen",
        objective="Unmodified full proposal_object_set_loss, original default coefficients",
        cpu_wall_seconds=WALL_SECONDS, automatic_retries=0, planned_structure_forwards=6,
        planned_selector_forwards=12, optimizer_created=False, optimizer_steps=0, gpu_used=False,
        training_authorized=False, private_identity_queries=0,
        scope="Local sign/headroom check only. Stop pure gradient phase after this check; no automatic trial or prototype search.")
    return plan, original, original_plan, saved, prior_cases


def summarize(cases):
    new, old = cases[:4], cases[4:]
    effects = [r["effects"][c["sample_key"]] for c in new for r in c["rows"] if r["positive"]]
    old_summary = {}
    for direction in (c["sample_key"] for c in new):
        rows = [(r, r["effects"][direction]) for c in old for r in c["rows"]]
        old_summary[direction] = dict(probes=len(rows), positive_probes=sum(r["positive"] for r, _ in rows),
            positive_scores_decrease=sum(r["positive"] and e["logit_change"] < 0 for r, e in rows),
            linearized_correct_to_wrong=sum(e["linearized_correct_to_wrong"] for _, e in rows),
            linearized_wrong_to_correct=sum(e["linearized_wrong_to_correct"] for _, e in rows),
            minimum_retained_positive_headroom=min((e["correct_side_headroom_linearized"] for r, e in rows
                if r["positive"] and e["selected_before"]), default=None))
    first = new[0]
    correct = [r for r in first["rows"] if r["positive"]]
    wrong = next(r for r in first["rows"] if r["candidate"] == first["selection"]["hardest_negative"])
    return dict(witness_positive_probes=len(effects), witness_positive_scores_increase=sum(e["logit_change"] > 0 for e in effects),
        witness_positive_scores_decrease=sum(e["logit_change"] < 0 for e in effects),
        c_triple_c_both_endpoints_up=all(r["effects"][first["sample_key"]]["logit_change"] > 0 for r in correct),
        c_triple_c_retained_negative_down=wrong["effects"][first["sample_key"]]["logit_change"] < 0,
        witness_linearized_decision_changes=sum(e["selected_before"] != e["selected_linearized"]
            for c in new for r in c["rows"] for e in r["effects"].values()),
        old_cross_direction=old_summary,
        accuracy_evaluated=False, finite_updates_evaluated=False, full_regression_evaluated=False)


def audit(output=OUTPUT):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only diagnostic required")
    output = Path(output)
    if not output.is_dir() or any((output / name).exists() for name in ("plan.json", "report.json")):
        raise ValueError("Fresh budgeted directory required")
    torch.set_num_threads(2)
    torch.manual_seed(20260919)
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started >= WALL_SECONDS:
            raise TimeoutError("Fixed cold-AdamW CPU deadline exceeded")
    plan, original, original_plan, saved, prior_cases = prepare_plan(deadline)
    previous.write(output / "plan.json", plan)
    cases, directions = [], {}
    report = dict(complete=False, plan_sha256=replay.digest(output / "plan.json"), gpu_used=False,
        optimizer_created=False, optimizer_steps=0, training_authorized=False, training_performed=False,
        private_identity_queries=0, ocr_loaded=False, actual_structure_forwards=0,
        actual_selector_forwards=0, actual_pair_forwards=0)
    try:
        model, config, payload = bundle.load_bundle(previous.INITIAL, expected_sha256=original_plan["initial_sha256"])
        model.eval().requires_grad_(False)
        model.set_head.requires_grad_(True)
        before = state_digest(model)
        parameters = tuple(model.set_head.parameters())
        if (before != original_plan["expected_initial_state"] or asdict(config) != original["config"]
                or payload["progress"]["optimizer_steps"] != 0 or sum(p.numel() for p in parameters) != 178565
                or any(p.requires_grad and not n.startswith("set_head.") for n, p in model.named_parameters())):
            raise ValueError("Initial/config/selector scope changed")
        collator = data.CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
            bond_queries=config.maximum_bonds), purpose="diagnostic")
        threshold = plan["thresholds"]["atom_threshold"]
        for number, witness in enumerate(plan["witnesses"] + plan["old_controls"]):
            deadline()
            key = witness["sample_key"]
            print(json.dumps(dict(event="case_started", key=key)), flush=True)
            sample = data.load_sample(witness.get("root", replay.DATA), key, purpose="diagnostic")
            with torch.no_grad():
                front = continuation.structure_forward(model, pixels.image_tensor(sample["image"]).unsqueeze(0), config, threshold)
            report["actual_structure_forwards"] += 1
            report["actual_selector_forwards"] += 1
            raw, bonds = front[3:5]
            if number < 4 and upstream_diagnostic.candidate_trace(raw, front[6], front[5]) != saved[key]["candidates"]:
                raise ValueError("Saved witness predictions changed")
            cache_before = tensor_tree_digest({"raw": raw, "bonds": bonds})
            result = previous.selector_call(model.set_head, raw, bonds)
            report["actual_selector_forwards"] += 1
            if any(not torch.equal(result[k].detach(), front[5][k]) for k in ("atom_presence_logits", "bond_presence_logits")):
                raise ValueError("Differentiable replay changed baseline scores")
            batch = collator([sample])
            if batch["purpose"] != "diagnostic" or batch["training_authorized"]:
                raise ValueError("Diagnostic targets only")
            labels = associate_predicted_proposals(raw, bonds, batch, config)
            logits = result["atom_presence_logits"]
            full, prior, direction_info = None, None, None
            if number < 4:
                selection, prior = witness, prior_cases[key]
                if labels["atom_presence_targets"][0].nonzero().flatten().tolist() != witness["positive_slots"]:
                    raise ValueError("Fixed witness assignment changed")
                losses = proposal_object_set_loss(result, labels["atom_presence_targets"], labels["bond_presence_targets"],
                    raw["atom_mask"], bonds["bond_mask"], labels["target_atom_refs"], labels["bond_supervision_mask"],
                    atom_decision_threshold=threshold, bond_decision_threshold=plan["thresholds"]["bond_threshold"])
                full = previous.flat_gradient(losses["loss"], parameters)
                delta, direction_info = reference_direction(full)
                direction_info["full_loss"] = float(losses["loss"].detach())
                directions[key] = delta
                active = {key: delta}
            else:
                selection = probe_selection(logits[0].detach(), labels["atom_presence_targets"][0], raw["atom_mask"][0], threshold)
                active = directions
            rows = measure_probes(logits, labels, parameters, selection, active, threshold, full_gradient=full, prior=prior)
            case = dict(sample_key=key, canonical_smiles=witness["canonical_smiles"],
                panel="current_witness" if number < 4 else "old_training_control", selection=selection,
                direction=direction_info, rows=rows, raw_positive_count=int(labels["atom_presence_targets"].sum()),
                raw_candidate_count=int(raw["atom_mask"].sum()), candidate_cache_sha256=cache_before)
            if number < 4:
                hard = next(r for r in rows if r["candidate"] == selection["hardest_negative"])
                case["margins"] = [dict(positive=r["candidate"], negative=hard["candidate"],
                    margin_before=r["logit"] - hard["logit"],
                    cold_adamw_margin_change=r["effects"][key]["logit_change"] - hard["effects"][key]["logit_change"])
                    for r in rows if r["positive"]]
                del losses
            if cache_before != tensor_tree_digest({"raw": raw, "bonds": bonds}) or any(p.grad is not None for p in model.parameters()):
                raise ValueError("Cache/gradient invariant failed")
            previous.write(output / (key + ".json"), case)
            cases.append(case)
            print(json.dumps(dict(event="case_completed", images=len(cases), probes=len(rows))), flush=True)
            del front, raw, bonds, result, logits, batch, labels, sample, full
        replay.verify_bindings(plan["input_sha256"], deadline)
        if state_digest(model) != before or any(p.grad is not None for p in model.parameters()) or torch.cuda.is_initialized():
            raise ValueError("State/gradient/device invariant failed")
        report.update(complete=True, summary=summarize(cases), states_before=before, states_after=state_digest(model),
            parameter_count=sum(p.numel() for p in parameters), original_unit_sgd_and_predictions_reproduced=True,
            limitations=["Independent cold first steps, not warm AdamW or cumulative training.",
                "J dot delta at LR 5e-6 is a linearized score change, not an actual changed prediction.",
                "Four named tiny witnesses and two fixed old training controls, not generalization or whole-graph accuracy.",
                "Selector-only norm clipping differs from historical joint-model norm clipping.",
                "Old probes evaluate the SAME new-image directions, not gradients of old-image losses.",
                "Positive raw assignment is selector supervision, not chemical identity correctness."])
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
        result = audit()
        print(json.dumps(dict(complete=result["complete"], images=result["checked_images"])), flush=True)
        return 0
    receipt = runtime.run_budgeted([sys.executable, "-m", "tools.diagnose_curriculum_v02_selector_adamw", "--worker"],
        output=OUTPUT, budget=runtime.ResourceBudget(WALL_SECONDS))
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
