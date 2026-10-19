"""Fixed CPU selector-only gradient contrast; never update model parameters.

Same 32 images/initial/thresholds as the saved audit. Freeze front-end, OCR,
pair/role/bbox. Compare the real set objective to ONE stop-gradient diagnostic:
detach positive probabilities only in atom count. Forward loss values remain
equal. All directional effects are first-order, per-image unit-SGD references,
not optimizer steps, historical AdamW updates, or improved predictions.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import inspect
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.predicted_incidence import associate_predicted_proposals
from chemvector.proposal_object_set import proposal_object_set_loss
from chemvector.training.curriculum import bundle, continuation, data, pixels, runtime, upstream_diagnostic
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.interface_diagnostic import tensor_tree_digest
from tools import audit_curriculum_v02_selector_matching as replay

PRIOR = replay.OUTPUT / "report.json"
PRIOR_SHA = "b112bde629072e5e361cef88f770647fc728af4fb38340e5c86c5714e63d9ea1"
OUTPUT = replay.REPO / "outputs/audits/curriculum-v02-selector-parameter-gradients-20260919"
INITIAL = replay.REPO / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
ARMS = ("original_full_set", "positive_count_detached_full_set", "original_atom_partial")
WALL_SECONDS = 300


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def count_contrast(logits, positive, mask, original_count):
    """Diagnostic-only gradient intervention. Never imported by production."""
    if logits.shape != positive.shape or mask.shape != positive.shape or positive.dtype != torch.bool:
        raise ValueError("Malformed count contrast tensors")
    probabilities = logits.sigmoid()
    count = F.smooth_l1_loss((probabilities * mask).sum(-1), positive.sum(-1).float())
    if not torch.equal(count.detach(), original_count.detach()):
        raise ValueError("Original count does not match existing objective")
    isolated = torch.where(positive, probabilities.detach(), probabilities)
    changed_count = F.smooth_l1_loss((isolated * mask).sum(-1), positive.sum(-1).float())
    if not torch.equal(count.detach(), changed_count.detach()):
        raise ValueError("Count intervention changed forward values")
    return .1 * (changed_count - original_count)


def flat_gradient(value, parameters):
    grads = torch.autograd.grad(value, parameters, allow_unused=True, retain_graph=True)
    flat = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten().double()
                      for p, g in zip(parameters, grads)])
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("Nonfinite selector parameter gradient")
    return flat


def direction_record(probe_gradient, gradients):
    result = {}
    for name, grad in gradients.items():
        norm = float(grad.norm())
        dot = float(probe_gradient @ grad)
        drift = -dot / norm if norm else None
        result[name] = dict(dot_with_loss_gradient=dot, unit_negative_gradient_derivative=drift)
    return result


def selector_call(head, raw, bonds):
    return head(raw["atom_features"], raw["atom_geometry"], raw["atom_scores"], raw["atom_mask"],
        bonds["bond_features"], bonds["bond_endpoints"], bonds["bond_types"], bonds["bond_scores"], bonds["bond_mask"])


def parameter_probe(output, raw, bonds, labels, parameters, prior_case, *, atom_threshold, bond_threshold):
    if any(not p.requires_grad or p.grad is not None for p in parameters):
        raise ValueError("Expected selector-only gradient leaves, no accumulated gradients")
    positive, mask = labels["atom_presence_targets"], raw["atom_mask"]
    logits = output["atom_presence_logits"]
    losses = proposal_object_set_loss(output, positive, labels["bond_presence_targets"], mask, bonds["bond_mask"],
        labels["target_atom_refs"], labels["bond_supervision_mask"], atom_decision_threshold=atom_threshold,
        bond_decision_threshold=bond_threshold)
    delta = count_contrast(logits, positive, mask, losses["atom_count_loss"])
    objectives = dict(original_full_set=losses["loss"], positive_count_detached_full_set=losses["loss"] + delta,
        original_atom_partial=losses["atom_presence_loss"] + .5 * losses["atom_ranking_loss"] + .1 * losses["atom_count_loss"])
    gradients = {name: flat_gradient(value, parameters) for name, value in objectives.items()}
    contrast_gradient = flat_gradient(delta, parameters)
    expected = gradients[ARMS[0]] + contrast_gradient
    error = float((expected - gradients[ARMS[1]]).norm()) / max(float(gradients[ARMS[1]].norm()), 1e-12)
    if error > 1e-3:
        raise ValueError("Shared parameter gradient contrast did not reconstruct")
    cosine = float(gradients[ARMS[0]] @ gradients[ARMS[1]]) / max(
        float(gradients[ARMS[0]].norm()) * float(gradients[ARMS[1]].norm()), 1e-30)
    partial_logits = torch.autograd.grad(objectives[ARMS[2]], logits, retain_graph=True)[0]
    relevant = prior_case["relevant_candidates"]
    near = {i for row in prior_case["target_details"] for i in row["near_raw_ids"]}
    candidates = []
    for row in relevant:
        i = row["candidate"]
        if bool(positive[0, i]) != row["raw_positive"]:
            raise ValueError("Real raw supervision changed from saved atom replay")
        if abs(float(partial_logits[0, i]) - row["atom_only_logit_derivatives"]["atom_only_total"]) > 2e-6:
            raise ValueError("Atom partial cached-logit gradient did not reproduce")
        category = ("positive_retained" if row["selected"] else "positive_rejected") if row["raw_positive"] else (
                    "duplicate_negative" if i in near else "background_negative")
        probe = flat_gradient(logits[0, i], parameters)
        candidates.append(dict(candidate=i, category=category, selected=row["selected"],
            selector_logit=float(logits[0, i].detach()), raw_target=row["raw_target"],
            previous_atom_partial_logit_derivative=row["atom_only_logit_derivatives"]["atom_only_total"],
            jacobian_l2=float(probe.norm()), directions=direction_record(probe, gradients)))
    # Every unselected distant negative matters too, not only displayed witnesses.
    near_mask = torch.zeros_like(mask)
    if near:
        near_mask[0, sorted(near)] = True
    groups = dict(positive_retained=positive & (logits.detach() >= atom_threshold),
        positive_rejected=positive & (logits.detach() < atom_threshold),
        duplicate_negative=mask & ~positive & near_mask, background_negative=mask & ~positive & ~near_mask)
    group_results = {}
    for name, group in groups.items():
        count = int(group.sum())
        group_results[name] = dict(count=count, mean_logit=None, directions=None)
        if count:
            mean = logits[group].mean()
            group_results[name].update(mean_logit=float(mean.detach()),
                directions=direction_record(flat_gradient(mean, parameters), gradients))
    if any(p.grad is not None for p in parameters):
        raise ValueError("Diagnostic populated parameter .grad")
    return dict(losses={k: float(v.detach()) for k, v in losses.items()},
        objectives={k: float(v.detach()) for k, v in objectives.items()},
        gradient_l2={k: float(v.norm()) for k, v in gradients.items()},
        contrast_gradient_l2=float(contrast_gradient.norm()), gradient_sum_relative_error=error,
        full_set_vs_prototype_gradient_cosine=max(-1., min(1., cosine)),
        candidates=candidates, groups=group_results,
        supervised_bond_proposals=int(labels["bond_supervision_mask"].sum()))


def summarize(cases, atom_threshold):
    result = {}
    for arm in ARMS:
        counts, groups = Counter(), Counter()
        for case in cases:
            for row in case["candidates"]:
                drift = row["directions"][arm]["unit_negative_gradient_derivative"]
                sign = "up" if drift is not None and drift > 1e-8 else "down" if drift is not None and drift < -1e-8 else "zero"
                counts[row["category"] + ":" + sign] += 1
                if row["category"].startswith("positive"):
                    counts["all_positive:" + sign] += 1
                if row["category"] == "positive_retained" and atom_threshold <= row["selector_logit"] < 0:
                    counts["positive_retained_between_threshold_and_zero:" + sign] += 1
            for name, row in case["groups"].items():
                if row["count"]:
                    drift = row["directions"][arm]["unit_negative_gradient_derivative"]
                    groups[name + (":mean_up" if drift is not None and drift > 1e-8 else
                                   ":mean_down" if drift is not None and drift < -1e-8 else ":mean_zero")] += 1
        result[arm] = dict(relevant_candidate_counts=dict(counts), per_image_group_mean_counts=dict(groups))
    return dict(images=len(cases), arms=result, relevant_candidates=sum(len(c["candidates"]) for c in cases),
                sign_tolerance=1e-8, positive_denominator=112,
                caution="Per-image first-order unit-SGD direction in selector parameters; no actual prediction changes or accuracy.")


def audit(output=OUTPUT):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only gradient diagnostic required")
    output = Path(output)
    if not output.is_dir() or (output / "plan.json").exists() or (output / "report.json").exists():
        raise ValueError("Use fresh budgeted worker directory; never retry/overwrite")
    torch.set_num_threads(2)
    torch.manual_seed(20260919)
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started >= WALL_SECONDS:
            raise TimeoutError("CPU gradient diagnostic deadline exceeded")
    prior = replay.read_bound(PRIOR, PRIOR_SHA)
    original_plan = replay.read_bound(replay.SOURCE / "plan.json", replay.PLAN_SHA)
    original_report = replay.read_bound(replay.SOURCE / "report.json", replay.REPORT_SHA)
    prior_by_key = {r["sample_key"]: r for r in prior["cases"]}
    if len(prior_by_key) != 32 or set(prior_by_key) != {r["sample_key"] for r in original_plan["samples"]}:
        raise ValueError("Fixed panel incomplete")
    bindings = {**prior["input_sha256"], str(PRIOR): PRIOR_SHA}
    for path in (Path(__file__), replay.REPO / "tests/data_sdf/test_curriculum_v2_selector_gradients.py",
                 Path(inspect.getfile(proposal_object_set_loss)), Path(runtime.__file__)):
        name, value = str(path.absolute()), replay.digest(path)
        if name in bindings and value != bindings[name]:
            raise ValueError("Changed original implementation")
        bindings[name] = value
    replay.verify_bindings(bindings, deadline)
    plan = dict(schema_version="chemvector-selector-gradient-contrast-plan-0.1", samples=original_plan["samples"],
        input_sha256=bindings, cpu_wall_seconds=WALL_SECONDS, automatic_retries=0, mode="eval",
        planned_structure_forwards=32, planned_selector_forwards=64, planned_pair_forwards=0,
        optimizer_created=False, optimizer_steps=0, gpu_used=False, training_authorized=False,
        thresholds=original_plan["thresholds"], prototype="detach positive sigmoid terms in atom_count ONLY; equal forward loss",
        parameter_scope="all set_head parameters only; upstream features detached; OCR/pair/role/bbox frozen",
        direction="negative Euclidean loss gradient normalized to unit L2; per-image, no application",
        accuracy_measured=False, private_identity_queries=0)
    write(output / "plan.json", plan)
    cases = []
    report = dict(complete=False, plan_sha256=replay.digest(output / "plan.json"),
        optimizer_created=False, optimizer_steps=0, gpu_used=False, training_performed=False,
        training_authorized=False, private_identity_queries=0, ocr_loaded=False,
        actual_structure_forwards=0, actual_selector_forwards=0, actual_pair_forwards=0)
    try:
        model, config, payload = bundle.load_bundle(INITIAL, expected_sha256=original_plan["initial_sha256"])
        model.eval().requires_grad_(False)
        model.set_head.requires_grad_(True)
        before = state_digest(model)
        if (before != original_plan["expected_initial_state"] or asdict(config) != original_report["config"]
                or payload["progress"]["optimizer_steps"] != 0):
            raise ValueError("Initial model state/config/progress changed")
        params = tuple(model.set_head.parameters())
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        if not names or not all(n.startswith("set_head.") for n in names):
            raise ValueError("Trainable diagnostic scope expanded")
        collator = data.CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
            bond_queries=config.maximum_bonds), purpose="diagnostic")
        for row in plan["samples"]:
            deadline()
            key = row["sample_key"]
            print(json.dumps({"event": "case_started", "key": key, "index": len(cases) + 1}), flush=True)
            sample = data.load_sample(replay.DATA, key, purpose="diagnostic")
            with torch.no_grad():
                front = continuation.structure_forward(model, pixels.image_tensor(sample["image"]).unsqueeze(0),
                    config, plan["thresholds"]["atom_threshold"])
            report["actual_structure_forwards"] += 1
            report["actual_selector_forwards"] += 1
            raw, bonds = front[3], front[4]
            saved = replay.read_bound(replay.SOURCE / (key + ".json"), original_report["case_sha256"][key])
            if upstream_diagnostic.candidate_trace(raw, front[6], front[5]) != saved["candidates"]:
                raise ValueError("Initial image-only candidate prediction did not reproduce exactly")
            cache_hash = tensor_tree_digest({"raw": raw, "bonds": bonds})
            result = selector_call(model.set_head, raw, bonds)
            report["actual_selector_forwards"] += 1
            if not torch.equal(result["atom_presence_logits"].detach(), front[5]["atom_presence_logits"]):
                raise ValueError("Gradient selector replay changed atom logits")
            if not torch.equal(result["bond_presence_logits"].detach(), front[5]["bond_presence_logits"]):
                raise ValueError("Gradient selector replay changed bond logits")
            # Targets enter only after raw image-only predictions are frozen.
            batch = collator([sample])
            labels = associate_predicted_proposals(raw, bonds, batch, config)
            measured = parameter_probe(result, raw, bonds, labels, params, prior_by_key[key],
                atom_threshold=plan["thresholds"]["atom_threshold"], bond_threshold=plan["thresholds"]["bond_threshold"])
            if tensor_tree_digest({"raw": raw, "bonds": bonds}) != cache_hash:
                raise ValueError("Parameter probe mutated frozen inputs")
            cases.append(dict(sample_key=key, canonical_smiles=row["canonical_smiles"], **measured))
            write(output / (key + ".json"), cases[-1])
            print(json.dumps({"event": "case_completed", "images": len(cases),
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
            del front, raw, bonds, result, batch, labels, sample, measured
        if state_digest(model) != before or any(p.grad is not None for p in model.parameters()) or torch.cuda.is_initialized():
            raise ValueError("Model state/parameter gradient/device invariant failed")
        replay.verify_bindings(bindings, deadline)
        report.update(complete=True, states_before=before, states_after=state_digest(model),
            parameter_names=names, parameter_count=sum(p.numel() for p in params),
            summary=summarize(cases, plan["thresholds"]["atom_threshold"]),
            max_gradient_sum_relative_error=max(r["gradient_sum_relative_error"] for r in cases),
            limitations=["Fixed targeted panel, not held-out or whole-molecule accuracy.",
                "Full set-head loss, not full multi-task training objective or upstream gradients.",
                "Eval-mode per-image derivative at unchanged initial, not training dropout or mixed-batch optimization.",
                "Unit-SGD first-order reference, not AdamW, warm momentum or a finite parameter update.",
                "Prototype is gradient ablation only; no production loss, threshold or matching changes."])
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report.update(checked_images=len(cases), elapsed_seconds=time.monotonic() - started,
            case_sha256={c["sample_key"]: replay.digest(output / (c["sample_key"] + ".json")) for c in cases})
        write(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        report = audit()
        print(json.dumps({"complete": report["complete"], "summary": report["summary"]}), flush=True)
        return 0
    receipt = runtime.run_budgeted([sys.executable, "-m", "tools.diagnose_curriculum_v02_selector_gradients", "--worker"],
                                  output=OUTPUT, budget=runtime.ResourceBudget(WALL_SECONDS))
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
