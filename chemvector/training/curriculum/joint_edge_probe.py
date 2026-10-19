"""One-shot 400-attempt bounded trial for the conservative joint edge heads."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import (
    INITIAL, _chemical_digest, _decision, _evaluate, _nonchemical_digest, _pair_output,
    _save_delta,
)
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.interface_diagnostic import tensor_tree_digest, write_json
from chemvector.training.curriculum.joint_edge_objective import joint_edge_loss
from chemvector.training.curriculum.pair_scale_plan import sample_id
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.supervision_contract import pair_targets_with_unknown, select_known_pair_rows
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.curriculum.evaluation import FrozenOCRPredictor
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[3]
PREPARATION = ROOT / "outputs/audits/chemvector-chemical-head-probe-preparation-20260925"
REPORT = PREPARATION / "report.json"
SAMPLES = PREPARATION / "training-samples.json"
OUTPUT = ROOT / "outputs/experiments/chemvector-joint-edge-probe-20261005-400"
CHECKPOINTS = (100, 200, 400)
MAX_ATTEMPTS = 400
DEADLINE_SECONDS = 900
PAIR_THRESHOLD = 0.85
PRESENCE_WEIGHT = 0.25
CHEMICAL_WEIGHT = 1.0
REPLAY_WEIGHT = 1.0
AROMATIC_SAFETY_WEIGHT = 10.0
AROMATIC_ANCHOR_WEIGHT = 10.0
AROMATIC_ANCHOR_EPSILON = 0.005


def _digest_except_heads(model):
    return tensor_tree_digest({
        name: value for name, value in model.state_dict().items()
        if name not in {
            "pair.presence.weight", "pair.presence.bias",
            "pair.chemical.weight", "pair.chemical.bias",
        }
    })


def _reference_logits(hidden, heads, device):
    return (
        torch.nn.functional.linear(hidden, heads["presence.weight"].to(device),
                                   heads["presence.bias"].to(device)).squeeze(-1),
        torch.nn.functional.linear(hidden, heads["chemical.weight"].to(device),
                                   heads["chemical.bias"].to(device)),
    )


def _step(model, config, loader, manifest, row, collator, device, atom_threshold,
          initial_heads, optimizer, attempt):
    sample = loader(manifest, row)
    batch = collator([sample])
    tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        features, _, _, _, _, _, selected = structure_forward(model, tensors["pixel_values"], config, atom_threshold)
        mapping, _ = match_atom_targets(selected, tensors, config)
        contract = pair_targets_with_unknown(selected, mapping, tensors)
        pairs, rows = contract["pair_indices"], contract["sample_indices"]
        full_initial = _pair_output(model.pair, features, selected, pairs, rows)
        ref_presence, ref_chemical = _reference_logits(full_initial["hidden"].detach(), initial_heads, device)
        chosen = select_known_pair_rows(contract, selected, negative_ratio=6, minimum_negatives=48)
    if not len(chosen):
        return {"updated": False, "skip_reason": "no_known_pairs", "sampled_rows": 0}
    output = _pair_output(model.pair, features, selected, pairs, rows)
    chosen_output = {key: value[chosen] for key, value in output.items()}
    loss, details = joint_edge_loss(
        chosen_output,
        contract["known_mask"][chosen],
        contract["legacy_targets"]["presence"][chosen],
        contract["legacy_targets"]["chemical"][chosen],
        pair_indices=pairs[chosen], sample_indices=rows[chosen],
        presence_weight=PRESENCE_WEIGHT, chemical_weight=CHEMICAL_WEIGHT,
        replay_weight=REPLAY_WEIGHT,
        reference_presence_logits=ref_presence[chosen],
        reference_chemical_logits=ref_chemical[chosen],
        aromatic_safety_weight=AROMATIC_SAFETY_WEIGHT,
        aromatic_anchor_weight=AROMATIC_ANCHOR_WEIGHT,
        aromatic_anchor_epsilon=AROMATIC_ANCHOR_EPSILON,
    )
    if loss is None or not bool(torch.isfinite(loss)):
        raise ValueError("Non-finite/empty joint edge loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    params = [p for group in optimizer.param_groups for p in group["params"]]
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
        raise ValueError("Missing/non-finite trainable gradient")
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        after = _pair_output(model.pair, features, selected, pairs, rows)
    before_mask = full_initial["presence_logits"].sigmoid() >= PAIR_THRESHOLD
    after_mask = after["presence_logits"].sigmoid() >= PAIR_THRESHOLD
    if not torch.equal(before_mask, after_mask):
        raise ValueError(f"Presence threshold mask changed: {sample_id(row)}")
    return {
        "updated": True, "skip_reason": None, "sampled_rows": len(chosen),
        "positive_rows": int(contract["legacy_targets"]["presence"][chosen].sum()),
        "unknown_rows_in_loss": int((~contract["known_mask"][chosen]).sum()),
        "gradient_norm": float(norm), "loss": float(loss.detach()), "details": details,
        "presence_threshold_mask_unchanged": True,
    }


def run() -> dict:
    started = time.monotonic()
    if OUTPUT.exists():
        raise FileExistsError("Joint edge trial output already exists; no retry")
    prep = json.loads(REPORT.read_text(encoding="utf-8"))
    plan = json.loads(SAMPLES.read_text(encoding="utf-8"))
    sources = source_checks()
    if (not prep.get("complete") or prep.get("training_performed")
            or prep.get("sealed_test_read") or len(plan["samples"]) != 1200):
        raise ValueError("Fixed non-authorizing 1,200-row preparation is required")
    if file_digest(INITIAL) != prep["initial_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is required; no CPU fallback")
    torch.set_num_threads(2)
    torch.manual_seed(20261005)
    torch.cuda.manual_seed_all(20261005)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    model, config, payload = load_bundle_for_probe(INITIAL, prep["initial_checkpoint_sha256"])
    if payload["progress"]["optimizer_steps"] != 0:
        raise ValueError("Joint trial requires the zero-step initial checkpoint")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = [*model.pair.presence.parameters(), *model.pair.chemical.parameters()]
    for parameter in trainable:
        parameter.requires_grad_(True)
    frozen_before = _digest_except_heads(model)
    initial_heads = {name: value.detach().cpu().clone() for name, value in model.pair.state_dict().items()
                     if name in {"presence.weight", "presence.bias", "chemical.weight", "chemical.bias"}}
    loader = RecordLoader(sources["manifest"])
    rows_by_key = {sample_id(row): row for row in sources["manifest"]["samples"]}
    sample_rows = [rows_by_key[sample_id(row)] for row in plan["samples"]]
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=128, bond_queries=256), purpose="training")
    predictor = FrozenOCRPredictor(payload["provenance"], device)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    baseline, _ = _evaluate(model, config, sources["manifest"], loader, predictor, device)
    write_json(OUTPUT / "baseline-summary.json", {"records": baseline})
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0.0)
    counters = Counter(training_attempts=0, optimizer_steps=0, skipped_empty=0, evaluation_images=0)
    decisions = []
    events = []
    write_json(OUTPUT / "protocol.json", {
        "schema_version": "chemvector-joint-edge-probe-0.1",
        "training_authorized": True, "deployment_authorized": False,
        "user_statement": "Implement the approved 400-attempt conservative joint edge trial",
        "initial_sha256": file_digest(INITIAL), "sample_count": len(sample_rows),
        "maximum_attempts": MAX_ATTEMPTS, "wall_seconds": DEADLINE_SECONDS,
        "trainable_parameters": ["pair.presence.weight", "pair.presence.bias",
                                  "pair.chemical.weight", "pair.chemical.bias"],
        "objective": {"presence_weight": PRESENCE_WEIGHT, "chemical_weight": CHEMICAL_WEIGHT,
                      "replay_weight": REPLAY_WEIGHT, "aromatic_safety_weight": AROMATIC_SAFETY_WEIGHT,
                      "aromatic_anchor_weight": AROMATIC_ANCHOR_WEIGHT,
                      "aromatic_anchor_epsilon": AROMATIC_ANCHOR_EPSILON},
        "sealed_test_read": False, "deployment_authorized": False,
    })
    for attempt, row in enumerate(sample_rows[:MAX_ATTEMPTS], 1):
        if time.monotonic() - started >= DEADLINE_SECONDS:
            raise TimeoutError("Joint edge trial wall-clock budget reached")
        result = _step(model, config, loader, sources["manifest"], row, collator, device,
                       payload["provenance"]["atom_threshold"], initial_heads, optimizer, attempt)
        counters["training_attempts"] += 1
        counters["optimizer_steps"] += int(result["updated"])
        counters["skipped_empty"] += int(not result["updated"])
        events.append({"attempt": attempt, "key": sample_id(row), **result})
        if attempt in CHECKPOINTS:
            current, _ = _evaluate(model, config, sources["manifest"], loader, predictor, device)
            counters["evaluation_images"] += len(current)
            decision = _decision(baseline, current, complete=attempt == MAX_ATTEMPTS)
            if decision["reasons"]:
                decision["action"] = "stopped_keep_initial"
                decision["candidate_eligible"] = False
            decisions.append({"attempt": attempt, **decision})
            write_json(OUTPUT / f"validation-{attempt:06d}.json", {"records": current})
            write_json(OUTPUT / f"decision-{attempt:06d}.json", decisions[-1])
            _save_delta(OUTPUT / f"joint-{attempt:06d}.pt", {
                "schema_version": "chemvector-joint-edge-probe-delta-0.1",
                "initial_sha256": prep["initial_checkpoint_sha256"],
                "heads": {k: v.detach().cpu() for k, v in model.pair.state_dict().items()
                          if k in {"presence.weight", "presence.bias", "chemical.weight", "chemical.bias"}},
                "attempt": attempt, "deployment_authorized": False,
            })
            if decisions[-1]["reasons"]:
                break
    if _digest_except_heads(model) != frozen_before:
        raise ValueError("Frozen parameters changed")
    result = {
        "schema_version": "chemvector-joint-edge-probe-report-0.1",
        "complete": counters["training_attempts"] == MAX_ATTEMPTS,
        "training_performed": counters["optimizer_steps"] > 0,
        "gpu_used": True, "sealed_test_read": False, "deployment_authorized": False,
        "initial_sha256": prep["initial_checkpoint_sha256"], "counters": dict(counters),
        "events": events, "decisions": decisions,
        "stop_reason": decisions[-1]["reasons"] if decisions and decisions[-1]["reasons"] else None,
        "frozen_before_sha256": frozen_before, "frozen_after_sha256": _digest_except_heads(model),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(OUTPUT / "report.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
