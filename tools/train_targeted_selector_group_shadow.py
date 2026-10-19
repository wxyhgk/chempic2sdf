"""One authorized, bounded in-memory group-selector shadow run.

Only the new scorer is optimized.  The frozen initial model is represented by
the exact forward cache; no production checkpoint or model parameter is
written.  Holdout labels are read for evaluation after fitting and never enter
the optimizer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-group-release-20261006-r2"
AUTH = ROOT / "outputs/audits/chemvector-targeted-selector-group-run-authorization-20261006-r1/receipt.json"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_TARGETED_SELECTOR_SHADOW_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-targeted-selector-group-shadow-20261006-r1/report.json"),
))
FEATURE_DIM = 12
STEPS = 100
THRESHOLD = 0.5
SEED = 20261006


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GroupScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


def _group_status(group: dict) -> str:
    if any(m["raw_target"] >= 0 and m["selected_target"] < 0 for m in group["members"]):
        return "strict_positive"
    if any(m["selected_target"] >= 0 for m in group["members"]):
        return "protected"
    if all(m["raw_target"] < 0 for m in group["members"]):
        return "confirmed_negative"
    raise ValueError("Unknown group contains an unsupported label mixture")


def _features(group: dict) -> tuple[torch.Tensor, list[int]]:
    values = torch.tensor([m["feature"] for m in group["members"]], dtype=torch.float32)
    context = values.mean(0, keepdim=True).expand_as(values)
    return torch.cat((values, context), dim=1), [int(m["slot"]) for m in group["members"]]


def _examples(groups: list[dict]) -> list[dict]:
    result = []
    for group in groups:
        status = _group_status(group)
        features, slots = _features(group)
        labels = []
        for member in group["members"]:
            labels.append(1.0 if status == "strict_positive" and member["raw_target"] >= 0 and member["selected_target"] < 0 else 0.0)
        if status == "protected":
            continue
        # Other members in a strict-positive group are unknown; they are not
        # negatives and do not enter fitting.
        if status == "strict_positive":
            keep = [index for index, value in enumerate(labels) if value == 1.0]
            features = features[keep]
            labels = [labels[index] for index in keep]
            slots = [slots[index] for index in keep]
        result.append({"group": group, "status": status, "features": features,
                       "labels": torch.tensor(labels, dtype=torch.float32), "slots": slots})
    return result


def _evaluate(groups: list[dict], scorer: GroupScorer, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> dict:
    scorer.eval()
    strict_available = strict_recovered = confirmed_false = unknown_selected = selected = 0
    selected_by_profile = Counter()
    per_group = []
    with torch.no_grad():
        for group in groups:
            status = _group_status(group)
            if status == "protected":
                per_group.append({"group_id": group["group_id"], "status": status, "selected": 0})
                continue
            feature, slots = _features(group)
            scores = scorer(((feature - mean) / std).to(device)).sigmoid().cpu()
            winner = int(torch.argmax(scores))
            chosen = bool(float(scores[winner]) >= THRESHOLD)
            strict_available += int(status == "strict_positive")
            if not chosen:
                per_group.append({"group_id": group["group_id"], "status": status, "selected": 0})
                continue
            selected += 1
            member = group["members"][winner]
            if status == "strict_positive":
                if member["raw_target"] >= 0 and member["selected_target"] < 0:
                    strict_recovered += 1
                else:
                    unknown_selected += 1
            else:
                confirmed_false += 1
            selected_by_profile[group["profile_id"]] += 1
            per_group.append({"group_id": group["group_id"], "status": status,
                              "selected": 1, "winner_slot": int(slots[winner]),
                              "winner_probability": float(scores[winner])})
    known_precision = strict_recovered / max(strict_recovered + confirmed_false, 1)
    conservative_precision = strict_recovered / max(selected, 1)
    return {"groups": len(groups), "strict_available": strict_available,
            "selected": selected, "strict_recovered": strict_recovered,
            "confirmed_false_selected": confirmed_false, "unknown_selected": unknown_selected,
            "known_precision": known_precision, "conservative_precision": conservative_precision,
            "strict_recall": strict_recovered / max(strict_available, 1),
            "baseline_prefix_retention": 1.0, "selected_by_profile": dict(selected_by_profile),
            "decisions": per_group}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    if not auth.get("run_authorized") or auth.get("max_steps") != STEPS or auth.get("threshold") != THRESHOLD:
        raise ValueError("Run receipt does not authorize this exact shadow")
    if digest(INITIAL) != INITIAL_SHA or auth.get("initial_sha256") != INITIAL_SHA:
        raise ValueError("Immutable initial checkpoint changed")
    manifest = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("training_run_authorized") is not False or manifest.get("sealed_test_read") is not False:
        raise ValueError("Release is not run-gated")
    train_cache = json.loads((RELEASE / "train-group-cache.json").read_text(encoding="utf-8"))["groups"]
    holdout_cache = json.loads((RELEASE / "holdout-group-cache.json").read_text(encoding="utf-8"))["groups"]
    if not train_cache or not holdout_cache or any(g["split"] != "train" for g in train_cache) or any(g["split"] != "holdout" for g in holdout_cache):
        raise ValueError("Cache split contract changed")
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("Authorized shadow requires CUDA")
    device = torch.device("cuda:0")
    started = time.monotonic()
    scorer = GroupScorer().to(device).train()
    train_examples = _examples(train_cache)
    # Normalize from train only.  Protected groups never enter gradients.
    train_values = torch.cat([e["features"] for e in train_examples], 0)
    train_labels = torch.cat([e["labels"] for e in train_examples], 0)
    mean = train_values.mean(0)
    std = train_values.std(0).clamp_min(1e-5)
    positives = int(train_labels.sum())
    if positives < 50:
        raise ValueError("Training cache has fewer than 50 strict positives")
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=3e-3, weight_decay=0.0)
    pos_weight = ((train_labels == 0).sum() / train_labels.sum()).clamp(1.0, 100.0).to(device)
    schedule = [e for e in train_examples if e["status"] == "strict_positive"] + [e for e in train_examples if e["status"] == "confirmed_negative"]
    if not schedule:
        raise ValueError("Empty fitting schedule")
    checkpoints = [{"step": 0, "train": _evaluate(train_cache, scorer, mean, std, device),
                    "holdout": _evaluate(holdout_cache, scorer, mean, std, device)}]
    events = []
    for step in range(1, STEPS + 1):
        if time.monotonic() - started >= 1800:
            raise TimeoutError("Authorized wall budget exhausted")
        item = schedule[(step - 1) % len(schedule)]
        x = ((item["features"] - mean) / std).to(device)
        y = item["labels"].to(device)
        logits = scorer(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite group scorer loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tuple(scorer.parameters()), 1.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "group_id": item["group"]["group_id"], "loss": float(loss.detach()), "gradient_norm": float(norm)})
        if step in (25, 50, 75, 100):
            checkpoints.append({"step": step, "train": _evaluate(train_cache, scorer, mean, std, device),
                                "holdout": _evaluate(holdout_cache, scorer, mean, std, device)})
    source_after = digest(INITIAL)
    final = checkpoints[-1]
    step0 = checkpoints[0]
    gate = {
        "baseline_prefix_retention_100": all(c["train"]["baseline_prefix_retention"] == 1.0 and c["holdout"]["baseline_prefix_retention"] == 1.0 for c in checkpoints),
        "holdout_known_precision_ge_085": final["holdout"]["known_precision"] >= 0.85,
        "holdout_conservative_precision_ge_085": final["holdout"]["conservative_precision"] >= 0.85,
        "holdout_recall_improves_over_step0": final["holdout"]["strict_recall"] > step0["holdout"]["strict_recall"],
        "no_increase_confirmed_false": final["holdout"]["confirmed_false_selected"] <= step0["holdout"]["confirmed_false_selected"],
        "initial_checkpoint_unchanged": source_after == INITIAL_SHA,
    }
    gate["pass"] = all(gate.values())
    result = {
        "schema_version": "chemvector-targeted-selector-group-shadow-0.1",
        "complete": True, "training_performed": True, "in_memory_only": True,
        "optimizer_steps": STEPS, "gpu_used": True, "device": str(device),
        "wall_seconds": time.monotonic() - started, "sealed_test_read": False,
        "production_checkpoint_written": False, "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA, "initial_sha256_after": source_after,
        "run_authorization_sha256": digest(AUTH), "release_manifest_sha256": digest(RELEASE / "manifest.json"),
        "parameter_count": sum(p.numel() for p in scorer.parameters()), "feature_dim": FEATURE_DIM,
        "unknown_group_members_excluded_from_loss": True, "protected_prefix_rescored": False,
        "threshold": THRESHOLD, "checkpoints": checkpoints, "events": events,
        "gate": gate,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "optimizer_steps": STEPS, "final_holdout": final["holdout"], "gate": gate}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
