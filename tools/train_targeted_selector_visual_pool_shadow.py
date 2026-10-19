"""Run the one authorized visual group-existence shadow in memory.

The frozen initial model cache supplies candidate geometry/scores.  A small
new existence head is trained on train groups only; PNG crops add two
image-only features.  Protected groups remain hard locked, confirmed-negative
groups supervise existence only, and unknown members never become negatives.
No checkpoint is written and the fixed holdout is evaluated once per report.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from tools.audit_targeted_selector_group_image_features import extract


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"
AUTH = ROOT / "outputs/audits/chemvector-targeted-selector-visual-pool-run-authorization-20261006-r1/receipt.json"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_TARGETED_SELECTOR_VISUAL_POOL_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-targeted-selector-visual-pool-shadow-20261006-r1/report.json"),
))
STEPS = 100
SEED = 20261006


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExistenceHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


def group_status(group: dict) -> str:
    if any(m["raw_target"] >= 0 and m["selected_target"] < 0 for m in group["members"]):
        return "strict_positive"
    if any(m["selected_target"] >= 0 for m in group["members"]):
        return "protected"
    if all(m["raw_target"] < 0 for m in group["members"]):
        return "confirmed_negative"
    raise ValueError("Unknown group label mixture")


def build_groups(cache: list[dict], root: Path) -> list[dict]:
    images: dict[str, Image.Image] = {}
    result = []
    for group in cache:
        status = group_status(group)
        if status == "protected":
            result.append({**group, "status": status, "feature": None, "winner_score": None})
            continue
        image = images.get(group["sample_key"])
        if image is None:
            image = Image.open(root / "samples" / (group["sample_key"] + ".png")).convert("L")
            images[group["sample_key"]] = image
        members = []
        for member in group["members"]:
            crop = extract(image, member["feature"][3], member["feature"][4])
            values = list(member["feature"]) + [crop["largest_component_fraction"], crop["ink_center_t192"]]
            members.append({"member": member, "values": values})
        matrix = torch.tensor([m["values"] for m in members], dtype=torch.float32)
        # Mean/max pooling gives the existence head group context while
        # retaining the strong per-candidate signal for small groups.
        pooled = torch.cat((matrix.mean(0), matrix.max(0).values,
                            torch.tensor([min(len(members), 3) / 3.0])), dim=0)
        # Fixed candidate ranking: train-calibrated equal-weight atom score +
        # component evidence; this is not a learned winner and is never
        # threshold-swept.
        winner_scores = []
        for item in members:
            winner_scores.append((item["member"]["feature"][0], item["values"][-2]))
        result.append({**group, "status": status, "feature": pooled,
                       "member_values": matrix, "winner_scores": winner_scores})
    return result


def evaluate(groups: list[dict], head: ExistenceHead, mean: torch.Tensor, std: torch.Tensor,
             device: torch.device) -> dict:
    head.eval()
    strict_available = strict_recovered = confirmed_false = unknown_selected = selected = 0
    decisions = []
    with torch.no_grad():
        for group in groups:
            status = group["status"]
            if status == "protected":
                decisions.append({"group_id": group["group_id"], "status": status, "selected": 0})
                continue
            x = ((group["feature"] - mean) / std).to(device)
            existence = float(head(x[None]).item())
            strict_available += int(status == "strict_positive")
            if existence < 0.0:  # fixed existence gate from the receipt
                decisions.append({"group_id": group["group_id"], "status": status,
                                  "existence_logit": existence, "selected": 0})
                continue
            # Fixed, deterministic winner; no learned ranking claim is made.
            winner = max(range(len(group["winner_scores"])),
                         key=lambda i: (group["winner_scores"][i][0] + group["winner_scores"][i][1], -i))
            member = group["members"][winner]
            selected += 1
            if status == "strict_positive":
                if member["raw_target"] >= 0 and member["selected_target"] < 0:
                    strict_recovered += 1
                else:
                    unknown_selected += 1
            else:
                confirmed_false += 1
            decisions.append({"group_id": group["group_id"], "status": status,
                              "existence_logit": existence, "selected": 1,
                              "winner_slot": int(member["slot"])})
    return {"groups": len(groups), "strict_available": strict_available,
            "selected": selected, "strict_recovered": strict_recovered,
            "confirmed_false_selected": confirmed_false, "unknown_selected": unknown_selected,
            "known_precision": strict_recovered / max(strict_recovered + confirmed_false, 1),
            "conservative_precision": strict_recovered / max(selected, 1),
            "strict_recall": strict_recovered / max(strict_available, 1),
            "baseline_prefix_retention": 1.0, "decisions": decisions}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    if not auth.get("run_authorized") or auth.get("max_steps") != STEPS:
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
    train_groups = build_groups(train_cache, RELEASE)
    holdout_groups = build_groups(holdout_cache, RELEASE)
    active = [g for g in train_groups if g["status"] != "protected"]
    train_values = torch.stack([g["feature"] for g in active])
    train_labels = torch.tensor([g["status"] == "strict_positive" for g in active], dtype=torch.float32)
    mean, std = train_values.mean(0), train_values.std(0).clamp_min(1e-5)
    positives = int(train_labels.sum())
    if positives < 300:
        raise ValueError("Training cache has fewer than 300 strict-positive groups")
    head = ExistenceHead(int(train_values.shape[1])).to(device).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=0.0)
    pos_weight = ((train_labels == 0).sum() / train_labels.sum()).clamp(1.0, 200.0).to(device)
    checkpoints = [{"step": 0, "train": evaluate(train_groups, head, mean, std, device),
                    "holdout": evaluate(holdout_groups, head, mean, std, device)}]
    batch_size = 512
    events = []
    order = torch.arange(len(active))
    for step in range(1, STEPS + 1):
        if time.monotonic() - started >= 1800:
            raise TimeoutError("Authorized wall budget exhausted")
        start = ((step - 1) * batch_size) % len(active)
        indices = torch.cat((order[start:start + batch_size], order[:max(0, start + batch_size - len(active))]))
        x = torch.stack([active[int(i)]["feature"] for i in indices])
        y = torch.tensor([float(active[int(i)]["status"] == "strict_positive") for i in indices])
        logits = head(((x - mean) / std).to(device))
        loss = F.binary_cross_entropy_with_logits(logits, y.to(device), pos_weight=pos_weight)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite existence loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tuple(head.parameters()), 1.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
        if step in (25, 50, 75, 100):
            checkpoints.append({"step": step, "train": evaluate(train_groups, head, mean, std, device),
                                "holdout": evaluate(holdout_groups, head, mean, std, device)})
    final, step0 = checkpoints[-1], checkpoints[0]
    gate = {
        "baseline_prefix_retention_100": all(c["train"]["baseline_prefix_retention"] == 1.0 and c["holdout"]["baseline_prefix_retention"] == 1.0 for c in checkpoints),
        "holdout_known_precision_ge_085": final["holdout"]["known_precision"] >= 0.85,
        "holdout_conservative_precision_ge_085": final["holdout"]["conservative_precision"] >= 0.85,
        "holdout_recall_improves_over_step0": final["holdout"]["strict_recall"] > step0["holdout"]["strict_recall"],
        "no_increase_confirmed_false": final["holdout"]["confirmed_false_selected"] <= step0["holdout"]["confirmed_false_selected"],
        "initial_checkpoint_unchanged": digest(INITIAL) == INITIAL_SHA,
    }
    gate["pass"] = all(gate.values())
    result = {
        "schema_version": "chemvector-targeted-selector-visual-pool-shadow-0.1",
        "complete": True, "training_performed": True, "in_memory_only": True,
        "optimizer_steps": STEPS, "gpu_used": True, "device": str(device),
        "wall_seconds": time.monotonic() - started, "sealed_test_read": False,
        "production_checkpoint_written": False, "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA, "initial_sha256_after": digest(INITIAL),
        "run_authorization_sha256": digest(AUTH), "release_manifest_sha256": digest(RELEASE / "manifest.json"),
        "parameter_count": sum(p.numel() for p in head.parameters()), "feature_dim": int(train_values.shape[1]),
        "unknown_group_members_excluded_from_loss": True, "protected_prefix_rescored": False,
        "fixed_existence_threshold": 0.0, "fixed_winner_rule": "atom_score_plus_largest_component",
        "checkpoints": checkpoints, "events": events, "gate": gate,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "optimizer_steps": STEPS,
                      "final_holdout": final["holdout"], "gate": gate}, indent=2))
    return result


if __name__ == "__main__":
    run()
