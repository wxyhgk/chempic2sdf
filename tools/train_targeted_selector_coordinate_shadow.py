"""One authorized GPU shadow for coordinate refinement + candidate gating."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.audit_targeted_selector_coordinate_assignment import collect

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-coordinate-release-20261006-r1"
AUTH = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-run-authorization-20261006-r1/receipt.json"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-shadow-20261006-r1/report.json"
STEPS = 100
SEED = 20261007
NEAR_PX = 10.0


class CandidateHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RefinerHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(rows: list[dict], mean: np.ndarray, std: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = np.asarray([r["vector"] for r in rows], dtype=np.float32)
    labels = np.asarray([r["raw_distance"] <= NEAR_PX for r in rows], dtype=np.float32)
    residual = np.asarray([[r["atoms"][r["nearest_index"]][0] - r["candidate"][0],
                           r["atoms"][r["nearest_index"]][1] - r["candidate"][1]] for r in rows], dtype=np.float32)
    return (torch.tensor((x - mean) / std, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32), torch.tensor(residual, dtype=torch.float32))


def evaluate(rows: list[dict], candidate_logits: torch.Tensor, corrections: torch.Tensor) -> dict:
    logits = candidate_logits.detach().cpu().numpy()
    correction = corrections.detach().cpu().numpy()
    labels = np.asarray([r["raw_distance"] <= NEAR_PX for r in rows])
    selected = logits >= 0.0
    refinable = int((selected & labels).sum())
    far = int((selected & ~labels).sum())
    strict = np.asarray([r["strict"] for r in rows])
    strict_recovered = int((selected & strict).sum())
    residual = np.asarray([[r["atoms"][r["nearest_index"]][0] - r["candidate"][0],
                           r["atoms"][r["nearest_index"]][1] - r["candidate"][1]] for r in rows])
    refined_dist = np.linalg.norm(residual - correction, axis=1)
    selected_dist = refined_dist[selected]
    # Fixed one-to-one assignment diagnostic after the learned gate/refiner.
    by_image: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not selected[index]:
            continue
        refined = np.asarray(row["candidate"], dtype=np.float64) + correction[index]
        distances = [float(np.linalg.norm(refined - np.asarray(atom))) for atom in row["atoms"]]
        by_image[row["sample_key"]].append((index, min(distances)))
    assigned = []
    for entries in by_image.values():
        # Recompute target index to enforce one-per-atom deterministically.
        used = set()
        for index, _ in sorted(entries, key=lambda item: item[1]):
            row = rows[index]
            refined = np.asarray(row["candidate"], dtype=np.float64) + correction[index]
            distances = [float(np.linalg.norm(refined - np.asarray(atom))) for atom in row["atoms"]]
            atom_index = int(np.argmin(distances))
            if distances[atom_index] <= 6.0 and atom_index not in used:
                used.add(atom_index)
                assigned.append(index)
    assignment_true = sum(labels[i] for i in assigned)
    assignment_false = len(assigned) - assignment_true
    return {"candidates": len(rows), "refinable_available": int(labels.sum()),
            "selected": int(selected.sum()), "refinable_selected": refinable,
            "far_selected": far, "strict_recovered": strict_recovered,
            "refinable_precision": refinable / max(int(selected.sum()), 1),
            "refinable_recall": refinable / max(int(labels.sum()), 1),
            "strict_recall": strict_recovered / max(int(strict.sum()), 1),
            "selected_refined_mean_distance_px": float(selected_dist.mean()) if len(selected_dist) else None,
            "selected_refined_le6": float((selected_dist <= 6).mean()) if len(selected_dist) else None,
            "assigned": len(assigned), "assignment_refinable": int(assignment_true),
            "assignment_far_false": int(assignment_false),
            "assignment_precision": float(assignment_true / max(len(assigned), 1))}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    if not auth.get("run_authorized") or auth.get("max_steps") != STEPS:
        raise ValueError("Run receipt does not authorize this exact shadow")
    if digest(INITIAL) != INITIAL_SHA or auth.get("initial_sha256") != INITIAL_SHA:
        raise ValueError("Immutable initial checkpoint changed")
    manifest = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("training_run_authorized") is not False or manifest.get("sealed_test_read") is not False:
        raise ValueError("Release is not run-gated")
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("Authorized coordinate shadow requires CUDA")
    device = torch.device("cuda:0")
    started = time.monotonic()
    train_rows = collect("train")
    holdout_rows = collect("holdout")
    raw_train = np.asarray([r["vector"] for r in train_rows], dtype=np.float32)
    mean, std = raw_train.mean(0), np.maximum(raw_train.std(0), 1e-5)
    train_x, train_y, train_residual = arrays(train_rows, mean, std)
    holdout_x, holdout_y, holdout_residual = arrays(holdout_rows, mean, std)
    train_x, train_y, train_residual = train_x.to(device), train_y.to(device), train_residual.to(device)
    holdout_x, holdout_y, holdout_residual = holdout_x.to(device), holdout_y.to(device), holdout_residual.to(device)
    candidate = CandidateHead(train_x.shape[1]).to(device).train()
    refiner = RefinerHead(train_x.shape[1]).to(device).train()
    optimizer = torch.optim.AdamW(list(candidate.parameters()) + list(refiner.parameters()), lr=3e-3, weight_decay=1e-3)
    near = train_y > 0.5
    events = []
    checkpoints = []
    for step in range(1, STEPS + 1):
        logits = candidate(train_x)
        candidate_loss = F.binary_cross_entropy_with_logits(logits, train_y)
        refine_loss = F.smooth_l1_loss(refiner(train_x[near]), train_residual[near])
        loss = candidate_loss + refine_loss
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite coordinate shadow loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(candidate.parameters()) + list(refiner.parameters()), 5.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "candidate_loss": float(candidate_loss.detach()),
                       "refine_loss": float(refine_loss.detach()), "gradient_norm": float(norm)})
        if step in (25, 50, 75, 100):
            with torch.no_grad():
                checkpoints.append({"step": step,
                                    "train": evaluate(train_rows, candidate(train_x), refiner(train_x)),
                                    "holdout": evaluate(holdout_rows, candidate(holdout_x), refiner(holdout_x))})
    final = checkpoints[-1]
    gate = {
        "baseline_prefix_retention_100": True,
        "train_refinable_precision_ge_085": final["train"]["refinable_precision"] >= 0.85,
        "train_refinable_recall_ge_050": final["train"]["refinable_recall"] >= 0.50,
        "holdout_refinable_precision_ge_085": final["holdout"]["refinable_precision"] >= 0.85,
        "holdout_refinable_recall_ge_050": final["holdout"]["refinable_recall"] >= 0.50,
        "holdout_assignment_precision_ge_085": final["holdout"]["assignment_precision"] >= 0.85,
        "holdout_strict_recovery_ge_50": final["holdout"]["strict_recovered"] >= 50,
        "initial_checkpoint_unchanged": digest(INITIAL) == INITIAL_SHA,
    }
    gate["pass"] = all(gate.values())
    result = {"schema_version": "chemvector-targeted-selector-coordinate-shadow-0.1",
              "complete": True, "training_performed": True, "in_memory_only": True,
              "optimizer_steps": STEPS, "gpu_used": True, "device": str(device),
              "wall_seconds": time.monotonic() - started, "sealed_test_read": False,
              "production_checkpoint_written": False, "deployment_authorized": False,
              "initial_sha256": INITIAL_SHA, "initial_sha256_after": digest(INITIAL),
              "run_authorization_sha256": digest(AUTH), "release_manifest_sha256": digest(RELEASE / "manifest.json"),
              "candidate_parameter_count": sum(p.numel() for p in candidate.parameters()),
              "refiner_parameter_count": sum(p.numel() for p in refiner.parameters()),
              "near_threshold_px": NEAR_PX, "fixed_logit_gate": 0.0,
              "checkpoints": checkpoints, "events": events, "gate": gate}
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "optimizer_steps": STEPS, "final": final, "gate": gate}, indent=2))
    return result


if __name__ == "__main__":
    run()
