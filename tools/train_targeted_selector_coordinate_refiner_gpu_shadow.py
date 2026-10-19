"""One authorized GPU shadow with a frozen candidate gate and refiner-only training."""

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
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-coordinate-release-20261007-r2"
AUTH = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-refiner-run-authorization-20261007-r2/receipt.json"
GATE = ROOT / "outputs/audits/chemvector-targeted-selector-frozen-coordinate-gate-20261007-r2"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-coordinate-refiner-shadow-20261007-r2/report.json"
STEPS = 100
SEED = 20261008
NEAR_PX = 10.0


class RefinerHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def residuals(rows: list[dict]) -> np.ndarray:
    return np.asarray([[r["atoms"][r["nearest_index"]][0] - r["candidate"][0],
                        r["atoms"][r["nearest_index"]][1] - r["candidate"][1]] for r in rows], dtype=np.float32)


def evaluate(rows: list[dict], selected: np.ndarray, corrections: np.ndarray) -> dict:
    labels = np.asarray([r["raw_distance"] <= NEAR_PX for r in rows])
    strict = np.asarray([r["strict"] for r in rows])
    refinable = int((selected & labels).sum())
    selected_dist = np.linalg.norm(residuals(rows) - corrections, axis=1)[selected]
    by_image: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not selected[index]:
            continue
        refined = np.asarray(row["candidate"], dtype=np.float64) + corrections[index]
        distances = [float(np.linalg.norm(refined - np.asarray(atom))) for atom in row["atoms"]]
        by_image[row["sample_key"]].append((index, min(distances)))
    assigned: list[int] = []
    for entries in by_image.values():
        used: set[int] = set()
        for index, _ in sorted(entries, key=lambda item: item[1]):
            row = rows[index]
            refined = np.asarray(row["candidate"], dtype=np.float64) + corrections[index]
            distances = [float(np.linalg.norm(refined - np.asarray(atom))) for atom in row["atoms"]]
            atom_index = int(np.argmin(distances))
            if distances[atom_index] <= 6.0 and atom_index not in used:
                used.add(atom_index)
                assigned.append(index)
    assignment_true = sum(bool(labels[i]) for i in assigned)
    return {
        "candidates": len(rows), "refinable_available": int(labels.sum()),
        "selected": int(selected.sum()), "refinable_selected": refinable,
        "far_selected": int((selected & ~labels).sum()),
        "strict_recovered": int((selected & strict).sum()),
        "refinable_precision": refinable / max(int(selected.sum()), 1),
        "refinable_recall": refinable / max(int(labels.sum()), 1),
        "strict_recall": int((selected & strict).sum()) / max(int(strict.sum()), 1),
        "selected_refined_mean_distance_px": float(selected_dist.mean()) if len(selected_dist) else None,
        "selected_refined_le6": float((selected_dist <= 6).mean()) if len(selected_dist) else None,
        "assigned": len(assigned), "assignment_refinable": assignment_true,
        "assignment_far_false": len(assigned) - assignment_true,
        "assignment_precision": assignment_true / max(len(assigned), 1),
    }


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    gate_report = json.loads((GATE / "report.json").read_text(encoding="utf-8"))
    if not auth.get("run_authorized") or auth.get("max_steps") != STEPS:
        raise ValueError("Run receipt does not authorize this exact shadow")
    if auth.get("seed") != SEED:
        raise ValueError("Seed mismatch")
    if digest(INITIAL) != INITIAL_SHA or auth.get("initial_sha256") != INITIAL_SHA:
        raise ValueError("Immutable initial checkpoint changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Authorized coordinate refiner shadow requires CUDA")
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    started = time.monotonic()
    train_rows, holdout_rows = collect("train"), collect("holdout")
    raw_train = np.asarray([r["vector"] for r in train_rows], dtype=np.float32)
    mean = raw_train.mean(0)
    std = np.maximum(raw_train.std(0), 1e-5)
    train_x = torch.tensor((raw_train - mean) / std, dtype=torch.float32, device=device)
    holdout_x = torch.tensor((np.asarray([r["vector"] for r in holdout_rows], dtype=np.float32) - mean) / std,
                             dtype=torch.float32, device=device)
    train_residual = torch.tensor(residuals(train_rows), dtype=torch.float32, device=device)
    holdout_residual = torch.tensor(residuals(holdout_rows), dtype=torch.float32, device=device)
    near = torch.tensor([r["raw_distance"] <= NEAR_PX for r in train_rows], dtype=torch.bool, device=device)
    # Candidate gate is frozen and loaded only to reproduce its fixed selection.
    payload = torch.load(GATE / "candidate-head.pt", map_location=device, weights_only=False)
    from tools.prepare_targeted_selector_frozen_coordinate_gate import CandidateHead
    candidate = CandidateHead(int(payload["feature_dim"])).to(device)
    candidate.load_state_dict(payload["state_dict"])
    candidate.eval()
    with torch.no_grad():
        train_selected = candidate(train_x).detach().cpu().numpy() >= 0.0
        holdout_selected = candidate(holdout_x).detach().cpu().numpy() >= 0.0
    refiner = RefinerHead(train_x.shape[1]).to(device).train()
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=3e-3, weight_decay=1e-3)
    events = []
    checkpoints = []
    for step in range(1, STEPS + 1):
        correction = refiner(train_x[near])
        loss = F.smooth_l1_loss(correction, train_residual[near])
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite refiner loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(tuple(refiner.parameters()), 5.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
        if step in (25, 50, 75, 100):
            with torch.no_grad():
                checkpoints.append({"step": step,
                    "train": evaluate(train_rows, train_selected, refiner(train_x).cpu().numpy()),
                    "holdout": evaluate(holdout_rows, holdout_selected, refiner(holdout_x).cpu().numpy())})
    final = checkpoints[-1]
    gate = {
        "frozen_candidate_gate_retained": final["train"]["refinable_precision"] >= 0.85 and final["holdout"]["refinable_precision"] >= 0.85,
        "holdout_assignment_precision_ge_085": final["holdout"]["assignment_precision"] >= 0.85,
        "holdout_strict_recovery_ge_50": final["holdout"]["strict_recovered"] >= 50,
        "initial_checkpoint_unchanged": digest(INITIAL) == INITIAL_SHA,
    }
    gate["pass"] = all(gate.values())
    result = {"schema_version": "chemvector-targeted-selector-coordinate-refiner-shadow-0.1",
              "complete": True, "training_performed": True, "in_memory_only": True,
              "training_scope": "refiner_only_frozen_candidate_gate", "optimizer_steps": STEPS,
              "gpu_used": True, "device": str(device), "wall_seconds": time.monotonic() - started,
              "sealed_test_read": False, "production_checkpoint_written": False, "deployment_authorized": False,
              "initial_sha256": INITIAL_SHA, "initial_sha256_after": digest(INITIAL),
              "run_authorization_sha256": digest(AUTH), "release_manifest_sha256": digest(RELEASE / "manifest.json"),
              "candidate_gate_report_sha256": digest(GATE / "report.json"),
              "candidate_gate_weights_sha256": digest(GATE / "candidate-head.pt"),
              "refiner_parameter_count": sum(p.numel() for p in refiner.parameters()),
              "near_threshold_px": NEAR_PX, "fixed_logit_gate": 0.0,
              "checkpoints": checkpoints, "events": events, "gate": gate}
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "optimizer_steps": STEPS, "final": final, "gate": gate}, indent=2))
    return result


if __name__ == "__main__":
    run()
