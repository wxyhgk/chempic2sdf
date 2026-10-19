#!/usr/bin/env python3
"""Target-independent endpoint-pair compatibility preflight.

This is the bounded experiment following the r5 vertex shadow.  A row is a
pair of *two* residual vertex candidates around the two endpoints of one
frozen bond query.  Features are made entirely from frozen inference and the
candidate field; target geometry is consulted only afterwards to make an
in-memory diagnostic label.  Thus a target coordinate/atom/bond never enters
the feature vector or a cache/artifact.

``CHEMVECTOR_ENDPOINT_PAIR_MODE=cpu`` (the default) runs the fixed 160/80
image CPU preflight.  ``...=shadow`` runs the same feature contract with a
frozen initial model and an isolated <=800-step GPU MLP; it writes only a
sidecar receipt/state and never calls the production graph path.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe

DATA = Path(os.environ.get(
    "CHEMVECTOR_ENDPOINT_PAIR_DATA",
    str(ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"),
))
# Keep the reviewed scored export selectable without editing the protocol.
# The default is retained for exact replay of the historical preflight.
CANDIDATES = Path(os.environ.get(
    "CHEMVECTOR_ENDPOINT_PAIR_CANDIDATES",
    str(ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5/candidates-scored.jsonl"),
))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
MODE = os.environ.get("CHEMVECTOR_ENDPOINT_PAIR_MODE", "cpu").lower()
OUT = Path(os.environ.get(
    "CHEMVECTOR_ENDPOINT_PAIR_OUTPUT",
    str(ROOT / ("outputs/audits/chemvector-identity-aligned-endpoint-pair-compatibility-r5-cpu-preflight"
                if MODE == "cpu" else "outputs/audits/chemvector-identity-aligned-endpoint-pair-compatibility-r5-shadow")))
)
IMAGE = 768.0
PAIR_RADIUS = 32.0
TARGET_RADIUS = 6.0
ENDPOINT_RADIUS = 12.0
DUPLICATE_RADIUS = 6.0
PARTNER_RADIUS = 24.0
LOCAL_RADIUS = 16.0
MAX_PER_SIDE = 24
TRAIN_LIMIT = 160
HOLDOUT_LIMIT = 80
SEED = 20261017
try:
    REQUESTED_STEPS = int(os.environ.get("CHEMVECTOR_ENDPOINT_PAIR_STEPS", "800"))
except ValueError as exc:
    raise RuntimeError("CHEMVECTOR_ENDPOINT_PAIR_STEPS must be an integer") from exc
if not 1 <= REQUESTED_STEPS <= 800:
    raise RuntimeError("CHEMVECTOR_ENDPOINT_PAIR_STEPS must be in [1,800]")
MAX_STEPS = REQUESTED_STEPS


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _candidates() -> dict[str, list[dict[str, float]]]:
    if not CANDIDATES.is_file():
        raise FileNotFoundError(CANDIDATES)
    result: dict[str, list[dict[str, float]]] = {}
    forbidden_top = {"targets", "target_indices", "matched", "element", "visible_element", "atom_id", "graph", "bonds", "ground_truth", "label"}
    for line_no, line in enumerate(CANDIDATES.read_text().splitlines(), 1):
        row = json.loads(line)
        key = row.get("sample_key")
        if not isinstance(key, str) or row.get("source") != "shadow":
            raise ValueError(f"invalid candidate source at line {line_no}")
        if set(row) & forbidden_top:
            raise ValueError(f"candidate artifact contains forbidden target field at line {line_no}")
        values = row.get("candidates")
        if not isinstance(values, list) or key in result:
            raise ValueError(f"invalid candidate row at line {line_no}")
        clean = []
        for v in values:
            if set(v) != {"x", "y", "score"}:
                raise ValueError(f"candidate contains non-inference field at line {line_no}")
            x, y, score = float(v["x"]), float(v["y"]), float(v["score"])
            if not (0 <= x <= 1 and 0 <= y <= 1 and np.isfinite([x, y, score]).all()):
                raise ValueError(f"invalid candidate value at line {line_no}")
            clean.append({"x": x, "y": y, "score": score})
        result[key] = clean
    if len(result) != 1170:
        raise ValueError(f"expected 1,170 scored candidate rows, found {len(result)}")
    return result


def _target(sample: dict[str, Any]) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    atoms = sorted(sample["graph"]["atoms"], key=lambda x: x["atom_id"])
    by_id = {x["atom_id"]: x for x in sample["layers"]["atoms"]}
    points = torch.tensor([by_id[x["atom_id"]]["center"] for x in atoms], dtype=torch.float32) / IMAGE
    bonds = [(int(x["begin_atom"]), int(x["end_atom"])) for x in sample["graph"]["bonds"]]
    return points, bonds


def _proposal_target(endpoint: torch.Tensor, points: torch.Tensor, bonds: list[tuple[int, int]]) -> tuple[int, tuple[int, int]] | None:
    """Map a frozen endpoint pair to one target bond for labels only."""
    if not len(bonds):
        return None
    scores = []
    for bi, (a, b) in enumerate(bonds):
        expected = points[[a, b]]
        fwd = torch.linalg.vector_norm((endpoint - expected) * IMAGE, dim=-1).amax()
        rev = torch.linalg.vector_norm((endpoint - expected.flip(0)) * IMAGE, dim=-1).amax()
        scores.append((float(torch.minimum(fwd, rev)), bi))
    scores.sort()
    if scores[0][0] > ENDPOINT_RADIUS or (len(scores) > 1 and scores[1][0] <= ENDPOINT_RADIUS):
        return None
    bi = scores[0][1]
    a, b = bonds[bi]
    if torch.linalg.vector_norm(endpoint[0] - points[a]) <= ENDPOINT_RADIUS / IMAGE and torch.linalg.vector_norm(endpoint[1] - points[b]) <= ENDPOINT_RADIUS / IMAGE:
        return bi, (a, b)
    if torch.linalg.vector_norm(endpoint[0] - points[b]) <= ENDPOINT_RADIUS / IMAGE and torch.linalg.vector_norm(endpoint[1] - points[a]) <= ENDPOINT_RADIUS / IMAGE:
        return bi, (b, a)
    return None


def _candidate_feature_rows(
    sample: dict[str, Any], values: list[dict[str, float]], bonds: dict[str, torch.Tensor], selected: dict[str, torch.Tensor]
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]], list[int]]:
    """Build rows without reading targets; labels are made in a second pass."""
    cp = torch.tensor([[x["x"], x["y"]] for x in values], dtype=torch.float32)
    cs = torch.tensor([x["score"] for x in values], dtype=torch.float32)
    baseline = selected["atom_geometry"][0, selected["atom_mask"][0].bool(), :2].detach().cpu()
    baseline_mask = selected["atom_mask"][0].bool()
    if "atom_scores" in selected:
        baseline_score = selected["atom_scores"][0, baseline_mask].detach().cpu().float()
    else:
        baseline_score = torch.ones(len(baseline), dtype=torch.float32)
    if len(baseline):
        residual = torch.cdist(cp * IMAGE, baseline * IMAGE).amin(dim=1) > DUPLICATE_RADIUS
    else:
        residual = torch.ones(len(cp), dtype=torch.bool)
    out: list[np.ndarray] = []
    meta: list[tuple[int, int, int, int]] = []
    labels: list[int] = []
    endpoints = bonds["bond_endpoints"][0].detach().cpu()
    active = bonds["bond_mask"][0].bool().detach().cpu()
    scores = bonds["bond_scores"][0].detach().cpu()
    types = bonds["bond_types"][0].detach().cpu()
    # Candidate neighborhood statistics are inference-only and computed before
    # any target lookup.
    # Immutable baseline atoms are owner pseudo-rows.  A pair may be
    # residual×residual or baseline×residual; baseline rows are never
    # rescored/claimed by this sidecar.
    owner_points = torch.cat((cp, baseline), dim=0)
    owner_scores = torch.cat((cs, baseline_score), dim=0)
    owner_is_candidate = torch.cat((residual, torch.zeros(len(baseline), dtype=torch.bool)))
    all_local = torch.cdist(owner_points * IMAGE, owner_points * IMAGE) if len(owner_points) else torch.empty((0, 0))
    for qi in torch.where(active)[0].tolist():
        ep = endpoints[qi]
        length = max(float(torch.linalg.vector_norm(ep[1] - ep[0]) * IMAGE), 1.0)
        type_vec = types[qi].numpy().astype(np.float32)
        side_lists: list[list[int]] = []
        side_stats: list[dict[int, tuple[float, float, float, float, float]]] = []
        for side in range(2):
            d = torch.linalg.vector_norm((owner_points - ep[side]) * IMAGE, dim=1)
            idx = [int(x) for x in torch.where(d <= PAIR_RADIUS)[0].tolist()]
            idx.sort(key=lambda i: (-float(owner_scores[i]), float(d[i]), i)); idx = idx[:MAX_PER_SIDE]
            side_lists.append(idx)
            stats: dict[int, tuple[float, float, float, float, float]] = {}
            for i in idx:
                other = 1 - side
                od = torch.linalg.vector_norm((owner_points - ep[other]) * IMAGE, dim=1)
                support = owner_is_candidate & (od <= PARTNER_RADIUS)
                local = owner_is_candidate & (all_local[i] <= LOCAL_RADIUS)
                stats[i] = (float(d[i] / IMAGE), float((owner_points[i, 0] - ep[side, 0])), float((owner_points[i, 1] - ep[side, 1])), float(support.sum()), float(owner_scores[support].max() if bool(support.any()) else 0.0))
                # Coordinates are normalized below; no target values involved.
            side_stats.append(stats)
        for i in side_lists[0]:
            for j in side_lists[1]:
                if i == j or float(torch.linalg.vector_norm((owner_points[i] - owner_points[j]) * IMAGE)) <= DUPLICATE_RADIUS:
                    continue
                a, b = side_stats[0][i], side_stats[1][j]
                pair_dist = float(torch.linalg.vector_norm((owner_points[i] - owner_points[j]) * IMAGE) / IMAGE)
                # 2 x (score, baseline flag, endpoint distance, dx, dy,
                # partner count, partner max score) + bond geometry/type.
                direction = (ep[1] - ep[0]).numpy().astype(np.float32)
                feat = np.concatenate((
                    np.asarray([float(owner_scores[i]), float(not owner_is_candidate[i]), *a[:4], a[4]], np.float32),
                    np.asarray([float(owner_scores[j]), float(not owner_is_candidate[j]), *b[:4], b[4]], np.float32),
                    np.asarray([scores[qi], length / IMAGE, direction[0], direction[1], pair_dist], np.float32),
                    type_vec,
                ))
                out.append(feat); meta.append((qi, i, j, 0)); labels.append(0)
    # Labels are deliberately assigned only after the complete feature pass.
    points, target_bonds = _target(sample)
    target_dist = torch.cdist(owner_points * IMAGE, points * IMAGE) if len(owner_points) and len(points) else torch.empty((len(owner_points), 0))
    for row, (qi, i, j, _) in enumerate(meta):
        mapped = _proposal_target(endpoints[qi], points, target_bonds)
        if mapped is None:
            continue
        _, (a, b) = mapped
        ci = torch.where(target_dist[i] <= TARGET_RADIUS)[0]
        cj = torch.where(target_dist[j] <= TARGET_RADIUS)[0]
        # Strict ownership: exactly one target atom in the 6 px ball and no
        # second atom in the 20 px ambiguity neighborhood.
        ni = torch.where(target_dist[i] <= 20.0)[0]; nj = torch.where(target_dist[j] <= 20.0)[0]
        # At least one owner must be residual: baseline×baseline is already
        # production-owned and cannot become a sidecar edge.
        labels[row] = int(bool(owner_is_candidate[i] or owner_is_candidate[j]) and len(ci) == 1 and len(cj) == 1 and len(ni) == 1 and len(nj) == 1 and int(ci[0]) == a and int(cj[0]) == b)
    return out, meta, labels


class PairMLP(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 24), nn.GELU(), nn.Linear(24, 1))
        if sum(p.numel() for p in self.parameters()) > 10_000:
            raise ValueError("pair sidecar exceeds 10k parameter budget")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _auprc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s, kind="stable"); yy = y[order]; pos = max(int(yy.sum()), 1)
    tp = np.cumsum(yy); return float(np.sum((tp / np.arange(1, len(yy)+1)) * yy) / pos)


def _fit_and_measure(x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray, device: torch.device, steps: int) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.manual_seed(SEED)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-5)
    model = PairMLP(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    xt, yt = torch.tensor((x-mean)/std, device=device), torch.tensor(y, device=device).float()
    rng = np.random.default_rng(SEED); pos = np.flatnonzero(y > 0); neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0: raise RuntimeError("pair slice lacks both classes")
    batch = min(2048, max(len(pos), 1) * 2); events = []
    for step in range(1, steps + 1):
        idx = np.r_[rng.choice(pos, batch // 2, replace=True), rng.choice(neg, batch // 2, replace=True)]; rng.shuffle(idx)
        ii = torch.tensor(idx, device=device); logits = model(xt[ii]); loss = F.binary_cross_entropy_with_logits(logits, yt[ii], pos_weight=torch.tensor(min(len(neg)/len(pos), 20.0), device=device))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step in (1, 100, 300, 800): events.append({"step": step, "loss": float(loss.detach())})
    with torch.inference_mode():
        st = model(xt).sigmoid().cpu().numpy(); sv = model(torch.tensor((xv-mean)/std, device=device)).sigmoid().cpu().numpy()
    # Train-only calibrated operating point; no holdout sweep.
    eligible = []
    for t in np.unique(st):
        sel = st >= t; p = float((sel & (y == 1)).sum() / max(int(sel.sum()), 1)); r = float((sel & (y == 1)).sum() / max(int(y.sum()), 1))
        # Calibrate conservatively on train; the eventual complete-edge gate
        # is 0.80, while this preflight reserves margin for the frozen
        # baseline/edge merge and forbids a holdout threshold sweep.
        if p >= .85: eligible.append((r, -float(t), float(t)))
    threshold = max(eligible)[2] if eligible else None
    def metric(yy, ss):
        sel = np.zeros(len(ss), bool) if threshold is None else ss >= threshold; tp = int((sel & (yy == 1)).sum()); n = int(sel.sum())
        return {"selected": n, "tp": tp, "fp": n-tp, "precision": tp/max(n, 1), "recall": tp/max(int(yy.sum()), 1), "auprc": _auprc(yy, ss)}
    return {"parameters": sum(p.numel() for p in model.parameters()), "steps": steps, "events": events, "threshold": threshold, "train": metric(y, st)}, {"holdout": metric(yv, sv), "model": model, "mean": mean, "std": std}


def run() -> dict[str, Any]:
    if OUT.exists(): raise FileExistsError(OUT)
    if digest(INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint digest changed")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    cand = _candidates()
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x]
    accepted = [x for x in rows if x.get("render_status") == "accepted"]
    if len(accepted) != 1170: raise RuntimeError(f"r5 accepted count changed: {len(accepted)}")
    train = sorted([x for x in accepted if x["split"] == "train"], key=lambda x:x["sample_key"])[:TRAIN_LIMIT]
    hold = sorted([x for x in accepted if x["split"] == "holdout"], key=lambda x:x["sample_key"])[:HOLDOUT_LIMIT]
    device = torch.device("cuda:0" if MODE == "shadow" else "cpu")
    if MODE == "shadow" and not torch.cuda.is_available(): raise RuntimeError("shadow requires CUDA")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA); model.to(device).eval().requires_grad_(False)
    split_rows = {}
    counts = {}
    for name, chosen in (("train", train), ("holdout", hold)):
        fx, yy = [], []
        for row in chosen:
            sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
            with torch.inference_mode():
                _, _, _, _, bonds, _, selected = structure_forward(model, image_tensor(sample["image"])[None].to(device), config, payload["provenance"]["atom_threshold"])
            # Convert frozen tensors to CPU before feature construction.  This
            # keeps the pair cache target-independent and bounded.
            b = {k: v.detach().cpu() for k, v in bonds.items()}; s = {k: v.detach().cpu() for k, v in selected.items()}
            f, _, labels = _candidate_feature_rows(sample, cand[row["sample_key"]], b, s); fx.extend(f); yy.extend(labels)
        split_rows[name] = (np.asarray(fx, np.float32), np.asarray(yy, np.float32)); counts[name] = {"pairs":len(yy), "positive":int(sum(yy)), "negative":int(len(yy)-sum(yy))}
    x, y = split_rows["train"]; xv, yv = split_rows["holdout"]
    train_metrics, held = _fit_and_measure(x, y, xv, yv, device, MAX_STEPS if MODE == "shadow" else min(MAX_STEPS, 500))
    result = {"schema_version":"chemvector-identity-aligned-endpoint-pair-compatibility-r5-0.2", "complete":True, "mode":MODE, "training_performed":True, "sealed_test_read":False, "deployment_authorized":False, "production_checkpoint_written":False, "initial_sha256":INITIAL_SHA, "candidate_artifact_sha256":digest(CANDIDATES), "fixed_slice":{"train_images":len(train),"holdout_images":len(hold),"pair_radius_px":PAIR_RADIUS,"target_radius_px":TARGET_RADIUS,"endpoint_radius_px":ENDPOINT_RADIUS,"max_per_side":MAX_PER_SIDE,"threshold_policy":"lowest train score reaching pair precision >= 0.85; max recall tie-break"}, "counts":counts, "metrics":train_metrics | {"holdout":held["holdout"]}, "gate":{"feature_target_independent":True,"split_fixed":True,"holdout_precision_ge_080":held["holdout"]["precision"] >= .80,"holdout_recall_ge_050":held["holdout"]["recall"] >= .50,"go":False,"next":"pair sidecar remains isolated; do not integrate production regardless of metric"}}
    OUT.mkdir(parents=True); (OUT / "report.json").write_text(json.dumps(result, indent=2)+"\n")
    if MODE == "shadow": torch.save({"schema_version":"chemvector-endpoint-pair-sidecar-state-0.1","initial_sha256":INITIAL_SHA,"state_dict":held["model"].state_dict()}, OUT / "sidecar-state.pt")
    print(json.dumps(result, indent=2)); return result


if __name__ == "__main__": run()
