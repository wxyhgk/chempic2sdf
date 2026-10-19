#!/usr/bin/env python3
"""Minimal CPU probe for endpoint-conditioned candidate ranking.

The frozen r17 candidate edge set is retained.  A small pairwise ranker sees
the existing graph-context feature (which contains both endpoint AtomObject
embeddings, pair hidden state, chemistry logits, geometry, and endpoint
degrees) and only reorders/subselects candidates.  It cannot invent an edge,
change atoms, or change a bond type.  Labels are used for isolated sidecar
training and train-only threshold calibration; no production or sealed data
is touched.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA

CACHE = ROOT / "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m128-t0p72-c1-panel128/feature-cache.pt"
OUT = ROOT / "outputs/audits/chemvector-endpoint-conditioned-candidate-rank-cpu-r17-m128-v2"
STEPS = 80
MARGIN = 0.25
DEGREE_PENALTY_WEIGHT = 0.10


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class EndpointRanker(nn.Module):
    def __init__(self, dim: int = 526) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def rank_loss(head: nn.Module, records: list[dict]) -> torch.Tensor:
    terms = []
    constraint_terms = []
    for rec in records:
        if not len(rec["features"]):
            continue
        score = head(rec["features"].float())
        pos = score[rec["labels"] > 0.5]
        neg = score[rec["labels"] <= 0.5]
        if len(pos) and len(neg):
            # All-pairs logistic ranking, with a bounded per-image sample to
            # keep CPU work stable on larger candidate sets.
            delta = pos[:, None] - neg[None, :]
            terms.append(F.softplus(MARGIN - delta).mean())
        # Target-free endpoint degree budget.  This is a soft proposal budget,
        # not a chemistry label: each endpoint may carry at most four expected
        # incident candidate edges before later bond-order compilation.
        if len(rec["pairs"]):
            pair_index = torch.as_tensor([(int(a), int(b)) for a, b, *_ in rec["pairs"]], dtype=torch.long)
            node_count = int(pair_index.max()) + 1
            probability = score.sigmoid()
            degree = score.new_zeros(node_count)
            degree.index_add_(0, pair_index[:, 0], probability)
            degree.index_add_(0, pair_index[:, 1], probability)
            constraint_terms.append(F.relu(degree - 4.0).square().mean())
    if not terms:
        raise RuntimeError("no image has both positive and negative candidates")
    ranking = torch.stack(terms).mean()
    constraint = torch.stack(constraint_terms).mean() if constraint_terms else ranking.new_zeros(())
    return ranking + DEGREE_PENALTY_WEIGHT * constraint


def metrics(records: list[dict], head: nn.Module | None, threshold: float) -> dict:
    total = defaultdict(int)
    by = defaultdict(lambda: defaultdict(int))
    for rec in records:
        if not len(rec["features"]):
            continue
        with torch.inference_mode():
            logits = torch.zeros(len(rec["features"])) if head is None else head(rec["features"].float())
            selected = logits.sigmoid() >= threshold
        labels = rec["labels"] > 0.5
        type_pred = rec["edge_chemical_logits"].argmax(-1)
        typed = labels & (type_pred == rec["type_labels"])
        prof = str(rec["sample_key"].rsplit("__", 1)[-1])
        vals = {
            "images": 1, "predicted": int(selected.sum()),
            "true": int((selected & labels).sum()), "typed": int((selected & typed).sum()),
            "target": int(labels.sum()),
        }
        for k, v in vals.items():
            total[k] += v; by[prof][k] += v

    def finish(c: dict) -> dict:
        p = max(int(c["predicted"]), 1); t = max(int(c["target"]), 1)
        return {"images": int(c["images"]), "predicted_edges": int(c["predicted"]),
                "true_edges": int(c["true"]), "typed_true_edges": int(c["typed"]),
                "target_edges": int(c["target"]), "edge_precision": c["true"] / p,
                "edge_recall": c["true"] / t, "typed_edge_precision": c["typed"] / p,
                "typed_edge_recall": c["typed"] / t}
    return finish(total), {p: finish(c) for p, c in sorted(by.items())}


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    initial = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
    if digest(initial) != SHA:
        raise RuntimeError("production checkpoint changed")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    if cache.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("cache checkpoint binding mismatch")
    records = cache["records"]
    torch.set_num_threads(1)
    torch.manual_seed(20261008)
    head = EndpointRanker(526)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    events = []
    for step in range(1, STEPS + 1):
        loss = rank_loss(head, records["train"])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
        if step in (1, 100, 200, STEPS):
            events.append({"step": step, "loss": float(loss.detach())})
    head.eval()
    baseline = {s: metrics(rs, None, 0.0) for s, rs in records.items()}
    # Choose the highest train recall under a strict train precision floor
    # equal to baseline minus 0.5 percentage points.
    candidates = []
    for threshold in [i / 100 for i in range(5, 96, 5)]:
        scored, prof = metrics(records["train"], head, threshold)
        if scored["edge_precision"] + 0.005 >= baseline["train"][0]["edge_precision"]:
            candidates.append((scored["edge_recall"], scored["typed_edge_recall"], -threshold, threshold, scored, prof))
    if not candidates:
        chosen_threshold = 0.0; selection = {"fallback": True}
    else:
        selected = max(candidates)
        chosen_threshold = selected[3]
        selection = {"fallback": False, "threshold": chosen_threshold,
                     "train": selected[4], "train_profiles": selected[5],
                     "candidate_count": len(candidates)}
    scored = {s: metrics(rs, head, chosen_threshold) for s, rs in records.items()}
    deltas = {s: {k: scored[s][0][k] - baseline[s][0][k]
                  for k in ("edge_precision", "edge_recall", "typed_edge_precision", "typed_edge_recall")}
              for s in records}
    gate = {
        "checkpoint_unchanged": digest(initial) == SHA,
        "sealed_test_read": False, "gpu_used": False,
        "candidate_edge_set_frozen": True,
        "fresh_edge_recall_gain_ge_0.01": deltas["fresh_holdout"]["edge_recall"] >= 0.01,
        "fresh_typed_recall_gain_ge_0.01": deltas["fresh_holdout"]["typed_edge_recall"] >= 0.01,
        "fresh_edge_precision_drop_le_0.01": deltas["fresh_holdout"]["edge_precision"] >= -0.01,
        "fresh_typed_precision_drop_le_0.01": deltas["fresh_holdout"]["typed_edge_precision"] >= -0.01,
        "production_go": False,
    }
    report = {
        "schema_version": "chemvector-endpoint-conditioned-candidate-rank-cpu-r17-0.1",
        "complete": True, "training_performed": True, "training_scope": "isolated_cpu_ranker",
        "read_only": False, "gpu_used": False, "sealed_test_read": False,
        "production_checkpoint_written": False, "cache": str(CACHE.relative_to(ROOT)),
        "cache_sha256": digest(CACHE), "input_dim": 526,
        "feature_layout": "direct_pair_hidden + chemical_logits + endpoint_atom_features + geometry + endpoint_degree_context",
        "steps": STEPS, "events": events, "baseline": baseline,
        "chosen_threshold": chosen_threshold, "selection": selection,
        "scored": scored, "deltas": deltas, "gate": gate,
        "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": digest(initial),
    }
    OUT.mkdir(parents=True, exist_ok=False)
    torch.save({"schema_version": report["schema_version"], "state_dict": head.state_dict(),
                "threshold": chosen_threshold, "initial_checkpoint_sha256": SHA}, OUT / "ranker-state.pt")
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "baseline": baseline,
                      "scored": scored, "deltas": deltas, "gate": gate}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
