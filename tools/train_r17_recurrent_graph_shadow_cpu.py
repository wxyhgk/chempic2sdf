#!/usr/bin/env python3
"""Bounded CPU shadow for a weight-tied recurrent graph refinement head.

The experiment deliberately starts *after* frozen graph artifact export.  It
does not call the vision model and does not create atoms or pairs.  A small
edge/node state machine is applied three times with shared parameters.  Its
only learned output is a bounded residual on the frozen bond-presence logit;
zero initialization makes iteration zero exactly the frozen artifact output.

Targets are used to fit/evaluate the audit sidecar only.  They never enter the
feature tensors, the candidate set, or the prediction digest.  This is a
shadow experiment, not a production checkpoint or a graph-transaction path.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / os.environ.get(
    "R17_REC_ARTIFACT",
    "outputs/audits/chemvector-r17-known-clean-graph-artifact-v1/graph-artifact.jsonl",
)
MANIFEST = ROOT / os.environ.get(
    "R17_REC_MANIFEST",
    "outputs/audits/chemvector-r17-selector-known-clean-shadow-cohort-v1/manifest.json",
)
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/samples"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
OUT = ROOT / os.environ.get(
    "R17_REC_OUT",
    "outputs/audits/chemvector-r17-recurrent-graph-shadow-cpu-v1",
)
STEPS = int(os.environ.get("R17_REC_STEPS", "500"))
ITERATIONS = int(os.environ.get("R17_REC_ITERATIONS", "3"))
THRESHOLD = float(os.environ.get("R17_REC_THRESHOLD", "0.72"))
RADIUS_PX = 16.0
HIDDEN = 24
TYPE_NAMES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _nearest(point: torch.Tensor, centers: torch.Tensor, radius: float) -> tuple[int | None, float]:
    d = torch.linalg.vector_norm(centers - point[None], dim=1)
    value, index = torch.min(d, dim=0)
    value_f = float(value)
    return (int(index), value_f) if value_f <= radius else (None, value_f)


def _target(sample_key: str) -> tuple[torch.Tensor, dict[tuple[int, int], str]]:
    layers = json.loads((SOURCE / f"{sample_key}.layers.json").read_text())
    graph = json.loads((SOURCE / f"{sample_key}.graph.json").read_text())
    atoms = sorted(layers["atoms"], key=lambda row: int(row["atom_id"]))
    centers = torch.tensor([row["center"] for row in atoms], dtype=torch.float32)
    edges: dict[tuple[int, int], str] = {}
    for bond in graph["bonds"]:
        edge = tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        edges[edge] = str(bond.get("chemical_type", "SINGLE")).upper()
    return centers, edges


def _artifact_rows() -> list[dict[str, Any]]:
    rows = []
    with ARTIFACT.open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _label_row(row: dict[str, Any]) -> tuple[torch.Tensor, list[str | None]]:
    """Map endpoints to target atoms, matching the independent artifact audit."""
    centers, truth = _target(row["sample_key"])
    endpoint = torch.tensor(row["bond_endpoints"], dtype=torch.float32) * 768.0
    labels = torch.zeros(endpoint.size(0), dtype=torch.float32)
    types: list[str | None] = [None] * endpoint.size(0)
    for i in range(endpoint.size(0)):
        a, _ = _nearest(endpoint[i, 0], centers, RADIUS_PX)
        b, _ = _nearest(endpoint[i, 1], centers, RADIUS_PX)
        if a is None or b is None or a == b:
            continue
        edge = tuple(sorted((a, b)))
        if edge in truth:
            labels[i] = 1.0
            types[i] = truth[edge]
    return labels, types


def _features(row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str | None]]:
    """Build target-free edge/node features and frozen endpoint incidence."""
    points = torch.tensor(row["atom_points"], dtype=torch.float32)
    keep = torch.tensor(row["atom_keep_mask"], dtype=torch.bool)
    endpoints = torch.tensor(row["bond_endpoints"], dtype=torch.float32)
    presence = torch.tensor(row["bond_presence_logits"], dtype=torch.float32).clamp(0.0, 1.0)
    type_logits = torch.tensor(row["bond_type_logits"], dtype=torch.float32)
    mask = torch.tensor(row["bond_mask"], dtype=torch.bool)
    # Endpoint -> frozen atom incidence is derived only from artifact geometry.
    dist_a = torch.cdist(endpoints[:, 0], points)
    dist_b = torch.cdist(endpoints[:, 1], points)
    dist_a[:, ~keep] = 1e6
    dist_b[:, ~keep] = 1e6
    ia = dist_a.argmin(dim=1)
    ib = dist_b.argmin(dim=1)
    valid = mask & keep[ia] & keep[ib]
    delta = endpoints[:, 1] - endpoints[:, 0]
    length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    midpoint = (endpoints[:, 0] + endpoints[:, 1]) * 0.5
    # Node input has no atom identity or target information.
    node = torch.cat((points, keep.float()[:, None], torch.tensor(row["atom_raw_mask"], dtype=torch.float32)[:, None]), dim=-1)
    edge = torch.cat((
        presence[:, None],
        type_logits,
        endpoints.reshape(-1, 4),
        delta,
        length,
        midpoint,
        valid.float()[:, None],
    ), dim=-1)
    labels, types = _label_row(row)
    edge_indices = torch.stack((ia, ib), dim=-1)
    return node, edge, edge_indices, valid, labels, types


class LoopedGraph(nn.Module):
    """Weight-tied node/edge updates with a bounded zero-initialized residual."""

    def __init__(self, edge_dim: int, node_dim: int = 4, hidden: int = HIDDEN, iterations: int = ITERATIONS):
        super().__init__()
        self.iterations = iterations
        self.node_encoder = nn.Sequential(nn.LayerNorm(node_dim), nn.Linear(node_dim, hidden), nn.GELU())
        self.edge_encoder = nn.Sequential(nn.LayerNorm(edge_dim + 2 * hidden + 1), nn.Linear(edge_dim + 2 * hidden + 1, hidden), nn.GELU())
        self.edge_update = nn.Sequential(nn.LayerNorm(2 * hidden + hidden + 1), nn.Linear(3 * hidden + 1, hidden), nn.GELU())
        self.node_update = nn.Sequential(nn.LayerNorm(2 * hidden + 2), nn.Linear(2 * hidden + 2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.delta = nn.Linear(hidden, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, node_x: torch.Tensor, edge_x: torch.Tensor, indices: torch.Tensor, valid: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        # Shapes: node [B,N,4], edge [B,E,F], indices [B,E,2], valid/base [B,E].
        node = self.node_encoder(node_x.float())
        base_logit = torch.logit(base.float().clamp(1e-4, 1.0 - 1e-4))
        logits = base_logit
        batch = node.size(0)
        batch_index = torch.arange(batch, device=node.device)[:, None]
        for _ in range(self.iterations):
            ia, ib = indices[..., 0], indices[..., 1]
            na = node[batch_index, ia]
            nb = node[batch_index, ib]
            probability = logits.sigmoid() * valid.float()
            eh = self.edge_encoder(torch.cat((edge_x.float(), na, nb, probability[..., None]), dim=-1))
            msg = eh * probability[..., None]
            aggregate = torch.zeros_like(node)
            aggregate.scatter_add_(1, ia[..., None].expand(-1, -1, node.size(-1)), msg)
            aggregate.scatter_add_(1, ib[..., None].expand(-1, -1, node.size(-1)), msg)
            degree = torch.zeros(batch, node.size(1), 1, device=node.device, dtype=node.dtype)
            degree.scatter_add_(1, ia[..., None], probability[..., None])
            degree.scatter_add_(1, ib[..., None], probability[..., None])
            node = node + self.node_update(torch.cat((node, aggregate / degree.clamp_min(1.0), (degree / 4.0).clamp_max(1.0), valid.new_zeros(batch, node.size(1), 1)), dim=-1))
            na = node[batch_index, ia]
            nb = node[batch_index, ib]
            edge_state = self.edge_update(torch.cat((na, nb, eh, probability[..., None]), dim=-1))
            residual = 2.0 * torch.tanh(self.delta(edge_state).squeeze(-1) / 2.0)
            logits = (logits + residual).masked_fill(~valid, -30.0)
        return logits


def _score(rows: list[dict[str, Any]], probabilities: list[torch.Tensor] | None = None, threshold: float = THRESHOLD) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    total = defaultdict(float)
    by = defaultdict(lambda: defaultdict(float))
    for idx, row in enumerate(rows):
        _, _, _, valid, labels, types = _features(row)
        p = torch.tensor(row["bond_presence_logits"], dtype=torch.float32) if probabilities is None else probabilities[idx]
        selected = (p >= threshold) & valid
        centers, truth = _target(row["sample_key"])
        endpoints = torch.tensor(row["bond_endpoints"], dtype=torch.float32) * 768.0
        predicted: set[tuple[int, int]] = set(); true: set[tuple[int, int]] = set(); typed_edges: set[tuple[int, int]] = set()
        type_logits = torch.tensor(row["bond_type_logits"], dtype=torch.float32)
        for i in torch.where(selected)[0].tolist():
            a, _ = _nearest(endpoints[i, 0], centers, RADIUS_PX); b, _ = _nearest(endpoints[i, 1], centers, RADIUS_PX)
            if a is None or b is None or a == b: continue
            e = tuple(sorted((a, b))); predicted.add(e)
            if e in truth:
                true.add(e)
                tid = int(row["bond_type_ids"][i]) - 4 if int(row["bond_type_ids"][i]) >= 4 else int(row["bond_type_ids"][i])
                if 0 <= tid < 4 and TYPE_NAMES[tid] == truth[e]:
                    typed_edges.add(e)
        profile = row["profile_id"]
        for bucket, value in (("predicted", len(predicted)), ("true", len(true)), ("typed", len(typed_edges)), ("target", len(truth))):
            total[bucket] += value; by[profile][bucket] += value
        by[profile]["images"] += 1; total["images"] += 1
    def finish(c: dict[str, float]) -> dict[str, float]:
        return {"images": c["images"], "predicted_edges": c["predicted"], "true_edges": c["true"], "typed_true_edges": c["typed"], "target_edges": c["target"], "edge_precision": c["true"] / max(c["predicted"], 1), "edge_recall": c["true"] / max(c["target"], 1), "typed_edge_precision": c["typed"] / max(c["predicted"], 1), "typed_edge_recall": c["typed"] / max(c["target"], 1)}
    return finish(total), {profile: finish(values) for profile, values in by.items()}


def run() -> None:
    if OUT.exists(): raise FileExistsError(OUT)
    if not INITIAL.exists() or digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("production initial checkpoint is missing or changed")
    manifest = json.loads(MANIFEST.read_text())["rows"]
    rows = _artifact_rows()
    expected = {r["sample_key"]: r for r in manifest}
    if len(rows) != len(expected) or {r["sample_key"] for r in rows} != set(expected): raise RuntimeError("artifact/manifest mismatch")
    rows = sorted(rows, key=lambda r: r["sample_key"])
    train = [r for r in rows if r["split"] == "train"]
    hold = [r for r in rows if r["split"] == "holdout"]
    if {expected[r["sample_key"]]["molecule_id"] for r in train} & {expected[r["sample_key"]]["molecule_id"] for r in hold}: raise RuntimeError("identity overlap")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(20261008); random.seed(20261008)
    train_pack = [_features(r) for r in train]; hold_pack = [_features(r) for r in hold]
    node = torch.stack([p[0] for p in train_pack]); edge = torch.stack([p[1] for p in train_pack]); indices = torch.stack([p[2] for p in train_pack]); valid = torch.stack([p[3] for p in train_pack]); base = edge[..., 0]; labels = torch.stack([p[4] for p in train_pack])
    model = LoopedGraph(edge.size(-1), iterations=ITERATIONS); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    load_state = os.environ.get("R17_REC_LOAD_STATE")
    if load_state:
        payload = torch.load(ROOT / load_state, map_location="cpu", weights_only=True)
        if payload.get("artifact_sha256") != digest(ARTIFACT): raise RuntimeError("sidecar/artifact digest mismatch")
        model.load_state_dict(payload["state_dict"])
    pos_weight = ((labels.numel() - labels.sum()) / labels.sum().clamp_min(1)).clamp(1, 32)
    events = []
    for step in range(1, STEPS + 1) if not load_state else []:
        logits = model(node, edge, indices, valid, base)
        loss = F.binary_cross_entropy_with_logits(logits[valid], labels[valid], pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step in (1, 100, 300, STEPS): events.append({"step": step, "loss": float(loss.detach())})
    model.eval()
    with torch.inference_mode():
        hold_node = torch.stack([p[0] for p in hold_pack]); hold_edge = torch.stack([p[1] for p in hold_pack]); hold_indices = torch.stack([p[2] for p in hold_pack]); hold_valid = torch.stack([p[3] for p in hold_pack]); hold_base = hold_edge[..., 0]
        refined = model(hold_node, hold_edge, hold_indices, hold_valid, hold_base).sigmoid()
    baseline, baseline_profiles = _score(hold)
    scored, scored_profiles = _score(hold, [refined[i] for i in range(refined.size(0))])
    sweep = {}
    for threshold in (0.72, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95):
        candidate, profiles = _score(hold, [refined[i] for i in range(refined.size(0))], threshold)
        delta = {p: {k: profiles[p][k] - baseline_profiles[p][k] for k in ("edge_recall", "typed_edge_recall", "edge_precision", "typed_edge_precision")} for p in profiles}
        sweep[str(threshold)] = {"scored": candidate, "deltas": delta, "strict_precision_gate": all(v["edge_precision"] >= -0.01 and v["typed_edge_precision"] >= -0.01 for v in delta.values())}
    digest_payload = torch.round(refined * 1_000_000).to(torch.int64).numpy().tobytes(); pred_digest = hashlib.sha256(digest_payload).hexdigest()
    deltas = {p: {k: scored_profiles[p][k] - baseline_profiles[p][k] for k in ("edge_recall", "typed_edge_recall", "edge_precision", "typed_edge_precision")} for p in scored_profiles}
    gate = {"checkpoint_unchanged": True, "sealed_test_read": False, "target_free_features": True, "weight_tied_iterations": ITERATIONS, "oa13_typed_recall_gain": deltas.get("oa13", {}).get("typed_edge_recall", 0) > 0, "oa33_typed_recall_gain": deltas.get("oa33", {}).get("typed_edge_recall", 0) > 0, "each_profile_typed_precision_drop_le_0.01": all(v["typed_edge_precision"] >= -0.01 for v in deltas.values()), "each_profile_edge_precision_drop_le_0.01": all(v["edge_precision"] >= -0.01 for v in deltas.values()), "production_go": False}
    after_sha = digest(INITIAL)
    gate["checkpoint_unchanged"] = after_sha == INITIAL_SHA
    result = {"schema_version": "chemvector-r17-recurrent-graph-shadow-cpu-0.1", "complete": True, "training_performed": True, "training_scope": "frozen_graph_artifact_weight_tied_recurrent_cpu_shadow", "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "initial_checkpoint_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": after_sha, "artifact": str(ARTIFACT.relative_to(ROOT)), "manifest": str(MANIFEST.relative_to(ROOT)), "artifact_sha256": digest(ARTIFACT), "target_free_contract": {"feature_source": "frozen graph artifact only", "target_fields_in_feature_tensors": [], "decision_digest": pred_digest, "holdout_target_access": "post-hoc scoring only", "candidate_set_frozen": True}, "train_images": len(train), "holdout_images": len(hold), "steps": STEPS, "iterations": ITERATIONS, "events": events, "holdout_prediction_digest": pred_digest, "baseline": baseline, "scored": scored, "baseline_profiles": baseline_profiles, "scored_profiles": scored_profiles, "deltas": deltas, "threshold_sweep": sweep, "gate": gate}
    OUT.mkdir(parents=True); torch.save({"schema_version": result["schema_version"], "state_dict": model.state_dict(), "artifact_sha256": result["artifact_sha256"], "target_free_features": True}, OUT / "sidecar-state.pt"); (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"output": str(OUT / "report.json"), "baseline": baseline, "scored": scored, "deltas": deltas, "gate": gate}, indent=2))


if __name__ == "__main__": run()
