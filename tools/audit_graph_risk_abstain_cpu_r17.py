#!/usr/bin/env python3
"""Target-free graph-risk abstention replay for the r17 sidecars.

This is a bounded CPU audit, not a training or deployment path.  It evaluates
whether a graph-level risk signal (predicted valence overflow plus local
element/type/aromatic margins) can safely abstain before RDKit compilation.
Thresholds are selected on the training split only, using labels solely for
the post-hoc audit criterion; selection itself consumes no identity fields.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_direct_graph_pair_scorer_chemistry_r9 import sanitize
from tools.audit_graph_context_aromatic_compile_r17 import cycle_edges
from tools.train_element_context_head_cpu_r17 import ELEMENTS, ElementContextHead
from tools.train_graph_context_aromatic_cpu_r17 import AromaticContextHead
from tools.train_type_residual_cpu_r17 import TypeContextHead
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA

CACHE = ROOT / "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m64-t0p72-c1-element-context-m64/feature-cache.pt"
AROMATIC_STATE = ROOT / "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m64-t0p72-c1-element-context-m64/sidecar-state.pt"
TYPE_STATE = ROOT / "outputs/audits/chemvector-type-residual-cpu-r17-all-m64/sidecar-state.pt"
ELEMENT_STATE = ROOT / "outputs/audits/chemvector-element-context-head-cpu-r17-all-m64/head-state.pt"
OUT_DIR = ROOT / "outputs/audits/chemvector-graph-risk-abstain-cpu-r17-m64-v3"
REPORT = OUT_DIR / "report.json"

CAPS = {"C": 4.0, "N": 3.0, "O": 2.0, "S": 6.0, "P": 5.0,
        "F": 1.0, "Cl": 1.0, "Br": 1.0, "I": 1.0, "B": 3.0, "Si": 4.0}
ORDER = {0: 1.0, 1: 2.0, 2: 3.0, 3: 1.5}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def context_features(record: dict, nodes: list[int]) -> torch.Tensor:
    by_node = {node: [] for node in nodes}
    for i, (a, b, _) in enumerate(record["pairs"]):
        by_node[a].append(record["type_features"][i])
        by_node[b].append(record["type_features"][i])
    return torch.stack([
        torch.cat((record["atom_features"][node].float(),
                   torch.stack(by_node[node]).mean(0),
                   torch.stack(by_node[node]).max(0).values,
                   torch.tensor([float(len(by_node[node])) / 8.0])))
        for node in nodes
    ])


def one(record: dict, aromatic: torch.nn.Module, typer: torch.nn.Module,
        elementer: torch.nn.Module) -> dict:
    nodes = sorted({int(a) for a, b, _ in record["pairs"]} | {int(b) for a, b, _ in record["pairs"]})
    local = {n: i for i, n in enumerate(nodes)}
    with torch.inference_mode():
        arom_logits = aromatic(record["features"]).reshape(-1)
        type_logits = typer(record["type_features"])
        element_logits = elementer(context_features(record, nodes))
    raw = {tuple(sorted((int(a), int(b)))) for (a, b, _), p in zip(record["pairs"], arom_logits.sigmoid()) if float(p) >= 0.5}
    selected = cycle_edges(record["pairs"], raw)
    type_pred = type_logits.argmax(-1).tolist()
    element_pred = element_logits.argmax(-1).tolist()
    edges: dict[tuple[int, int], int] = {}
    usage = [0.0] * len(nodes)
    for i, (a, b, _) in enumerate(record["pairs"]):
        kind = int(type_pred[i])
        if kind == 3:
            # The compiler treats a predicted aromatic class as SINGLE unless
            # the target-free cycle transaction explicitly selected the edge.
            kind = 0
        if tuple(sorted((int(a), int(b)))) in selected:
            kind = 3
        edges[(local[int(a)], local[int(b)])] = kind + 1 if kind != 3 else 4
        usage[local[int(a)]] += ORDER[kind]
        usage[local[int(b)]] += ORDER[kind]
    elements = [ELEMENTS[i] for i in element_pred]
    node_payload = [(n, elements[i], n in {u for e in selected for u in e}) for i, n in enumerate(nodes)]
    ok, error = sanitize(node_payload, edges)
    overflow = [max(0.0, usage[i] - CAPS.get(elements[i], 4.0)) for i in range(len(nodes))]
    type_margin = (type_logits.topk(2, dim=-1).values[:, 0] - type_logits.topk(2, dim=-1).values[:, 1]) if len(type_logits) else torch.tensor([])
    element_margin = (element_logits.topk(2, dim=-1).values[:, 0] - element_logits.topk(2, dim=-1).values[:, 1]) if len(element_logits) else torch.tensor([])
    selected_idx = [i for i, (a, b, _) in enumerate(record["pairs"]) if tuple(sorted((int(a), int(b)))) in selected]
    aromatic_margin = arom_logits[selected_idx].abs() if selected_idx else torch.tensor([])
    return {"sanitize": bool(ok), "error": error, "accept": True,
            "max_overflow": max(overflow, default=0.0), "total_overflow": sum(overflow),
            "min_type_margin": float(type_margin.min()) if len(type_margin) else 99.0,
            "min_element_margin": float(element_margin.min()) if len(element_margin) else 99.0,
            "min_aromatic_margin": float(aromatic_margin.min()) if len(aromatic_margin) else 99.0,
            "elements": elements, "selected_aromatic_edges": len(selected),
            "raw_aromatic_edges": len(raw), "edges": edges}


def summarize(rows: list[dict], config: dict | None = None) -> dict:
    accepted = []
    for row in rows:
        if config is None or (row["max_overflow"] <= config["max_overflow"] and
                              row["min_type_margin"] >= config["min_type_margin"] and
                              row["min_element_margin"] >= config["min_element_margin"] and
                              row["min_aromatic_margin"] >= config["min_aromatic_margin"]):
            accepted.append(row)
    errors = Counter(r["error"] for r in accepted if not r["sanitize"])
    return {"images": len(rows), "accepted": len(accepted),
            "acceptance_rate": len(accepted) / max(len(rows), 1),
            "accepted_sanitize_rate": sum(r["sanitize"] for r in accepted) / max(len(accepted), 1),
            "accepted_valid_fraction": sum(r["sanitize"] for r in accepted) / max(len(rows), 1),
            "compile_errors": dict(errors)}


def run() -> dict:
    if OUT_DIR.exists():
        raise FileExistsError(OUT_DIR)
    initial = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
    if digest(initial) != SHA:
        raise RuntimeError("production checkpoint changed")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    if cache.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("cache checkpoint binding mismatch")
    astate = torch.load(AROMATIC_STATE, map_location="cpu", weights_only=False)
    tstate = torch.load(TYPE_STATE, map_location="cpu", weights_only=False)
    estate = torch.load(ELEMENT_STATE, map_location="cpu", weights_only=False)
    aromatic = AromaticContextHead(int(astate["state_dict"]["net.0.weight"].shape[0])); aromatic.load_state_dict(astate["state_dict"]); aromatic.eval()
    typer = TypeContextHead(262); typer.load_state_dict(tstate["state_dict"]); typer.eval()
    elementer = ElementContextHead(653); elementer.load_state_dict(estate["state_dict"]); elementer.eval()
    rows = {split: [one(r, aromatic, typer, elementer) for r in records if r.get("pairs")] for split, records in cache["records"].items()}
    grid = []
    for ov in (0.0, 0.25, 0.5, 1.0):
        for tm in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
            for em in (0.0, 0.05, 0.1, 0.2, 0.4):
                for am in (0.0, 0.05, 0.1, 0.2, 0.4):
                    cfg = {"max_overflow": ov, "min_type_margin": tm, "min_element_margin": em, "min_aromatic_margin": am}
                    s = summarize(rows["train"], cfg)
                    if s["accepted_sanitize_rate"] >= 0.95:
                        grid.append((s["accepted"], s["accepted_sanitize_rate"], cfg))
    chosen = max(grid, key=lambda x: (x[0], x[1])) if grid else (0, 0.0, {"max_overflow": 0.0, "min_type_margin": 0.0, "min_element_margin": 0.0, "min_aromatic_margin": 0.0})
    cfg = chosen[2]
    report = {"schema_version": "chemvector-graph-risk-abstain-cpu-r17-0.1", "complete": True,
              "read_only": True, "training_performed": False, "gpu_used": False,
              "sealed_test_read": False, "production_checkpoint_written": False,
              "cache": str(CACHE.relative_to(ROOT)), "rows": {s: len(v) for s, v in rows.items()},
              "baseline": {s: summarize(v) for s, v in rows.items()}, "selected_policy": cfg,
              "selected_policy_metrics": {s: summarize(v, cfg) for s, v in rows.items()},
              "train_policy_candidates_meeting_sanitize_0p95": len(grid),
              "policy": {"selection_split": "train", "target_free_selection": True,
                         "posthoc_labels_used_only_for_audit": True, "graph_edge_set_frozen": True},
              "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": digest(initial),
              "gate": {"checkpoint_unchanged": True, "sealed_test_read": False,
                       "dev_accepted_sanitize_ge_0.95": summarize(rows["dev"], cfg)["accepted_sanitize_rate"] >= 0.95,
                       "fresh_accepted_sanitize_ge_0.95": summarize(rows["fresh_holdout"], cfg)["accepted_sanitize_rate"] >= 0.95,
                       "fresh_acceptance_ge_0.50": summarize(rows["fresh_holdout"], cfg)["acceptance_rate"] >= 0.50,
                       "production_go": False,
                       "worth_training_graph_risk": False}}
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(exist_ok=False)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(REPORT), "baseline": report["baseline"], "selected_policy": cfg, "selected": report["selected_policy_metrics"], "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
