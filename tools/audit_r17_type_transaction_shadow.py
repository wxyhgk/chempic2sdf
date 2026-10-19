#!/usr/bin/env python3
"""Target-free bond-type transaction shadow on the frozen r17 candidate graph.

Unlike rollback, this probe never deletes or adds an edge.  For invalid
candidate graphs it enumerates bounded one- and two-edge SINGLE/DOUBLE/TRIPLE
type edits on low-confidence non-aromatic edges, accepts only a compiler-valid
transaction, and joins source labels afterwards for identity/type accounting.
It is CPU-only and never writes the production checkpoint.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations, product
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_direct_graph_pair_scorer_chemistry_r9 import sanitize
from tools.audit_graph_context_aromatic_compile_r17 import cycle_edges
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from tools.train_element_context_head_cpu_r17 import ElementContextHead, ELEMENTS
from tools.train_graph_context_aromatic_cpu_r17 import AromaticContextHead
from tools.train_type_residual_cpu_r17 import TypeContextHead

CACHE = ROOT / os.environ.get(
    "R17_TYPE_TX_CACHE",
    "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m128-t0p72-c1-panel128/feature-cache.pt",
)
AROMATIC_STATE = ROOT / os.environ.get("R17_TYPE_TX_AROMATIC_STATE", str(CACHE.parent / "sidecar-state.pt"))
TYPE_STATE = ROOT / os.environ.get(
    "R17_TYPE_TX_TYPE_STATE", "outputs/audits/chemvector-type-residual-cpu-r17-all-m128-panel128/sidecar-state.pt"
)
ELEMENT_STATE = ROOT / os.environ.get(
    "R17_TYPE_TX_ELEMENT_STATE", "outputs/audits/chemvector-element-context-head-cpu-r17-all-m128-panel128/head-state.pt"
)
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/samples"
OUT = ROOT / os.environ.get("R17_TYPE_TX_OUT", "outputs/audits/chemvector-r17-type-transaction-m128-v1")
MAX_CANDIDATES = int(os.environ.get("R17_TYPE_TX_MAX_CANDIDATES", "6"))
MAX_ATTEMPTS = int(os.environ.get("R17_TYPE_TX_MAX_ATTEMPTS", "96"))
MAX_EDITS = int(os.environ.get("R17_TYPE_TX_MAX_EDITS", "2"))
TYPE_NAMES = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 4: "AROMATIC"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def context_features(record: dict, nodes: list[int]) -> torch.Tensor:
    by_node = {node: [] for node in nodes}
    for index, (left, right, _) in enumerate(record["pairs"]):
        by_node[left].append(record["type_features"][index])
        by_node[right].append(record["type_features"][index])
    rows = []
    for node in nodes:
        values = by_node[node]
        rows.append(torch.cat((record["atom_features"][node].float(),
                               torch.stack(values).mean(0),
                               torch.stack(values).max(0).values,
                               torch.tensor([float(len(values)) / 8.0]))))
    return torch.stack(rows)


def source_graph(sample_key: str) -> tuple[set[tuple[int, int]], dict[tuple[int, int], str]]:
    graph = json.loads((SOURCE / f"{sample_key}.graph.json").read_text())
    types = {}
    for bond in graph["bonds"]:
        edge = tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        types[edge] = str(bond.get("chemical_type", "SINGLE")).upper()
    return set(types), types


def compile_graph(nodes, edge_map, elements, aromatic_nodes):
    payload_nodes = [(node, elements[i], node in aromatic_nodes) for i, node in enumerate(nodes)]
    return sanitize(payload_nodes, edge_map)


def mapped_edge_types(record, nodes, edge_map):
    mapping = record.get("mapping", {})
    result = {}
    for (left, right), kind in edge_map.items():
        a, b = mapping.get(nodes[left]), mapping.get(nodes[right])
        if a is not None and b is not None and int(a) != int(b):
            result[tuple(sorted((int(a), int(b))))] = TYPE_NAMES[int(kind)]
    return result


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    initial = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
    if digest(initial) != SHA:
        raise RuntimeError("production checkpoint changed")
    for path in (CACHE, AROMATIC_STATE, TYPE_STATE, ELEMENT_STATE):
        if not path.exists():
            raise FileNotFoundError(path)
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    if cache.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("cache checkpoint binding mismatch")
    aromatic_state = torch.load(AROMATIC_STATE, map_location="cpu", weights_only=False)
    type_state = torch.load(TYPE_STATE, map_location="cpu", weights_only=False)
    element_state = torch.load(ELEMENT_STATE, map_location="cpu", weights_only=False)
    aromatic = AromaticContextHead(526); aromatic.load_state_dict(aromatic_state["state_dict"]); aromatic.eval()
    type_head = TypeContextHead(262); type_head.load_state_dict(type_state["state_dict"]); type_head.eval()
    element_head = ElementContextHead(); element_head.load_state_dict(element_state["state_dict"]); element_head.eval()

    rows = []
    with torch.inference_mode():
        for split, records in cache["records"].items():
            for record in records:
                if not record["pairs"]:
                    continue
                nodes = sorted({int(node) for left, right, _ in record["pairs"] for node in (left, right)})
                local = {node: i for i, node in enumerate(nodes)}
                aromatic_prob = aromatic(record["features"]).sigmoid()
                raw = {tuple(sorted((int(a), int(b)))) for (a, b, _), p in zip(record["pairs"], aromatic_prob.tolist()) if p >= 0.5}
                selected = cycle_edges(record["pairs"], raw)
                element_logits = element_head(context_features(record, nodes))
                elements = [ELEMENTS[i] for i in element_logits.argmax(-1).tolist()]
                element_conf = element_logits.softmax(-1).max(-1).values.tolist()
                type_logits = type_head(record["type_features"])
                type_prob = type_logits.softmax(-1)
                typed = type_logits.argmax(-1).tolist()
                edge_map = {}
                edge_info = {}
                for i, (left, right, _) in enumerate(record["pairs"]):
                    edge = tuple(sorted((local[left], local[right])))
                    kind = int(typed[i]) + 1
                    if kind == 4:
                        kind = 1
                    is_aromatic = (left, right) in selected
                    if is_aromatic:
                        kind = 4
                    edge_map[edge] = kind
                    endpoint = min(element_conf[nodes.index(left)], element_conf[nodes.index(right)])
                    edge_info[edge] = {
                        "pair_index": i,
                        "type_probability": type_prob[i].tolist(),
                        "confidence": float(endpoint * float(type_prob[i].max())),
                        "aromatic": is_aromatic,
                    }
                baseline = dict(edge_map)
                baseline_ok, baseline_error = compile_graph(nodes, baseline, elements, {u for e in selected for u in e})
                current = dict(baseline)
                attempts = 0
                accepted = []
                if not baseline_ok:
                    candidates = sorted((e for e, info in edge_info.items() if not info["aromatic"]),
                                        key=lambda e: (edge_info[e]["confidence"], e))[:MAX_CANDIDATES]
                    alternatives = {}
                    for edge in candidates:
                        probs = edge_info[edge]["type_probability"]
                        current_kind = current[edge]
                        ranked = sorted((kind + 1 for kind in range(3) if kind + 1 != current_kind),
                                        key=lambda kind: (-probs[kind - 1], kind))
                        alternatives[edge] = ranked
                    proposals = []
                    for size in range(1, MAX_EDITS + 1):
                        for edges in combinations(candidates, size):
                            for kinds in product(*(alternatives[e] for e in edges)):
                                score = sum(math.log(max(edge_info[e]["type_probability"][kind - 1], 1e-8)) for e, kind in zip(edges, kinds))
                                proposals.append((score, edges, kinds))
                    proposals.sort(key=lambda item: (-item[0], item[1], item[2]))
                    for score, edges, kinds in proposals[:MAX_ATTEMPTS]:
                        attempts += 1
                        trial = dict(baseline)
                        for edge, kind in zip(edges, kinds):
                            trial[edge] = kind
                        trial_ok, trial_error = compile_graph(nodes, trial, elements, {u for e in selected for u in e})
                        if trial_ok:
                            current = trial
                            accepted = [{"edge": list(edge), "from": int(baseline[edge]), "to": int(kind)} for edge, kind in zip(edges, kinds)]
                            break
                final_ok, final_error = compile_graph(nodes, current, elements, {u for e in selected for u in e})
                truth_edges, truth_types = source_graph(record["sample_key"])
                baseline_types = mapped_edge_types(record, nodes, baseline)
                final_types = mapped_edge_types(record, nodes, current)
                rows.append({
                    "split": split, "sample_key": record["sample_key"],
                    "baseline_sanitize": bool(baseline_ok), "transaction_sanitize": bool(final_ok),
                    "baseline_error": baseline_error, "transaction_error": final_error,
                    "attempts": attempts, "accepted": accepted,
                    "baseline_edges": sorted([list(e) for e in baseline_types]),
                    "transaction_edges": sorted([list(e) for e in final_types]),
                    "target_edges": sorted([list(e) for e in truth_edges]),
                    "baseline_types": {f"{k[0]},{k[1]}": v for k, v in baseline_types.items()},
                    "transaction_types": {f"{k[0]},{k[1]}": v for k, v in final_types.items()},
                    "target_types": {f"{k[0]},{k[1]}": v for k, v in truth_types.items()},
                })
    summary = {}
    for split in ("train", "dev", "fresh_holdout"):
        subset = [r for r in rows if r["split"] == split]
        base_edges = [set(tuple(e) for e in r["baseline_edges"]) for r in subset]
        tx_edges = [set(tuple(e) for e in r["transaction_edges"]) for r in subset]
        truth_edges = [set(tuple(e) for e in r["target_edges"]) for r in subset]
        base_type_correct = tx_type_correct = 0; type_total = 0
        for r in subset:
            target = {tuple(int(x) for x in k.split(",")): v for k, v in r["target_types"].items()}
            for edge, truth in target.items():
                key = f"{edge[0]},{edge[1]}"
                if key in r["baseline_types"]:
                    type_total += 1
                    base_type_correct += int(r["baseline_types"][key] == truth)
                    tx_type_correct += int(r["transaction_types"].get(key) == truth)
        summary[split] = {
            "images": len(subset),
            "baseline_sanitize_rate": sum(r["baseline_sanitize"] for r in subset) / max(len(subset), 1),
            "transaction_sanitize_rate": sum(r["transaction_sanitize"] for r in subset) / max(len(subset), 1),
            "accepted_transactions": sum(bool(r["accepted"]) for r in subset),
            "accepted_edge_edits": sum(len(r["accepted"]) for r in subset),
            "edge_set_unchanged": all(a == b for a, b in zip(base_edges, tx_edges)),
            "baseline_edge_recall": sum(len(a & t) for a, t in zip(base_edges, truth_edges)) / max(sum(len(t) for t in truth_edges), 1),
            "transaction_edge_recall": sum(len(a & t) for a, t in zip(tx_edges, truth_edges)) / max(sum(len(t) for t in truth_edges), 1),
            "baseline_type_accuracy": base_type_correct / max(type_total, 1),
            "transaction_type_accuracy": tx_type_correct / max(type_total, 1),
            "type_rows": type_total,
            "abstain_rate": sum((not r["baseline_sanitize"]) and not r["accepted"] for r in subset) / max(len(subset), 1),
        }
    report = {
        "schema_version": "chemvector-r17-type-transaction-shadow-0.1", "complete": True,
        "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "cache": str(CACHE.relative_to(ROOT)), "rows": len(rows), "max_candidates": MAX_CANDIDATES,
        "max_attempts": MAX_ATTEMPTS, "max_edits": MAX_EDITS, "summary": summary, "rows_detail": rows,
        "policy": {"target_free_selection": True, "edge_set_frozen": True, "aromatic_edges_frozen": True,
                   "evaluation_target_join_posthoc": True, "production_integration": "not_authorized"},
        "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": digest(initial),
        "gate": {"checkpoint_unchanged": digest(initial) == SHA, "sealed_test_read": False,
                 "edge_set_unchanged": all(set(tuple(e) for e in r["baseline_edges"]) == set(tuple(e) for e in r["transaction_edges"]) for r in rows),
                 "aromatic_edges_untouched": True,
                 "fresh_sanitize_gain_ge_0.03": summary["fresh_holdout"]["transaction_sanitize_rate"] - summary["fresh_holdout"]["baseline_sanitize_rate"] >= 0.03,
                 "fresh_edge_recall_no_drop": summary["fresh_holdout"]["transaction_edge_recall"] >= summary["fresh_holdout"]["baseline_edge_recall"],
                 "production_go": False},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "summary": summary, "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
