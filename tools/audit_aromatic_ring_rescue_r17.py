#!/usr/bin/env python3
"""Target-free complete-ring transaction audit for the r17 aromatic sidecar.

The frozen AtomPair candidate set is not expanded.  We enumerate only complete
5/6-cycles already present in that set, require a fixed confidence pattern
(at least one edge may be below the ordinary aromatic threshold), and add a
cycle as one rollback-able transaction.  Targets are used only after the
selection for metrics; production and sealed-test data are untouched.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import networkx as nx
import torch
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_direct_graph_pair_scorer_chemistry_r9 import sanitize
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from tools.train_element_head_cpu_r17 import ELEMENTS, ElementHead
from tools.train_graph_context_aromatic_cpu_r17 import AromaticContextHead
from tools.train_type_residual_cpu_r17 import TypeContextHead

RUN = ROOT / "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m16-t0p72-c1-cache3"
CACHE = RUN / "feature-cache.pt"
AROMATIC_STATE = RUN / "sidecar-state.pt"
TYPE_STATE = ROOT / "outputs/audits/chemvector-type-residual-cpu-r17-all-m16/sidecar-state.pt"
ELEMENT_STATE = ROOT / "outputs/audits/chemvector-element-head-cpu-r17-all-m16/head-state.pt"
OUT = ROOT / "outputs/audits/chemvector-aromatic-ring-rescue-r17-all-m16"
AROMATIC_THRESHOLD = 0.50
RING_MEAN_THRESHOLD = 0.85
RING_MAX_MISSING = 1


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def edge_set(record: dict) -> list[tuple[int, int]]:
    return [tuple(sorted((int(a), int(b)))) for a, b, _ in record["pairs"]]


def simple_rings(edges: list[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
    """Return unique 5/6-cycle edge sets from the frozen candidate graph."""
    graph = nx.Graph(); graph.add_edges_from(edges)
    found: set[frozenset[tuple[int, int]]] = set()
    for cycle in nx.simple_cycles(graph.to_directed()):
        if len(cycle) not in (5, 6):
            continue
        ring = frozenset(tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)]))) for i in range(len(cycle)))
        if len(ring) == len(cycle):
            found.add(ring)
    return sorted(found, key=lambda ring: (len(ring), tuple(sorted(ring))))


def transaction_groups(rings: list[frozenset[tuple[int, int]]]) -> list[frozenset[tuple[int, int]]]:
    """Group overlapping rings so fused systems commit/rollback atomically."""
    graph = nx.Graph(); graph.add_nodes_from(range(len(rings)))
    nodes = [{node for edge in ring for node in edge} for ring in rings]
    for i in range(len(rings)):
        for j in range(i):
            if nodes[i] & nodes[j]:
                graph.add_edge(i, j)
    groups = []
    for component in nx.connected_components(graph):
        groups.append(frozenset(edge for i in component for edge in rings[i]))
    return sorted(groups, key=lambda group: (len(group), tuple(sorted(group))))


def cycle_edges(edges: list[tuple[int, int]], selected: set[tuple[int, int]]) -> set[tuple[int, int]]:
    adjacency: dict[int, set[int]] = {}
    for left, right in edges:
        if (left, right) not in selected:
            continue
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    accepted = set()
    for left, right in selected:
        seen = {left}; stack = [left]
        while stack:
            node = stack.pop()
            for nxt in adjacency.get(node, ()):
                if tuple(sorted((node, nxt))) == (left, right):
                    continue
                if nxt == right:
                    accepted.add((left, right)); stack.clear(); break
                if nxt not in seen:
                    seen.add(nxt); stack.append(nxt)
    return accepted


def compile_graph(record: dict, selected: set[tuple[int, int]], element_head: ElementHead, type_head: TypeContextHead, *, oracle: bool = False):
    edges = edge_set(record)
    nodes = sorted({node for edge in edges for node in edge})
    local = {node: i for i, node in enumerate(nodes)}
    with torch.inference_mode():
        if oracle:
            elements = [record["target_elements"].get(int(record["mapping"].get(node, -1)), "C") for node in nodes]
        else:
            elements = [ELEMENTS[i] for i in element_head(record["atom_features"][nodes]).argmax(-1).tolist()]
        typed = type_head(record["type_features"]).argmax(-1).tolist()
    if oracle:
        aromatic_nodes = {node for node in nodes if int(record["mapping"].get(node, -1)) in record["target_aromatic_atoms"]}
    else:
        aromatic_nodes = {node for edge in selected for node in edge}
    nodes_payload = [(node, elements[i], node in aromatic_nodes) for i, node in enumerate(nodes)]
    predicted_edges = {}
    for index, (left, right) in enumerate(edges):
        kind = int(typed[index]) + 1
        if kind == 4:
            kind = 1
        if (left, right) in selected:
            kind = 4
        predicted_edges[(local[left], local[right])] = kind
    return sanitize(nodes_payload, predicted_edges)


def select_transactions(record: dict, aromatic_head: AromaticContextHead) -> tuple[set[tuple[int, int]], set[tuple[int, int]], dict]:
    edges = edge_set(record)
    with torch.inference_mode():
        probabilities = aromatic_head(record["features"]).sigmoid().tolist()
    scores = dict(zip(edges, probabilities))
    raw = {edge for edge, score in scores.items() if score >= AROMATIC_THRESHOLD}
    strict = cycle_edges(edges, raw)
    proposed = []
    for ring in simple_rings(edges):
        values = [scores[edge] for edge in ring]
        high = sum(value >= AROMATIC_THRESHOLD for value in values)
        if len(ring) - high <= RING_MAX_MISSING and sum(values) / len(values) >= RING_MEAN_THRESHOLD:
            if not ring.issubset(strict):
                proposed.append(ring)
    current = set(strict)
    accepted = 0
    rolled_back = 0
    for group in transaction_groups(proposed):
        candidate = current | set(group)
        # Selection is target-free.  The compiler is used only as a safety
        # check; failed transactions are explicitly abstained/rolled back.
        ok, _ = compile_graph(record, candidate, _ELEMENT_HEAD, _TYPE_HEAD)
        if ok:
            current = candidate; accepted += len(group)
        else:
            rolled_back += len(group)
    return raw, current, {"strict_edges": len(strict), "proposed_rings": len(proposed), "accepted_ring_edges": accepted, "rollback_or_abstain_edges": rolled_back}


def run() -> dict:
    global _ELEMENT_HEAD, _TYPE_HEAD
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    initial = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
    if digest(initial) != SHA:
        raise RuntimeError("production checkpoint changed")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    aromatic_state = torch.load(AROMATIC_STATE, map_location="cpu", weights_only=False)
    type_state = torch.load(TYPE_STATE, map_location="cpu", weights_only=False)
    element_state = torch.load(ELEMENT_STATE, map_location="cpu", weights_only=False)
    if cache.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("cache checkpoint binding mismatch")
    aromatic = AromaticContextHead(int(aromatic_state["state_dict"]["net.0.weight"].shape[0])); aromatic.load_state_dict(aromatic_state["state_dict"]); aromatic.eval()
    _TYPE_HEAD = TypeContextHead(262); _TYPE_HEAD.load_state_dict(type_state["state_dict"]); _TYPE_HEAD.eval()
    _ELEMENT_HEAD = ElementHead(); _ELEMENT_HEAD.load_state_dict(element_state["state_dict"]); _ELEMENT_HEAD.eval()
    totals = Counter(); profile = Counter(); pairs = []; transaction_journal = []
    with torch.inference_mode():
        for split, records in cache["records"].items():
            for record in records:
                if not len(record["pairs"]):
                    continue
                raw, rescued, journal = select_transactions(record, aromatic)
                strict = set()
                for edge in raw:
                    # Recompute strict only for reporting; select_transactions
                    # keeps this internal so no target can enter the route.
                    strict = cycle_edges(edge_set(record), raw)
                for arm, selected in (("strict", strict), ("ring_rescue", rescued)):
                    ok, err = compile_graph(record, selected, _ELEMENT_HEAD, _TYPE_HEAD)
                    oracle_ok, oracle_err = compile_graph(record, selected, _ELEMENT_HEAD, _TYPE_HEAD, oracle=True)
                    truth = {edge for edge, label in zip(edge_set(record), record["type_labels"].tolist()) if int(label) == 3}
                    totals[(split, arm, "images")] += 1
                    totals[(split, arm, "sanitize_ok")] += int(ok)
                    totals[(split, arm, "oracle_sanitize_ok")] += int(oracle_ok)
                    totals[(split, arm, "compile_error_" + err)] += int(not ok)
                    totals[(split, arm, "oracle_compile_error_" + oracle_err)] += int(not oracle_ok)
                    totals[(split, arm, "tp")] += len(selected & truth)
                    totals[(split, arm, "fp")] += len(selected - truth)
                    totals[(split, arm, "predicted_edges")] += len(selected)
                    totals[(split, arm, "target_edges")] += len(truth)
                    profile[(split, record["sample_key"].rsplit("__", 1)[-1], arm, "images")] += 1
                    profile[(split, record["sample_key"].rsplit("__", 1)[-1], arm, "sanitize_ok")] += int(ok)
                    profile[(split, record["sample_key"].rsplit("__", 1)[-1], arm, "tp")] += len(selected & truth)
                    profile[(split, record["sample_key"].rsplit("__", 1)[-1], arm, "fp")] += len(selected - truth)
                pairs.append({"split": split, "sample_key": record["sample_key"], **journal, "strict_edges": len(strict), "ring_rescue_edges": len(rescued)})
    summary = {}
    for split in cache["records"]:
        summary[split] = {}
        for arm in ("strict", "ring_rescue"):
            n = totals[(split, arm, "images")]; tp = totals[(split, arm, "tp")]; pred = totals[(split, arm, "predicted_edges")]; target = totals[(split, arm, "target_edges")]
            summary[split][arm] = {
                "images": n, "sanitize_rate": totals[(split, arm, "sanitize_ok")] / max(n, 1),
                "oracle_sanitize_rate": totals[(split, arm, "oracle_sanitize_ok")] / max(n, 1),
                "aromatic_precision": tp / max(pred, 1), "aromatic_recall": tp / max(target, 1),
                "predicted_aromatic_edges": pred, "target_aromatic_edges": target,
                "compile_errors": {key[2][len("compile_error_"):]: value for key, value in totals.items() if key[:2] == (split, arm) and key[2].startswith("compile_error_") and value},
                "oracle_compile_errors": {key[2][len("oracle_compile_error_"):]: value for key, value in totals.items() if key[:2] == (split, arm) and key[2].startswith("oracle_compile_error_") and value},
            }
    report = {
        "schema_version": "chemvector-aromatic-ring-rescue-r17-0.1", "complete": True,
        "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "cache": str(CACHE.relative_to(ROOT)), "aromatic_state_sha256": digest(AROMATIC_STATE),
        "type_state_sha256": digest(TYPE_STATE), "element_state_sha256": digest(ELEMENT_STATE),
        "thresholds": {"aromatic": AROMATIC_THRESHOLD, "ring_mean": RING_MEAN_THRESHOLD, "ring_max_missing": RING_MAX_MISSING},
        "summary": summary, "paired": pairs,
        "policy": {"target_free_selection": True, "candidate_edge_set_frozen": True, "transaction": "complete 5/6-cycle union with compiler rollback/abstain", "oracle_overlay_used": False, "production_integration": "not_authorized"},
        "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": digest(initial),
        "gate": {"checkpoint_unchanged": True, "sealed_test_read": False, "target_free_selection": True, "production_go": False,
                 "next": "only consider a larger CPU replay if fresh ring-rescue precision and compile gates pass; do not start GPU"},
    }
    OUT.mkdir(parents=True, exist_ok=False); (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "summary": summary, "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
