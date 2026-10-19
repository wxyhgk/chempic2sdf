#!/usr/bin/env python3
"""Read-only compile/transaction audit for the r17 graph-context sidecar."""
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
from tools.train_element_head_cpu_r17 import ELEMENTS, ElementHead
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from tools.train_graph_context_aromatic_cpu_r17 import AromaticContextHead
from tools.train_type_residual_cpu_r17 import TypeContextHead

RUN = ROOT / "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m16-t0p72-c1-cache3"
CACHE = RUN / "feature-cache.pt"
STATE = RUN / "sidecar-state.pt"
OUT = ROOT / "outputs/audits/chemvector-graph-context-aromatic-compile-r17-all-m16-v5"
TYPE_STATE = ROOT / "outputs/audits/chemvector-type-residual-cpu-r17-all-m16/sidecar-state.pt"
THRESHOLD = 0.5


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_element_head():
    path = ROOT / "outputs/audits/chemvector-element-head-cpu-r17-all-m16/head-state.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    head = ElementHead()
    head.load_state_dict(payload["state_dict"]); head.eval()
    return head, digest(path)


def cycle_edges(edges: list[tuple[int, int, int]], selected: set[tuple[int, int]]) -> set[tuple[int, int]]:
    adjacency: dict[int, set[int]] = {}
    for a, b, _ in edges:
        if (a, b) not in selected:
            continue
        adjacency.setdefault(a, set()).add(b); adjacency.setdefault(b, set()).add(a)
    accepted = set()
    for edge in selected:
        left, right = edge
        seen = {left}; stack = [left]
        while stack:
            node = stack.pop()
            for nxt in adjacency.get(node, ()):
                if tuple(sorted((node, nxt))) == edge:
                    continue
                if nxt == right:
                    accepted.add(edge); stack.clear(); break
                if nxt not in seen:
                    seen.add(nxt); stack.append(nxt)
    return accepted


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if digest(ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt") != SHA:
        raise RuntimeError("production checkpoint changed")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    state = torch.load(STATE, map_location="cpu", weights_only=False)
    if cache.get("initial_checkpoint_sha256") != SHA or state.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("sidecar binding mismatch")
    input_dim = int(state["state_dict"]["net.0.weight"].shape[0])
    head = AromaticContextHead(input_dim); head.load_state_dict(state["state_dict"]); head.eval()
    type_state = torch.load(TYPE_STATE, map_location="cpu", weights_only=False)
    type_head = TypeContextHead(262); type_head.load_state_dict(type_state["state_dict"]); type_head.eval()
    element_head, element_sha = load_element_head()
    totals = Counter(); profiles = Counter(); paired = []
    type_confusion = Counter()
    with torch.inference_mode():
        for split, records in cache["records"].items():
            for record in records:
                features = record["features"]
                if not len(features):
                    continue
                probabilities = head(features).sigmoid()
                raw = {tuple(sorted((a, b))) for (a, b, _), p in zip(record["pairs"], probabilities.tolist()) if p >= THRESHOLD}
                strict = cycle_edges(record["pairs"], raw)
                # The transaction audit uses the raw sidecar set and the strict
                # cycle transaction as separate arms.  Neither arm sees GT.
                for arm, selected in (("raw", raw), ("cycle_transaction", strict)):
                    node_ids = sorted({node for edge in record["pairs"] for node in edge[:2]})
                    local = {node: index for index, node in enumerate(node_ids)}
                    predicted_elements = element_head(record["atom_features"][node_ids]).argmax(-1).tolist() if node_ids else []
                    aromatic_nodes = {node for edge in selected for node in edge}
                    predicted_nodes = [(node, ELEMENTS[predicted_elements[i]], node in aromatic_nodes) for i, node in enumerate(node_ids)]
                    predicted_edges = {}
                    typed = type_head(record["type_features"]).argmax(-1).tolist() if len(record["type_features"]) else []
                    for edge_index, (a, b, _) in enumerate(record["pairs"]):
                        kind = int(typed[edge_index]) + 1
                        if kind == 4:
                            kind = 1
                        if (a, b) in selected:
                            kind = 4
                        predicted_edges[(local[a], local[b])] = kind
                        # Evaluation-only type confusion.  This never affects
                        # routing or selection and is joined after prediction.
                        truth_type = int(record["type_labels"][edge_index]) if edge_index < len(record["type_labels"]) else 0
                        type_confusion[(split, arm, truth_type, kind - 1)] += 1
                    ok, error = sanitize(predicted_nodes, predicted_edges)
                    # Oracle overlay is evaluation-only and cannot influence
                    # either arm's selected edge set or element predictions.
                    oracle_nodes = []
                    for i, node in enumerate(node_ids):
                        target = record["mapping"].get(node)
                        element = record["target_elements"].get(int(target), "C") if target is not None else "C"
                        aromatic = int(target) in record["target_aromatic_atoms"] if target is not None else False
                        oracle_nodes.append((node, element, aromatic))
                    oracle_ok, oracle_error = sanitize(oracle_nodes, predicted_edges)
                    totals[(split, arm, "oracle_sanitize_ok")] += int(oracle_ok)
                    totals[(split, arm, "oracle_compile_errors_" + oracle_error)] += int(not oracle_ok)
                    totals[(split, arm, "sanitize_ok")] += int(ok)
                    totals[(split, arm, "images")] += 1
                    totals[(split, arm, "aromatic_edges")] += len(selected)
                    # Strict transaction drops unsupported cycle edges.  Count
                    # those as rollback/abstain decisions; raw has none.
                    totals[(split, arm, "rollback_or_abstain_edges")] += len(raw - strict) if arm == "cycle_transaction" else 0
                    totals[(split, arm, "compile_errors_" + error)] += int(not ok)
                    profiles[(split, arm, "raw_candidates")] += len(raw)
                paired.append({"split": split, "sample_key": record["sample_key"], "raw_edges": len(raw), "cycle_edges": len(strict), "cycle_removed": len(raw - strict)})
    summary = {}
    for split in cache["records"]:
        summary[split] = {}
        for arm in ("raw", "cycle_transaction"):
            n = totals[(split, arm, "images")]
            summary[split][arm] = {
                "images": n,
                "sanitize_rate": totals[(split, arm, "sanitize_ok")] / max(n, 1),
                "oracle_sanitize_rate": totals[(split, arm, "oracle_sanitize_ok")] / max(n, 1),
                "aromatic_edges": totals[(split, arm, "aromatic_edges")],
                "rollback_or_abstain_edges": totals[(split, arm, "rollback_or_abstain_edges")],
                "compile_errors": {key[2][len("compile_errors_"):]: value for key, value in totals.items() if key[:2] == (split, arm) and key[2].startswith("compile_errors_") and value},
                "oracle_compile_errors": {key[2][len("oracle_compile_errors_"):]: value for key, value in totals.items() if key[:2] == (split, arm) and key[2].startswith("oracle_compile_errors_") and value},
                "type_confusion": [[type_confusion[(split, arm, truth, pred)] for pred in range(4)] for truth in range(4)],
            }
            summary[split][arm]["candidate_minus_oracle_sanitize_rate"] = summary[split][arm]["sanitize_rate"] - summary[split][arm]["oracle_sanitize_rate"]
    report = {
        "schema_version": "chemvector-graph-context-aromatic-compile-r17-0.1", "complete": True,
        "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "run": str(RUN.relative_to(ROOT)), "threshold": THRESHOLD, "element_head_sha256": element_sha,
        "summary": summary, "paired": paired,
        "policy": {"target_free_selection": True, "oracle_overlay_used": False, "cycle_transaction": "strict alternate-path only", "production_integration": "not_authorized"},
        "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": digest(ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"),
        "gate": {"checkpoint_unchanged": True, "target_free_selection": True, "sealed_test_read": False, "production_go": False, "next": "if compile gate remains low, run target-free complete-ring rescue with rollback/abstain; do not merge"},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "summary": summary, "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
