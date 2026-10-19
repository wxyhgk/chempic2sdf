#!/usr/bin/env python3
"""Attribute r17 compile failures to elements, bond types, or aromatic flags."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
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
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from tools.train_element_head_cpu_r17 import ELEMENTS, ElementHead
from tools.train_element_context_head_cpu_r17 import ElementContextHead
from tools.train_graph_context_aromatic_cpu_r17 import AromaticContextHead
from tools.train_type_residual_cpu_r17 import TypeContextHead
from tools.audit_graph_context_aromatic_compile_r17 import cycle_edges

RUN = ROOT / "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m16-t0p72-c1-cache3"
CACHE = ROOT / os.environ.get("R17_ATTRIBUTION_CACHE", "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m16-t0p72-c1-cache3/feature-cache.pt")
AROMATIC_STATE = ROOT / os.environ.get("R17_ATTRIBUTION_AROMATIC_STATE", "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m16-t0p72-c1-cache3/sidecar-state.pt")
TYPE_STATE = ROOT / os.environ.get("R17_ATTRIBUTION_TYPE_STATE", "outputs/audits/chemvector-type-residual-cpu-r17-all-m16/sidecar-state.pt")
ELEMENT_STATE = ROOT / os.environ.get("R17_ATTRIBUTION_ELEMENT_STATE", "outputs/audits/chemvector-element-head-cpu-r17-all-m16/head-state.pt")
OUT = ROOT / os.environ.get("R17_ATTRIBUTION_OUT", "outputs/audits/chemvector-r17-compile-attribution-all-m16-v2")
ELEMENT_CONTEXT_STATE = ROOT / os.environ.get("R17_ELEMENT_CONTEXT_STATE", "outputs/audits/chemvector-element-context-head-cpu-r17-all-m16/head-state.pt")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compile_variant(record, selected, elements, types, aromatic, *, nodes):
    local = {node: i for i, node in enumerate(nodes)}
    node_payload = [(node, elements[i], node in aromatic) for i, node in enumerate(nodes)]
    edge_payload = {}
    for index, (left, right, _) in enumerate(record["pairs"]):
        kind = int(types[index]) + 1
        if kind == 4:
            kind = 1
        if (left, right) in selected:
            kind = 4
        edge_payload[(local[left], local[right])] = kind
    return sanitize(node_payload, edge_payload)


def context_features(record, nodes):
    by_node = {node: [] for node in nodes}
    for index, (left, right, _) in enumerate(record["pairs"]):
        by_node[left].append(record["type_features"][index])
        by_node[right].append(record["type_features"][index])
    return torch.stack([torch.cat((record["atom_features"][node].float(), torch.stack(by_node[node]).mean(0), torch.stack(by_node[node]).max(0).values, torch.tensor([float(len(by_node[node])) / 8.0]))) for node in nodes])


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    initial = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
    if digest(initial) != SHA:
        raise RuntimeError("production checkpoint changed")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    aromatic_state = torch.load(AROMATIC_STATE, map_location="cpu", weights_only=False)
    type_state = torch.load(TYPE_STATE, map_location="cpu", weights_only=False)
    element_state = torch.load(ELEMENT_STATE, map_location="cpu", weights_only=False)
    element_context_state = torch.load(ELEMENT_CONTEXT_STATE, map_location="cpu", weights_only=False)
    if cache.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("cache checkpoint binding mismatch")
    aromatic = AromaticContextHead(int(aromatic_state["state_dict"]["net.0.weight"].shape[0])); aromatic.load_state_dict(aromatic_state["state_dict"]); aromatic.eval()
    type_head = TypeContextHead(262); type_head.load_state_dict(type_state["state_dict"]); type_head.eval()
    element_head = ElementHead(); element_head.load_state_dict(element_state["state_dict"]); element_head.eval()
    element_context_head = ElementContextHead(); element_context_head.load_state_dict(element_context_state["state_dict"]); element_context_head.eval()
    totals = Counter(); profiles = Counter()
    with torch.inference_mode():
        for split, records in cache["records"].items():
            for record in records:
                if not len(record["pairs"]):
                    continue
                edges = [tuple(sorted((int(a), int(b)))) for a, b, _ in record["pairs"]]
                probabilities = aromatic(record["features"]).sigmoid().tolist()
                raw = {edge for edge, p in zip(edges, probabilities) if p >= 0.5}
                selected = cycle_edges(record["pairs"], raw)
                nodes = sorted({node for edge in edges for node in edge})
                predicted_elements = [ELEMENTS[i] for i in element_head(record["atom_features"][nodes]).argmax(-1).tolist()]
                context_elements = [ELEMENTS[i] for i in element_context_head(context_features(record, nodes)).argmax(-1).tolist()]
                predicted_types = type_head(record["type_features"]).argmax(-1).tolist()
                oracle_elements = [record["target_elements"].get(int(record["mapping"].get(node, -1)), "C") for node in nodes]
                oracle_types = [int(value) for value in record["type_labels"].tolist()]
                target_aromatic = {edge for edge, label in zip(edges, record["type_labels"].tolist()) if int(label) == 3}
                variants = {
                    "candidate": (predicted_elements, predicted_types, {node for edge in selected for node in edge}),
                    "candidate_context_elements": (context_elements, predicted_types, {node for edge in selected for node in edge}),
                    "oracle_elements": (oracle_elements, predicted_types, {node for edge in selected for node in edge}),
                    "oracle_types": (predicted_elements, oracle_types, {node for edge in selected for node in edge}),
                    "oracle_aromatic": (predicted_elements, predicted_types, {node for edge in target_aromatic for node in edge}),
                    "oracle_all": (oracle_elements, oracle_types, {node for edge in target_aromatic for node in edge}),
                }
                profile_id = record["sample_key"].rsplit("__", 1)[-1]
                for name, (elements, types, aromatic_nodes) in variants.items():
                    ok, error = compile_variant(record, selected, elements, types, aromatic_nodes, nodes=nodes)
                    totals[(split, name, "images")] += 1
                    totals[(split, name, "sanitize_ok")] += int(ok)
                    totals[(split, name, "error_" + error)] += int(not ok)
                    profiles[(split, profile_id, name, "images")] += 1
                    profiles[(split, profile_id, name, "sanitize_ok")] += int(ok)
    summary = {}
    for split in cache["records"]:
        summary[split] = {}
        for name in ("candidate", "candidate_context_elements", "oracle_elements", "oracle_types", "oracle_aromatic", "oracle_all"):
            n = totals[(split, name, "images")]
            summary[split][name] = {
                "images": n,
                "sanitize_rate": totals[(split, name, "sanitize_ok")] / max(n, 1),
                "compile_errors": {key[2][len("error_"):]: value for key, value in totals.items() if key[:2] == (split, name) and key[2].startswith("error_") and value},
            }
    profile_summary = {}
    profile_ids = sorted({key[1] for key in profiles})
    for split in cache["records"]:
        profile_summary[split] = {}
        for profile_id in profile_ids:
            profile_summary[split][profile_id] = {}
            for name in ("candidate", "candidate_context_elements", "oracle_elements", "oracle_types", "oracle_aromatic", "oracle_all"):
                n = profiles[(split, profile_id, name, "images")]
                profile_summary[split][profile_id][name] = {
                    "images": n,
                    "sanitize_rate": profiles[(split, profile_id, name, "sanitize_ok")] / max(n, 1),
                }
    report = {
        "schema_version": "chemvector-r17-compile-attribution-0.1", "complete": True,
        "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "cache": str(CACHE.relative_to(ROOT)), "aromatic_state_sha256": digest(AROMATIC_STATE),
        "type_state_sha256": digest(TYPE_STATE), "element_state_sha256": digest(ELEMENT_STATE),
        "element_context_state_sha256": digest(ELEMENT_CONTEXT_STATE),
        "summary": summary, "profile_summary": profile_summary,
        "policy": {"target_free_selection": True, "oracle_overlays_evaluation_only": True, "production_integration": "not_authorized"},
        "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": digest(initial),
        "gate": {"checkpoint_unchanged": True, "sealed_test_read": False, "production_go": False,
                 "next": "use attribution to choose one bounded CPU repair; do not start GPU"},
    }
    OUT.mkdir(parents=True, exist_ok=False); (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "summary": summary, "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
