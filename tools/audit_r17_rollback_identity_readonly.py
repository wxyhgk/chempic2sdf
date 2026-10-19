#!/usr/bin/env python3
"""Read-only identity audit for the r17 target-free rollback journal.

This audit does not rerun inference and does not use labels for any decision.
It only joins the already-emitted rollback journal with the source graph after
the fact, checking whether a transaction removed a real edge and whether an
aromatic edge disappeared.  It is deliberately separate from the rollback
runner so that the decision journal remains immutable and inspectable.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / os.environ.get("R17_ROLLBACK_IDENTITY_REPORT", "outputs/audits/chemvector-r17-target-free-rollback-m64-v2/report.json")
CACHE = ROOT / os.environ.get("R17_ROLLBACK_IDENTITY_CACHE", "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m64-t0p72-c1-element-context-m64/feature-cache.pt")
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/samples"
OUT = ROOT / os.environ.get("R17_ROLLBACK_IDENTITY_OUT", "outputs/audits/chemvector-r17-rollback-identity-audit-m64-v1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def graph_payload(graph: dict) -> dict:
    atoms = []
    for atom in graph.get("atoms", []):
        atoms.append({
            "id": int(atom["atom_id"]),
            "element": str(atom.get("element") or "C"),
            "aromatic": bool(atom.get("aromatic", False)),
            "charge": int(atom.get("charge", 0) or 0),
        })
    bonds = []
    for bond in graph.get("bonds", []):
        bonds.append({
            "u": min(int(bond["begin_atom"]), int(bond["end_atom"])),
            "v": max(int(bond["begin_atom"]), int(bond["end_atom"])),
            "type": str(bond.get("chemical_type", "SINGLE")).upper(),
        })
    return {"atoms": sorted(atoms, key=lambda x: x["id"]),
            "bonds": sorted(bonds, key=lambda x: (x["u"], x["v"], x["type"]))}


def source_graph(sample_key: str) -> tuple[dict, set[tuple[int, int]], set[tuple[int, int]]]:
    graph = json.loads((SOURCE / f"{sample_key}.graph.json").read_text())
    edges = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))) for b in graph["bonds"]}
    aromatic = {
        tuple(sorted((int(b["begin_atom"]), int(b["end_atom"]))))
        for b in graph["bonds"]
        if str(b.get("chemical_type", "SINGLE")).upper() == "AROMATIC"
    }
    return graph, edges, aromatic


def edge_set(row: dict, field: str) -> set[tuple[int, int]]:
    return {tuple(sorted((int(x[0]), int(x[1])))) for x in row[field]}


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    report = json.loads(REPORT.read_text())
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    cache_rows = {r["sample_key"]: r for rows in cache["records"].values() for r in rows}

    rows = report["rows"]
    groups: dict[str, list[dict]] = defaultdict(list)
    source_join_errors: list[dict] = []
    transaction_counts = Counter()
    tx_details: list[dict] = []
    edge_set_errors: list[dict] = []
    aromatic_drops: list[dict] = []
    for row in rows:
        key = row["sample_key"]
        group = key.rsplit("__", 1)[0]
        groups[group].append(row)
        graph, truth, truth_aromatic = source_graph(key)
        # Exact post-hoc target join check: report must not alter source labels.
        if edge_set(row, "target_edges") != truth or edge_set(row, "target_aromatic_target_edges") != truth_aromatic:
            source_join_errors.append({"sample_key": key})

        baseline = edge_set(row, "baseline_predicted_target_edges")
        final = edge_set(row, "predicted_target_edges")
        baseline_aromatic = edge_set(row, "baseline_predicted_aromatic_target_edges")
        final_aromatic = edge_set(row, "predicted_aromatic_target_edges")
        if not final.issubset(baseline):
            edge_set_errors.append({"sample_key": key, "kind": "final_not_subset_baseline"})
        if baseline_aromatic != final_aromatic:
            aromatic_drops.append({
                "sample_key": key,
                "removed_aromatic_edges": [list(x) for x in sorted(baseline_aromatic - final_aromatic)],
                "added_aromatic_edges": [list(x) for x in sorted(final_aromatic - baseline_aromatic)],
            })

        record = cache_rows.get(key)
        if record is None:
            source_join_errors.append({"sample_key": key, "kind": "cache_missing"})
            continue
        nodes = sorted({int(n) for a, b, _ in record["pairs"] for n in (a, b)})
        mapping = record.get("mapping", {})
        # The rollback journal stores indices in this compact node table.
        for tx in row.get("transactions", []):
            transaction_counts["total"] += 1
            u, v = int(tx["edge"][0]), int(tx["edge"][1])
            detail = {"sample_key": key, "split": row["split"], "profile": row["profile"], "journal_edge": [u, v]}
            if not (0 <= u < len(nodes) and 0 <= v < len(nodes)):
                transaction_counts["invalid_index"] += 1
                detail["class"] = "invalid_index"
            else:
                tu, tv = mapping.get(nodes[u]), mapping.get(nodes[v])
                if tu is None or tv is None or int(tu) == int(tv):
                    transaction_counts["unmapped"] += 1
                    detail["class"] = "unmapped"
                else:
                    mapped = tuple(sorted((int(tu), int(tv))))
                    detail["mapped_target_edge"] = list(mapped)
                    if mapped in truth:
                        transaction_counts["true_target"] += 1
                        if mapped in truth_aromatic:
                            transaction_counts["true_aromatic_target"] += 1
                            detail["class"] = "true_aromatic_target"
                        else:
                            detail["class"] = "true_non_aromatic_target"
                    else:
                        transaction_counts["false_positive"] += 1
                        detail["class"] = "false_positive"
            tx_details.append(detail)

    identity_errors = []
    identity_hashes = {}
    for group, group_rows in sorted(groups.items()):
        hashes = {}
        for row in group_rows:
            graph, _, _ = source_graph(row["sample_key"])
            payload = graph_payload(graph)
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            hashes[row["profile"]] = digest
        identity_hashes[group] = hashes
        if len(set(hashes.values())) != 1:
            identity_errors.append({"group": group, "profiles": hashes})

    gate = {
        "rows_768": len(rows) == 768,
        "groups_192_x4": len(groups) == 192 and all(len(v) == 4 for v in groups.values()),
        "source_target_join_exact": not source_join_errors,
        "final_edge_subset_baseline": not edge_set_errors,
        "aromatic_edge_not_hidden": not aromatic_drops,
        "profile_source_identity_consistent": not identity_errors,
        # A true non-aromatic deletion is still a graph identity change.  It
        # may be acceptable for shadow recall accounting, but not for claiming
        # identity preservation or production integration.
        "no_true_target_edge_deletion": transaction_counts["true_target"] == 0,
        "no_true_aromatic_edge_deletion": transaction_counts["true_aromatic_target"] == 0,
        "no_unmapped_transaction": transaction_counts["unmapped"] == 0,
        "production_go": False,
    }
    report_out = {
        "schema_version": "chemvector-r17-rollback-identity-audit-m64-0.1",
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "inputs": {"rollback_report": str(REPORT.relative_to(ROOT)), "rollback_report_sha256": sha256(REPORT), "feature_cache": str(CACHE.relative_to(ROOT)), "feature_cache_sha256": sha256(CACHE)},
        "rows": len(rows),
        "groups": len(groups),
        "identity_errors": identity_errors,
        "source_join_errors": source_join_errors,
        "edge_set_errors": edge_set_errors,
        "aromatic_drops": aromatic_drops,
        "transaction_counts": dict(transaction_counts),
        "transactions": tx_details,
        "gate": gate,
        "interpretation": "shadow_only_true_target_deletions" if transaction_counts["true_target"] else "identity_preserving",
    }
    OUT.mkdir(parents=False)
    (OUT / "report.json").write_text(json.dumps(report_out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "transaction_counts": dict(transaction_counts), "gate": gate}, indent=2, ensure_ascii=False))
    return report_out


if __name__ == "__main__":
    run()
