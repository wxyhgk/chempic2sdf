#!/usr/bin/env python3
"""Bounded target-free rollback audit for the r17 graph compiler.

This audit deliberately sits after the frozen AtomObject/AtomPair candidates
and after the quarantined r17 sidecars.  It never uses labels to choose an
edge, threshold, or stopping condition.  A graph which already sanitizes is
left untouched.  For an invalid graph we try removing one lowest-confidence
remaining edge at a time, compiling a deep in-memory candidate after every
attempt.  A successful removal is committed to the local transaction; if no
single-edge transaction succeeds, the result is explicitly abstained and the
invalid baseline is retained.  The input cache/checkpoints and production
checkpoint are never modified.
"""
from __future__ import annotations

from collections import Counter, defaultdict
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
from tools.audit_graph_context_aromatic_compile_r17 import cycle_edges
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from tools.train_element_context_head_cpu_r17 import ElementContextHead, ELEMENTS
from tools.train_graph_context_aromatic_cpu_r17 import AromaticContextHead
from tools.train_type_residual_cpu_r17 import TypeContextHead


CACHE = ROOT / os.environ.get("R17_ROLLBACK_CACHE", "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m64-t0p72-c1-element-context-m64/feature-cache.pt")
AROMATIC_STATE = ROOT / os.environ.get("R17_ROLLBACK_AROMATIC_STATE", str(CACHE.parent / "sidecar-state.pt"))
TYPE_STATE = ROOT / os.environ.get("R17_ROLLBACK_TYPE_STATE", "outputs/audits/chemvector-type-residual-cpu-r17-all-m64/sidecar-state.pt")
ELEMENT_STATE = ROOT / os.environ.get("R17_ROLLBACK_ELEMENT_STATE", "outputs/audits/chemvector-element-context-head-cpu-r17-all-m64/head-state.pt")
PLAN = ROOT / "outputs/audits/chemvector-aromatic-specialist-plan-r17-v2/rows.jsonl"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/samples"
OUT = ROOT / os.environ.get("R17_ROLLBACK_OUT", "outputs/audits/chemvector-r17-target-free-rollback-m64-v2")
THRESHOLD = 0.5
MAX_TRANSACTIONS = 8
MAX_ATTEMPTS_PER_IMAGE = 64
# Counterfactual shadow switch.  It only constrains candidate rollback
# transactions; baseline compilation remains unchanged.  The default keeps
# the historical audit behavior byte-for-byte compatible.
REQUIRE_CONNECTED_TRIAL = os.environ.get("R17_ROLLBACK_REQUIRE_CONNECTED", "0") == "1"
PRESERVE_CYCLE_TRIAL = os.environ.get("R17_ROLLBACK_PRESERVE_CYCLES", "0") == "1"


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
        # Every node in this cache is an endpoint of at least one pair.
        rows.append(torch.cat((record["atom_features"][node].float(),
                               torch.stack(values).mean(0),
                               torch.stack(values).max(0).values,
                               torch.tensor([float(len(values)) / 8.0]))))
    return torch.stack(rows)


def target_edges(sample_key: str) -> tuple[set[tuple[int, int]], dict[tuple[int, int], str]]:
    graph = json.loads((SOURCE / f"{sample_key}.graph.json").read_text())
    out = {}
    for bond in graph["bonds"]:
        edge = tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        out[edge] = str(bond.get("chemical_type", "SINGLE")).upper()
    return set(out), out


def compile_graph(nodes, edge_map, node_elements, aromatic_nodes):
    payload_nodes = [(node, node_elements[i], node in aromatic_nodes)
                     for i, node in enumerate(nodes)]
    return sanitize(payload_nodes, edge_map)


def connected_graph(nodes, edge_map) -> bool:
    """Return whether all candidate nodes form one connected component.

    This is deliberately target-free and uses only the proposed candidate
    edge set.  A one-node graph is connected; an empty graph is not.
    """
    if not nodes:
        return False
    adjacency = {int(node): set() for node in nodes}
    for left, right in edge_map:
        if left not in adjacency or right not in adjacency:
            return False
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        node = stack.pop()
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(nodes)


def edge_has_alternate_path(edge_map, edge) -> bool:
    """Target-free test for whether an edge participates in a candidate cycle."""
    left, right = edge
    adjacency = {}
    for u, v in edge_map:
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    seen = {left}
    stack = [left]
    while stack:
        node = stack.pop()
        for nxt in adjacency.get(node, ()):
            if {node, nxt} == {left, right}:
                continue
            if nxt == right:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def edge_metrics(records, rows_by_key):
    counters = Counter()
    by_profile = defaultdict(Counter)
    for row in records:
        split, profile = row["split"], row["profile"]
        c = counters
        p = by_profile[(split, profile)]
        pred = {tuple(x) for x in row["predicted_target_edges"]}
        truth = {tuple(x) for x in row["target_edges"]}
        arom_pred = {tuple(x) for x in row["predicted_aromatic_target_edges"]}
        arom_truth = {tuple(x) for x in row["target_aromatic_target_edges"]}
        for out, gold, prefix in ((pred, truth, "edge"), (arom_pred, arom_truth, "aromatic_edge")):
            c[(split, prefix, "pred")] += len(out)
            c[(split, prefix, "truth")] += len(gold)
            c[(split, prefix, "tp")] += len(out & gold)
            p[(prefix, "pred")] += len(out)
            p[(prefix, "truth")] += len(gold)
            p[(prefix, "tp")] += len(out & gold)
    def finish(c, split=None):
        def val(prefix, key):
            return c[(split, prefix, key)] if split is not None else c[(prefix, key)]
        result = {}
        for prefix in ("edge", "aromatic_edge"):
            pred, truth, tp = val(prefix, "pred"), val(prefix, "truth"), val(prefix, "tp")
            result[prefix + "_precision"] = tp / max(pred, 1)
            result[prefix + "_recall"] = tp / max(truth, 1)
            result[prefix + "_predicted"] = pred
            result[prefix + "_target"] = truth
            result[prefix + "_true_positive"] = tp
        return result
    summary = {split: finish(counters, split) for split in ("train", "dev", "fresh_holdout")}
    profiles = {}
    for (split, profile), c in sorted(by_profile.items()):
        d = {}
        for prefix in ("edge", "aromatic_edge"):
            pred, truth, tp = c[(prefix, "pred")], c[(prefix, "truth")], c[(prefix, "tp")]
            d[prefix + "_precision"] = tp / max(pred, 1)
            d[prefix + "_recall"] = tp / max(truth, 1)
            d[prefix + "_predicted"] = pred
            d[prefix + "_target"] = truth
            d[prefix + "_true_positive"] = tp
        profiles.setdefault(split, {})[profile] = d
    return summary, profiles


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    initial = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
    if digest(initial) != SHA:
        raise RuntimeError("production checkpoint changed")
    for path in (CACHE, AROMATIC_STATE, TYPE_STATE, ELEMENT_STATE, PLAN):
        if not path.exists():
            raise FileNotFoundError(path)
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    if cache.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("cache checkpoint binding mismatch")
    target_perturbation = os.environ.get("R17_ROLLBACK_PERTURB_TARGET", "0") == "1"
    # Sentinel mode intentionally destroys every target-bearing field in the
    # in-memory cache.  It is a contract check only; the source graph used for
    # post-hoc metrics remains untouched.  Routing/rollback must be identical
    # to the normal run.
    if target_perturbation:
        for records in cache["records"].values():
            for record in records:
                record["target_elements"] = {}
                record["target_aromatic_atoms"] = set()
                record["mapping"] = {}
                record["type_labels"] = torch.zeros_like(record["type_labels"])
    aromatic_state = torch.load(AROMATIC_STATE, map_location="cpu", weights_only=False)
    type_state = torch.load(TYPE_STATE, map_location="cpu", weights_only=False)
    element_state = torch.load(ELEMENT_STATE, map_location="cpu", weights_only=False)
    aromatic = AromaticContextHead(526); aromatic.load_state_dict(aromatic_state["state_dict"]); aromatic.eval()
    type_head = TypeContextHead(262); type_head.load_state_dict(type_state["state_dict"]); type_head.eval()
    element_head = ElementContextHead(); element_head.load_state_dict(element_state["state_dict"]); element_head.eval()
    plan = {json.loads(line)["sample_key"]: json.loads(line)
            for line in PLAN.read_text().splitlines() if line.strip()}
    all_rows = []
    with torch.inference_mode():
        for split, records in cache["records"].items():
            for record in records:
                if not record["pairs"]:
                    continue
                nodes = sorted({node for left, right, _ in record["pairs"] for node in (left, right)})
                local = {node: i for i, node in enumerate(nodes)}
                probabilities = aromatic(record["features"]).sigmoid().tolist()
                raw = {tuple(sorted((a, b))) for (a, b, _), p in zip(record["pairs"], probabilities) if p >= THRESHOLD}
                selected = cycle_edges(record["pairs"], raw)
                ctx_logits = element_head(context_features(record, nodes))
                ctx_pred = ctx_logits.argmax(-1).tolist()
                ctx_conf = ctx_logits.softmax(-1).max(-1).values.tolist()
                typed_logits = type_head(record["type_features"])
                typed = typed_logits.argmax(-1).tolist()
                type_conf = typed_logits.softmax(-1).max(-1).values.tolist()
                # Build a deterministic, target-free edge table.  Aromatic
                # edges use their positive score; ordinary edges use the
                # score against being aromatic, plus type confidence.
                edge_map = {}
                edge_info = {}
                for i, (a, b, _) in enumerate(record["pairs"]):
                    edge = tuple(sorted((local[a], local[b])))
                    kind = int(typed[i]) + 1
                    if kind == 4:
                        kind = 1
                    if (a, b) in selected:
                        kind = 4
                    edge_map[edge] = kind
                    endpoint_conf = min(ctx_conf[nodes.index(a)], ctx_conf[nodes.index(b)])
                    aromatic_conf = probabilities[i] if (a, b) in selected else (1.0 - probabilities[i])
                    confidence = float(endpoint_conf * type_conf[i] * max(aromatic_conf, 1e-6))
                    edge_info[edge] = {"confidence": confidence, "pair_index": i,
                                       "aromatic_probability": probabilities[i], "type_confidence": type_conf[i]}
                node_elements = [ELEMENTS[i] for i in ctx_pred]
                aromatic_nodes = {node for edge in selected for node in edge}
                ok, error = compile_graph(nodes, edge_map, node_elements, aromatic_nodes)
                baseline_edges = dict(edge_map)
                current = dict(edge_map)
                attempts = 0; transactions = []; abstain = False
                while not ok and attempts < MAX_ATTEMPTS_PER_IMAGE and transactions.__len__() < MAX_TRANSACTIONS and current:
                    accepted = False
                    candidates = sorted(current, key=lambda edge: (edge_info[edge]["confidence"], edge))
                    for edge in candidates:
                        if attempts >= MAX_ATTEMPTS_PER_IMAGE:
                            break
                        trial = dict(current); del trial[edge]
                        if PRESERVE_CYCLE_TRIAL and edge_has_alternate_path(current, edge):
                            continue
                        trial_ok, trial_error = compile_graph(nodes, trial, node_elements,
                                                              {u for u in aromatic_nodes if any(u in e for e in trial if trial[e] == 4)})
                        if trial_ok and REQUIRE_CONNECTED_TRIAL and not connected_graph(nodes, trial):
                            trial_ok, trial_error = False, "disconnected"
                        attempts += 1
                        if trial_ok:
                            transactions.append({"edge": list(edge), "confidence": edge_info[edge]["confidence"],
                                                 "pair_index": edge_info[edge]["pair_index"],
                                                 "before_error": error, "attempts_before_accept": attempts})
                            current = trial; ok, error = True, "none"; accepted = True
                            break
                    if not accepted:
                        abstain = True
                        break
                # Evaluation-only target join starts here.  It cannot affect
                # the rollback decisions above.
                target_set, target_types = target_edges(record["sample_key"])
                mapping = record["mapping"]
                def mapped_edges(edges):
                    out = set(); arom_out = set()
                    for (a, b), kind in edges.items():
                        ta, tb = mapping.get(nodes[a]), mapping.get(nodes[b])
                        if ta is None or tb is None or ta == tb:
                            continue
                        e = tuple(sorted((int(ta), int(tb))))
                        out.add(e)
                        if kind == 4:
                            arom_out.add(e)
                    return out, arom_out
                pred_before, arom_before = mapped_edges(baseline_edges)
                pred_after, arom_after = mapped_edges(current)
                target_arom = {e for e, t in target_types.items() if t == "AROMATIC"}
                all_rows.append({"split": split, "profile": plan.get(record["sample_key"], {}).get("profile_id", record["sample_key"].rsplit("__", 1)[-1]),
                                 "sample_key": record["sample_key"], "baseline_sanitize": bool(compile_graph(nodes, baseline_edges, node_elements, aromatic_nodes)[0]),
                                 "rollback_sanitize": bool(ok), "baseline_error": compile_graph(nodes, baseline_edges, node_elements, aromatic_nodes)[1],
                                 "rollback_error": error, "attempts": attempts, "rollback_count": len(transactions),
                                 "abstain": bool(abstain), "transactions": transactions,
                                 "predicted_target_edges": [list(x) for x in pred_after], "target_edges": [list(x) for x in target_set],
                                 "predicted_aromatic_target_edges": [list(x) for x in arom_after], "target_aromatic_target_edges": [list(x) for x in target_arom],
                                 "baseline_predicted_target_edges": [list(x) for x in pred_before], "baseline_predicted_aromatic_target_edges": [list(x) for x in arom_before]})
    # Build paired baseline/rollback metrics and per-profile accounting.
    rollback_summary, rollback_profiles = edge_metrics(all_rows, plan)
    baseline_rows = []
    for row in all_rows:
        b = dict(row); b["predicted_target_edges"] = row["baseline_predicted_target_edges"]; b["predicted_aromatic_target_edges"] = row["baseline_predicted_aromatic_target_edges"]; baseline_rows.append(b)
    baseline_summary, baseline_profiles = edge_metrics(baseline_rows, plan)
    summary = {}
    for split in ("train", "dev", "fresh_holdout"):
        rows = [r for r in all_rows if r["split"] == split]
        n = len(rows); base_ok = sum(r["baseline_sanitize"] for r in rows); roll_ok = sum(r["rollback_sanitize"] for r in rows)
        summary[split] = {"images": n, "baseline_sanitize_rate": base_ok / max(n, 1), "rollback_sanitize_rate": roll_ok / max(n, 1),
                          "delta_sanitize": (roll_ok - base_ok) / max(n, 1), "invalid_baseline": n - base_ok,
                          "accepted_transactions": sum(r["rollback_count"] for r in rows),
                          "removed_edges": sum(r["rollback_count"] for r in rows),
                          "rollback_images": sum(r["rollback_count"] > 0 for r in rows),
                          "abstain_images": sum(r["abstain"] for r in rows),
                          "abstain_edges": sum(r["abstain"] and r["rollback_count"] == 0 for r in rows),
                          "abstain_rate": sum(r["abstain"] for r in rows) / max(n, 1), "attempts": sum(r["attempts"] for r in rows),
                          "baseline_edges": baseline_summary[split], "rollback_edges": rollback_summary[split]}
    profile_summary = {}
    for split, profiles in rollback_profiles.items():
        profile_summary[split] = {}
        for profile, metrics in profiles.items():
            rr = [r for r in all_rows if r["split"] == split and r["profile"] == profile]
            br = [r for r in baseline_rows if r["split"] == split and r["profile"] == profile]
            profile_summary[split][profile] = {"images": len(rr), "baseline_sanitize_rate": sum(r["baseline_sanitize"] for r in br) / max(len(br), 1),
                                               "rollback_sanitize_rate": sum(r["rollback_sanitize"] for r in rr) / max(len(rr), 1),
                                               "removed_edges": sum(r["rollback_count"] for r in rr),
                                               "rollback_images": sum(r["rollback_count"] > 0 for r in rr),
                                               "abstain_rate": sum(r["abstain"] for r in rr) / max(len(rr), 1),
                                               "baseline_edges": baseline_profiles[split][profile], "rollback_edges": metrics}
    initial_sha = digest(initial)
    report = {"schema_version": "chemvector-r17-target-free-rollback-m64-0.1", "complete": True,
              "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False,
              "production_checkpoint_written": False, "cache": str(CACHE.relative_to(ROOT)), "max_transactions": MAX_TRANSACTIONS,
              "max_attempts_per_image": MAX_ATTEMPTS_PER_IMAGE, "threshold": THRESHOLD, "summary": summary,
              "profile_summary": profile_summary, "policy": {"target_free_selection": True, "evaluation_target_join_posthoc": True,
              "invalid_baseline_policy": "single-edge low-confidence rollback with compile check; otherwise abstain", "production_integration": "not_authorized"},
              "target_perturbation_sentinel": target_perturbation,
              "require_connected_trial": REQUIRE_CONNECTED_TRIAL,
              "preserve_cycle_trial": PRESERVE_CYCLE_TRIAL,
              "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": initial_sha,
              "rows": all_rows,
              "gate": {"checkpoint_unchanged": initial_sha == SHA, "sealed_test_read": False,
                       "fresh_sanitize_gain_ge_0.03": summary["fresh_holdout"]["delta_sanitize"] >= 0.03,
                       "dev_sanitize_ge_0.90": summary["dev"]["rollback_sanitize_rate"] >= 0.90,
                       "fresh_profile_no_drop_gt_0.02": all(v["rollback_sanitize_rate"] >= v["baseline_sanitize_rate"] - 0.02 for v in profile_summary["fresh_holdout"].values()),
                       "fresh_profile_abstain_le_0.30": all(v["abstain_rate"] <= 0.30 for v in profile_summary["fresh_holdout"].values()),
                       "production_go": False}}
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "summary": summary, "profile_summary": profile_summary, "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
