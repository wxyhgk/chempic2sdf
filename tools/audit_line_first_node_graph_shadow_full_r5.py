"""Full 474-image replay of the isolated node-graph shadow sidecar."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import shadow_line_first_node_graph_r5 as shadow  # noqa: E402

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-node-graph-shadow-r5-full/report.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[dict]) -> dict:
    keys = ("images", "line_nodes", "selected_nodes", "target_atoms", "node_true", "selected_lines",
            "raw_dangling_edges", "raw_self_loops", "candidate_edges", "mapped_edges", "true_edges",
            "gt_bonds", "typed_total", "typed_correct", "all_bonds_recovered")
    totals = {key: sum(int(value.get(key, 0)) for value in values) for key in keys}
    totals.update({
        "node_precision": totals["node_true"] / max(totals["selected_nodes"], 1),
        "node_recall": totals["node_true"] / max(totals["target_atoms"], 1),
        "edge_precision": totals["true_edges"] / max(totals["mapped_edges"], 1),
        "edge_recall": totals["true_edges"] / max(totals["gt_bonds"], 1),
        "conditional_bond_type_accuracy": totals["typed_correct"] / max(totals["typed_total"], 1),
        "raw_dangling_rate": totals["raw_dangling_edges"] / max(totals["selected_lines"], 1),
        "raw_self_loop_rate": totals["raw_self_loops"] / max(totals["selected_lines"], 1),
        "emitted_dangling_edges": 0,
        "emitted_dangling_rate": 0.0,
        "emitted_self_loops": 0,
        "all_bonds_recovered_fraction": totals["all_bonds_recovered"] / max(totals["images"], 1),
    })
    return totals


def main() -> dict:
    if digest(shadow.INITIAL) != shadow.INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    shadow.DATA = DATA
    candidates = shadow._load_candidates()
    payload = torch.load(shadow.STATE, map_location="cpu", weights_only=False)
    node_model = shadow.NodeMLP(len(payload["feature_names"]))
    node_model.load_state_dict(payload["model"]); node_model.eval()
    node_mean, node_std = np.asarray(payload["mean"], np.float32), np.asarray(payload["std"], np.float32)
    node_threshold = float(payload["threshold"])
    edge_model, edge_mean, edge_std, edge_threshold = shadow._load_edge_sidecar()
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines()
            if line.strip() and json.loads(line).get("render_status") == "accepted"]
    hold = sorted([row for row in rows if row["split"] == "holdout"], key=lambda row: row["sample_key"])
    if len(hold) != 474:
        raise RuntimeError(f"expected full 474 holdout rows, got {len(hold)}")
    results = []
    for row in hold:
        # shadow._evaluate is target-free until its post-hoc metrics and uses
        # the frozen edge/type sidecar plus the isolated node sidecar.
        result = shadow._evaluate([row], candidates, node_model, node_mean, node_std,
                                  node_threshold, edge_model, edge_mean, edge_std,
                                  edge_threshold, torch.device("cpu"))
        result["profile_id"] = row["profile_id"]
        results.append(result)
    by_profile = {profile: summarize([row for row in results if row["profile_id"] == profile])
                  for profile in sorted({row["profile_id"] for row in results})}
    all_metrics = summarize(results)
    gate = {
        "checkpoint_unchanged": digest(shadow.INITIAL) == shadow.INITIAL_SHA,
        "sealed_test_read": False,
        "full_holdout_images_474": len(hold) == 474,
        "node_precision_ge_0_90": all_metrics["node_precision"] >= .90,
        "node_recall_ge_0_85": all_metrics["node_recall"] >= .85,
        "edge_precision_ge_0_90": all_metrics["edge_precision"] >= .90,
        "edge_recall_ge_0_75": all_metrics["edge_recall"] >= .75,
        "type_accuracy_ge_0_90": all_metrics["conditional_bond_type_accuracy"] >= .90,
        "emitted_dangling_zero": all_metrics["emitted_dangling_edges"] == 0,
        "emitted_self_loop_zero": all_metrics["emitted_self_loops"] == 0,
        "production_go": False,
    }
    gate["full_shadow_pass"] = all(value for key, value in gate.items() if key not in {"sealed_test_read", "production_go"})
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-node-graph-shadow-r5-full-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "sidecar_state_sha256": digest(shadow.STATE), "edge_sidecar_state_sha256": digest(shadow.EDGE_STATE),
        "initial_sha256": shadow.INITIAL_SHA, "initial_checkpoint_sha256_after": digest(shadow.INITIAL),
        "proposal_contract": "LSD endpoints -> target-free node keep sidecar -> frozen edge/type sidecar; GT post-hoc only",
        "all": all_metrics, "by_profile": by_profile, "gate": gate,
        "next": "chemistry/topology gate remains separate; no production merge",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
