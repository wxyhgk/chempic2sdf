"""Parallel target-free endpoint/edge ceiling audit for the 6k panel.

GT is joined only after line proposals are constructed.  This is a proposal
capacity audit, not a deployable identity or bond-type result.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import audit_line_first_direct_vs_node_graph_strict_r5 as strict

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-6k-r5-parallel"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-targeted-6k-endpoint-ceiling-r5"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def _job(row):
    strict.DATA = DATA
    value = strict._direct(row)
    value["profile_id"] = row["profile_id"]
    value["split"] = row["split"]
    return value


def _finish(rows: list[dict]) -> dict:
    c = Counter()
    fields = ("images", "pred_nodes", "matched_nodes", "target_atoms", "selected_edges",
              "mapped_edges", "true_edges", "gt_bonds", "unresolved_edges", "self_loops")
    for row in rows:
        c.update({field: int(row.get(field, 0)) for field in fields})
    out = dict(c)
    out.update({
        "node_precision": c["matched_nodes"] / max(c["pred_nodes"], 1),
        "node_recall": c["matched_nodes"] / max(c["target_atoms"], 1),
        "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1),
        "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1),
        "dangling_rate": c["unresolved_edges"] / max(c["selected_edges"], 1),
    })
    return out


def main(workers: int = 16) -> dict:
    if hashlib.sha256(strict.shadow.INITIAL.read_bytes()).hexdigest() != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    rows = sorted([row for row in rows if row.get("render_status") == "accepted"], key=lambda row: row["sample_key"])
    requested_split = os.environ.get("TARGET_SPLIT", "all")
    if requested_split != "all":
        rows = [row for row in rows if row["split"] == requested_split]
    if requested_split == "fresh_holdout" and len(rows) != 599:
        raise RuntimeError(f"expected 599 fresh holdout rows, got {len(rows)}")
    result = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, row) for row in rows]
        for index, future in enumerate(as_completed(futures), 1):
            result.append(future.result())
            if index % 500 == 0:
                print(json.dumps({"completed": index, "total": len(rows)}), flush=True)
    result.sort(key=lambda row: (row["split"], row["sample_key"] if "sample_key" in row else ""))
    splits = {}
    for split in sorted({row["split"] for row in result}):
        split_rows = [row for row in result if row["split"] == split]
        profiles = {profile: _finish([row for row in split_rows if row["profile_id"] == profile])
                    for profile in sorted({row["profile_id"] for row in split_rows})}
        splits[split] = {"all": _finish(split_rows), "by_profile": profiles}
    hold = splits.get("fresh_holdout", splits[next(iter(splits))])["all"]
    hp = splits.get("fresh_holdout", splits[next(iter(splits))])["by_profile"]
    gate = {
        "checkpoint_unchanged": hashlib.sha256(strict.shadow.INITIAL.read_bytes()).hexdigest() == INITIAL_SHA,
        "sealed_test_read": False,
        "fresh_holdout_images": hold["images"] == 599 if requested_split == "fresh_holdout" else "fresh_holdout" in splits,
        "proposal_precision_ge_0.95": hold["edge_precision"] >= .95,
        "proposal_recall_ge_0.80": hold["edge_recall"] >= .80,
        "all_profile_recall_ge_0.75": all(value["edge_recall"] >= .75 for value in hp.values()),
        "all_profile_sanitize_ge_0.90": False,
        "sanitize_not_evaluated_in_strict_direct_replay": True,
        "training_admissible": False,
        "gpu_go": False,
    }
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-targeted-6k-endpoint-ceiling-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False,
        "initial_sha256": INITIAL_SHA,
        "proposal_contract": "new-LSD cluster + glyph alias; GT post-hoc only; synthetic SINGLE ceiling",
        "dataset": str(DATA), "requested_split": requested_split, "rows": len(rows), "splits": splits, "gate": gate,
        "interpretation": "proposal capacity only; strict node identity and bond type remain separate gates",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "fresh_holdout": hold, "gate": gate}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
