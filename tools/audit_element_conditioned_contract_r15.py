#!/usr/bin/env python3
"""Contract and paired-receipt audit for the frozen r15 CPU sidecar.

This is a read-only audit.  It verifies that the element-conditioned repair
does not consume target fields, that the edge proposal is bitwise unchanged,
and that the fixed train/dev/fresh identities are disjoint.  It intentionally
does not open any sealed test artifact and never writes a model checkpoint.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.audit_element_conditioned_repair_r15 as repair
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import one
from tools.probe_atom_element_head_cpu_r15 import ELEMENTS
from tools.train_direct_graph_pair_scorer_cpu_r9 import (
    DATA, SHA, DirectPairResidual, _extract, rows,
)
from tools.train_direct_graph_type_residual_cpu_r12_corrected import TypeResidual

OUT = ROOT / "outputs/audits/chemvector-element-conditioned-contract-r15"
FIELDS = ("features", "base_logits", "chemical_logits", "pairs", "slots", "points")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fixed_train(all_rows: list[dict]) -> list[dict]:
    by_profile: dict[str, list[dict]] = defaultdict(list)
    for row in sorted((r for r in all_rows if r.get("split") == "train"), key=lambda r: r["sample_key"]):
        by_profile[row["profile_id"]].append(row)
    return [row for profile in sorted(by_profile) for row in by_profile[profile][:24]]


def same_forward(left: dict, right: dict) -> bool:
    for field in FIELDS:
        a, b = left[field], right[field]
        if isinstance(a, torch.Tensor):
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                return False
        elif a != b:
            return False
    return True


def edge_ids(item: dict, presence: torch.nn.Module, threshold: float) -> set[int]:
    scores = (item["base_logits"] + presence(item["features"])).sigmoid()
    return {int(i) for i in torch.where(scores >= threshold)[0].tolist()}


def load_components():
    if digest(INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed before audit")
    trial = repair.TRIAL
    ps = torch.load(trial / "scorer-state.pt", map_location="cpu", weights_only=False)
    presence = DirectPairResidual(262)
    presence.load_state_dict(ps["state_dict"])
    presence.eval()
    ts = torch.load(repair.TS, map_location="cpu", weights_only=False)
    typer = TypeResidual(262)
    typer.load_state_dict(ts["state_dict"])
    typer.eval()
    es = torch.load(repair.ES, map_location="cpu", weights_only=False)
    element_head = torch.nn.Sequential(torch.nn.LayerNorm(128), torch.nn.Linear(128, len(ELEMENTS)))
    element_head.load_state_dict(es["state_dict"])
    element_head.eval()
    model, cfg, _ = repair.load_bundle_for_probe(INITIAL, SHA)
    model.eval().requires_grad_(False)
    return model, cfg, presence, typer, element_head


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    all_rows = rows()
    splits = {
        "train": fixed_train(all_rows),
        "dev": sorted((r for r in all_rows if r.get("split") == "dev"), key=lambda r: r["sample_key"]),
        "fresh_holdout": sorted((r for r in all_rows if r.get("split") == "fresh_holdout"), key=lambda r: r["sample_key"]),
    }
    identities = {name: {r["molecule_id"] for r in value} for name, value in splits.items()}
    overlaps = {
        f"{a}:{b}": sorted(identities[a] & identities[b])
        for a, b in (("train", "dev"), ("train", "fresh_holdout"), ("dev", "fresh_holdout"))
    }
    model, cfg, presence, typer, element_head = load_components()
    rows_report: list[dict] = []
    sentinel_failures = []
    edge_failures = []
    paired = Counter()
    threshold = float(repair.THRESHOLD)
    for split, split_rows in splits.items():
        for row in split_rows:
            with_labels = _extract(row, model, cfg, need_labels=True)
            without_labels = _extract(row, model, cfg, need_labels=False)
            forward_equal = same_forward(with_labels, without_labels)
            if not forward_equal:
                sentinel_failures.append(row["sample_key"])
            q_without, pred_without = repair.element_repair(
                without_labels, presence, typer, element_head, model, cfg
            )
            q_with, pred_with = repair.element_repair(
                with_labels, presence, typer, element_head, model, cfg
            )
            repair_equal = (
                torch.equal(q_without["chemical_logits"], q_with["chemical_logits"])
                and pred_without == pred_with
            )
            before_edges = edge_ids(with_labels, presence, threshold)
            after_edges = edge_ids(without_labels, presence, threshold)
            if before_edges != after_edges:
                edge_failures.append(row["sample_key"])
            if split != "train":
                metrics = one(q_with, presence, threshold)
                paired.update(metrics)
            rows_report.append({
                "split": split,
                "sample_key": row["sample_key"],
                "forward_equal_without_labels": forward_equal,
                "repair_equal_without_labels": repair_equal,
                "edge_set_equal": before_edges == after_edges,
                "selected_edges": len(before_edges),
                "candidate_sanitize_ok": int(metrics["candidate_sanitize_ok"]) if split != "train" else None,
                "oracle_sanitize_ok": int(metrics["oracle_sanitize_ok"]) if split != "train" else None,
            })
    after = digest(INITIAL)
    report = {
        "schema_version": "chemvector-element-conditioned-contract-r15-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "profile_metadata_used": False,
        "threshold": threshold,
        "split_counts": {name: len(value) for name, value in splits.items()},
        "identity_overlaps": overlaps,
        "target_free_sentinel": {
            "checked_rows": len(rows_report),
            "forward_mismatches": len(sentinel_failures),
            "repair_mismatches": sum(not r["repair_equal_without_labels"] for r in rows_report),
            "failed_sample_keys": sentinel_failures,
        },
        "edge_set_freeze": {
            "checked_rows": len(rows_report),
            "mismatches": len(edge_failures),
            "failed_sample_keys": edge_failures,
        },
        "paired_nontrain_metrics": dict(paired),
        "rows": rows_report,
        "initial_checkpoint_sha256": SHA,
        "initial_checkpoint_sha256_after": after,
        "gate": {
            "identity_disjoint": all(not value for value in overlaps.values()),
            "target_free_forward_sentinel": not sentinel_failures,
            "target_free_repair_sentinel": all(r["repair_equal_without_labels"] for r in rows_report),
            "edge_set_bitwise_frozen": not edge_failures,
            "checkpoint_unchanged": after == SHA,
            "sealed_test_read": False,
            "production_go": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "gate": report["gate"]}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
