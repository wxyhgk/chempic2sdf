#!/usr/bin/env python3
"""Attribute the r17 presence probe's errors without opening sealed data.

This audit consumes the already materialized, identity-disjoint feature cache
and the sidecar produced by ``train_r17_pair_incidence_known_clean_cpu.py``.
It never re-runs the model and never changes the frozen candidate edge set.
The goal is to answer whether the observed precision loss is concentrated in
specific profiles, multi-bond/ring strata, or endpoint-tagged molecules.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from chemvector.bond_object import BOND_TYPES
from tools.train_r17_pair_incidence_known_clean_cpu import Residual
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA

CACHE = ROOT / os.environ.get(
    "R17_ATTR_CACHE",
    "outputs/audits/chemvector-r17-presence-threshold-calibration-cpu-20261022-v1/feature-cache.pt",
)
SIDECAR = ROOT / os.environ.get(
    "R17_ATTR_SIDECAR",
    "outputs/audits/chemvector-r17-complexity-matched-presence-cpu-20261022-v2/sidecar-state.pt",
)
MANIFEST = ROOT / os.environ.get(
    "R17_ATTR_MANIFEST",
    "outputs/datasets/chemvector-r17-complexity-matched-source-20261022-v2/manifest.json",
)
OUT = ROOT / os.environ.get(
    "R17_ATTR_OUT",
    "outputs/audits/chemvector-r17-presence-failure-attribution-cpu-20261022-v1",
)
THRESHOLDS = (0.95, 0.97)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _bucket_tags(tags: list[str]) -> list[str]:
    tags = {str(tag) for tag in tags}
    buckets = ["all"]
    if "GRAPH:AROMATIC" in tags:
        buckets.append("aromatic")
    if "RING" in tags:
        buckets.append("ring")
    if "GRAPH:DOUBLE" in tags:
        buckets.append("double")
    if "GRAPH:TRIPLE" in tags:
        buckets.append("triple")
    if any(tag.startswith("EP:N:") for tag in tags):
        buckets.append("endpoint_N")
    if any(tag.startswith("EP:O:") for tag in tags):
        buckets.append("endpoint_O")
    if any(tag.startswith("EP:S:") for tag in tags):
        buckets.append("endpoint_S")
    if any(tag.startswith("EP:") for tag in tags):
        buckets.append("endpoint_any")
    return buckets


def _counts(items: list[dict], threshold: float, head: Residual | None) -> tuple[dict, list[dict]]:
    aggregate = defaultdict(lambda: {"images": 0, "target": 0, "predicted": 0, "true": 0, "typed_true": 0, "fp": 0, "fn": 0, "type_mismatch": 0, "new_fp": 0, "new_true": 0})
    details: list[dict] = []
    for item in items:
        base = item["base_logits"].float()
        delta = torch.zeros_like(base) if head is None else head(item["features"].float()).detach()
        scored = base + delta
        selected = scored.sigmoid() >= threshold
        base_selected = base.sigmoid() >= threshold
        labels = item["labels"].float() > 0.5
        tp = selected & labels
        fp = selected & ~labels
        fn = ~selected & labels
        new_fp = fp & ~base_selected
        new_true = tp & ~base_selected
        typed_true = 0
        type_mismatch = 0
        for idx in torch.where(tp)[0].tolist():
            expected = item["type_labels"][idx]
            if expected is None:
                continue
            if BOND_TYPES[int(item["chemical_logits"][idx].argmax())] == expected:
                typed_true += 1
            else:
                type_mismatch += 1
        meta = item["_meta"]
        buckets = _bucket_tags(meta.get("endpoint_tags", []))
        values = {
            "images": 1,
            "target": int(labels.sum()),
            "predicted": int(selected.sum()),
            "true": int(tp.sum()),
            "typed_true": typed_true,
            "fp": int(fp.sum()),
            "fn": int(fn.sum()),
            "type_mismatch": type_mismatch,
            "new_fp": int(new_fp.sum()),
            "new_true": int(new_true.sum()),
        }
        for bucket in buckets:
            for key, value in values.items():
                aggregate[(meta["profile_id"], bucket)][key] += value
        if fp.any():
            for idx in torch.where(fp)[0].tolist():
                details.append({
                    "sample_key": item["key"],
                    "profile": meta["profile_id"],
                    "molecule_id": meta["molecule_id"],
                    "canonical_smiles": meta.get("canonical_smiles"),
                    "pair": [int(v) for v in item["pairs"][idx].tolist()],
                    "logit": float(scored[idx]),
                    "probability": float(scored[idx].sigmoid()),
                    "new_since_baseline": bool(new_fp[idx]),
                    "predicted_type": BOND_TYPES[int(item["chemical_logits"][idx].argmax())],
                    "endpoint_tags": meta.get("endpoint_tags", []),
                })

    def finish(row: dict) -> dict:
        out = dict(row)
        out["edge_precision"] = row["true"] / max(row["predicted"], 1)
        out["edge_recall"] = row["true"] / max(row["target"], 1)
        out["typed_edge_precision"] = row["typed_true"] / max(row["predicted"], 1)
        out["typed_edge_recall"] = row["typed_true"] / max(row["target"], 1)
        return out

    return {f"{profile}/{bucket}": finish(row) for (profile, bucket), row in sorted(aggregate.items())}, details


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    sidecar = torch.load(SIDECAR, map_location="cpu", weights_only=False)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    metadata = {row["sample_key"]: row for row in manifest["rows"]}
    if sidecar.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("sidecar is not bound to frozen initial checkpoint")
    if cache.get("binding", {}).get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("feature cache is not bound to frozen initial checkpoint")
    if cache.get("binding", {}).get("manifest_sha256") != digest(MANIFEST):
        raise RuntimeError("feature cache/manifest binding mismatch")
    head = Residual(262)
    head.load_state_dict(sidecar["state_dict"])
    head.eval()
    items = []
    for item in cache["holdout"]:
        clone = dict(item)
        if clone["key"] not in metadata:
            raise RuntimeError(f"missing manifest metadata for {clone['key']}")
        clone["_meta"] = metadata[clone["key"]]
        items.append(clone)
    reports = {}
    all_details = {}
    for threshold in THRESHOLDS:
        key = str(threshold).replace(".", "p")
        scored, details = _counts(items, threshold, head)
        baseline, _ = _counts(items, threshold, None)
        reports[key] = {"scored": scored, "baseline": baseline}
        all_details[key] = details
    result = {
        "schema_version": "chemvector-r17-presence-failure-attribution-cpu-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "cache": str(CACHE.relative_to(ROOT)),
        "sidecar": str(SIDECAR.relative_to(ROOT)),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "holdout_images": len(items),
        "reports": reports,
        "false_positive_details": all_details,
        "initial_checkpoint_sha256": SHA,
        "initial_checkpoint_sha256_after": digest(ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"),
        "gate": {
            "cache_bound_to_initial": True,
            "manifest_bound_to_cache": True,
            "candidate_edge_set_frozen": True,
            "checkpoint_unchanged": digest(ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt") == SHA,
            "sealed_test_read": False,
            "production_go": False,
        },
    }
    OUT.mkdir(parents=True)
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "gate": result["gate"], "false_positives": {k: len(v) for k, v in all_details.items()}}, indent=2))


if __name__ == "__main__":
    main()
