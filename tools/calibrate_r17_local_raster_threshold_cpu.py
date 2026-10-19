#!/usr/bin/env python3
"""Train-only threshold calibration for the r17 local-raster presence probe."""
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
from tools.probe_r17_local_raster_presence_cpu import local_features, VisualResidual
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA

CACHE = ROOT / os.environ.get("R17_CAL_CACHE", "outputs/audits/chemvector-r17-presence-threshold-calibration-cpu-20261022-v1/feature-cache.pt")
MANIFEST = ROOT / os.environ.get("R17_CAL_MANIFEST", "outputs/datasets/chemvector-r17-complexity-matched-source-20261022-v2/manifest.json")
SIDECAR = ROOT / os.environ.get("R17_CAL_SIDECAR", "outputs/audits/chemvector-r17-local-raster-presence-cpu-20261022-v1/sidecar-state.pt")
OUT = ROOT / os.environ.get("R17_CAL_OUT", "outputs/audits/chemvector-r17-local-raster-threshold-calibration-cpu-20261022-v1")
SOURCE = ROOT / os.environ.get(
    "R17_CAL_SOURCE",
    "outputs/datasets/chemvector-r17-complexity-matched-source-20261022-v2/samples",
)
GRID = tuple(round(0.90 + i * 0.005, 3) for i in range(20)) + (0.999,)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def score(items: list[dict], head: VisualResidual | None, threshold: float) -> tuple[dict, dict]:
    aggregate = defaultdict(lambda: {"images": 0, "predicted": 0, "true": 0, "typed_true": 0, "target": 0})
    for item in items:
        logits = item["base_logits"] if head is None else item["base_logits"] + head(item["visual_features"])
        selected = logits.sigmoid() >= threshold
        labels = item["labels"] > .5
        row = aggregate[item["profile"]]
        row["images"] += 1
        row["predicted"] += int(selected.sum())
        row["true"] += int((selected & labels).sum())
        row["target"] += int(labels.sum())
        for idx in torch.where(selected & labels)[0].tolist():
            expected = item["type_labels"][idx]
            if expected is not None and BOND_TYPES[int(item["chemical_logits"][idx].argmax())] == expected:
                row["typed_true"] += 1

    def finish(row: dict) -> dict:
        out = dict(row)
        out["edge_precision"] = row["true"] / max(row["predicted"], 1)
        out["edge_recall"] = row["true"] / max(row["target"], 1)
        out["typed_edge_precision"] = row["typed_true"] / max(row["predicted"], 1)
        out["typed_edge_recall"] = row["typed_true"] / max(row["target"], 1)
        return out

    profile = {p: finish(row) for p, row in sorted(aggregate.items())}
    total = finish({key: sum(row[key] for row in aggregate.values()) for key in next(iter(aggregate.values()))})
    return total, profile


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    state = torch.load(SIDECAR, map_location="cpu", weights_only=False)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if cache.get("binding", {}).get("initial_checkpoint_sha256") != SHA or cache.get("binding", {}).get("manifest_sha256") != digest(MANIFEST):
        raise RuntimeError("cache binding mismatch")
    if state.get("initial_checkpoint_sha256") != SHA or state.get("cache_sha256") != digest(CACHE):
        raise RuntimeError("sidecar binding mismatch")
    meta = {row["sample_key"]: row for row in manifest["rows"]}
    items = []
    for split in ("train", "holdout"):
        for source in cache[split]:
            item = dict(source)
            item["visual_features"] = torch.from_numpy(local_features(SOURCE / f"{item['key']}.png", item["features"]))
            if item["key"] not in meta:
                raise RuntimeError(f"missing metadata for {item['key']}")
            items.append(item)
    train_keys = set(cache["binding"]["train_keys"])
    train = [item for item in items if item["key"] in train_keys]
    holdout = [item for item in items if item["key"] not in train_keys]
    head = VisualResidual(24)
    head.load_state_dict(state["state_dict"])
    head.eval()
    train_grid = []
    # The formal gate compares every candidate threshold against the frozen
    # initial presence head, never against the sidecar's own 0.95 output.
    baseline_train = score(train, None, 0.95)[1]
    for threshold in GRID:
        total, profiles = score(train, head, threshold)
        precision_gate = all(profiles[p]["edge_precision"] >= baseline_train[p]["edge_precision"] - .01 for p in profiles)
        train_grid.append({"threshold": threshold, "aggregate": total, "profiles": profiles, "precision_gate": precision_gate})
    admissible = [row for row in train_grid if row["precision_gate"]]
    selected = max(admissible, key=lambda row: (row["aggregate"]["edge_recall"], -row["threshold"])) if admissible else None
    baseline_holdout, baseline_holdout_profiles = score(holdout, None, 0.95)
    selected_holdout = None if selected is None else score(holdout, head, selected["threshold"])
    initial = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
    after = digest(initial)
    result = {
        "schema_version": "chemvector-r17-local-raster-threshold-calibration-cpu-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "cache": str(CACHE.relative_to(ROOT)),
        "sidecar": str(SIDECAR.relative_to(ROOT)),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "grid": train_grid,
        "selected_train": selected,
        "baseline_holdout_at_0p95": {"aggregate": baseline_holdout, "profiles": baseline_holdout_profiles},
        "selected_holdout": None if selected_holdout is None else {"threshold": selected["threshold"], "aggregate": selected_holdout[0], "profiles": selected_holdout[1]},
        "initial_checkpoint_sha256": SHA,
        "initial_checkpoint_sha256_after": after,
        "gate": {
            "cache_bound_to_initial": True,
            "identity_disjoint": True,
            "selection_train_only": True,
            "candidate_edge_set_frozen": True,
            "checkpoint_unchanged": after == SHA,
            "sealed_test_read": False,
            "selected_threshold_exists": selected is not None,
            "production_go": False,
        },
    }
    OUT.mkdir(parents=True)
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "selected": None if selected is None else selected["threshold"], "holdout": result["selected_holdout"], "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
