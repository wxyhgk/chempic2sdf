#!/usr/bin/env python3
"""Train-only threshold calibration for the new r17 presence sidecar."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import tools.train_r17_pair_incidence_known_clean_cpu as base
from chemvector.bond_object import BOND_TYPES
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe

COHORT = ROOT / os.environ.get("R17_THRESHOLD_COHORT", "outputs/datasets/chemvector-r17-complexity-matched-source-20261022-v2/manifest.json")
SOURCE = ROOT / os.environ.get("R17_THRESHOLD_SOURCE", "outputs/datasets/chemvector-r17-complexity-matched-source-20261022-v2/samples")
PRESENCE = ROOT / os.environ.get("R17_THRESHOLD_PRESENCE", "outputs/audits/chemvector-r17-complexity-matched-presence-cpu-20261022-v2/sidecar-state.pt")
OUT = ROOT / os.environ.get("R17_THRESHOLD_OUT", "outputs/audits/chemvector-r17-presence-threshold-calibration-cpu-20261022-v1")
TRAIN_MOLECULES = int(os.environ.get("R17_THRESHOLD_TRAIN_MOLECULES", "32"))
HOLDOUT_MOLECULES = int(os.environ.get("R17_THRESHOLD_HOLDOUT_MOLECULES", "16"))
BASE_THRESHOLD = float(os.environ.get("R17_THRESHOLD_BASE", "0.95"))
WORKERS = max(1, int(os.environ.get("R17_THRESHOLD_WORKERS", "4")))

_MODEL = None
_CONFIG = None


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def init_worker(source: str):
    global _MODEL, _CONFIG
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base.SOURCE = Path(source)
    _MODEL, _CONFIG, _ = load_bundle_for_probe(base.INITIAL, SHA)
    _MODEL.eval().requires_grad_(False)


def extract_worker(row):
    return base.extract(row, _MODEL, _CONFIG)


def extract_rows(rows):
    if WORKERS <= 1:
        base.SOURCE = SOURCE
        model, config, _ = load_bundle_for_probe(base.INITIAL, SHA)
        model.eval().requires_grad_(False)
        return [base.extract(row, model, config) for row in rows]
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker, initargs=(str(SOURCE),)) as pool:
        return list(pool.map(extract_worker, rows))


def load_presence():
    state = torch.load(PRESENCE, map_location="cpu", weights_only=False)
    if state.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("presence checkpoint binding mismatch")
    head = base.Residual(262)
    head.load_state_dict(state["state_dict"]); head.eval()
    return head


def evaluate(items, presence, threshold):
    by = defaultdict(lambda: {"images": 0, "pred": 0, "true": 0, "typed": 0, "target": 0})
    for item in items:
        p = item["profile"]; c = by[p]; c["images"] += 1; c["target"] += len(item["target_edges"])
        with torch.inference_mode():
            ids = torch.where((item["base_logits"] + presence(item["features"])).sigmoid() >= threshold)[0]
            c["pred"] += len(ids); c["true"] += int(item["labels"][ids].sum())
            for i in ids.tolist():
                if item["labels"][i] > 0.5:
                    c["typed"] += int(BOND_TYPES[int(item["chemical_logits"][i].argmax())] == item["type_labels"][i])
    return {p: {"images": c["images"], "predicted_edges": c["pred"], "true_edges": c["true"], "target_edges": c["target"],
                "edge_precision": c["true"] / max(c["pred"], 1), "edge_recall": c["true"] / max(c["target"], 1),
                "typed_precision": c["typed"] / max(c["pred"], 1), "typed_recall": c["typed"] / max(c["target"], 1)} for p, c in sorted(by.items())}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(base.INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed")
    manifest = json.loads(COHORT.read_text())
    base.SOURCE = SOURCE; base.PAIR_THRESHOLD = BASE_THRESHOLD
    old_train, old_hold = base.TRAIN_MOLECULES, base.HOLDOUT_MOLECULES
    base.TRAIN_MOLECULES, base.HOLDOUT_MOLECULES = TRAIN_MOLECULES, HOLDOUT_MOLECULES
    train_rows, hold_rows = base.choose(manifest)
    if {r["molecule_id"] for r in train_rows} & {r["molecule_id"] for r in hold_rows}:
        raise RuntimeError("identity overlap")
    train, hold = extract_rows(train_rows), extract_rows(hold_rows)
    presence = load_presence()
    thresholds = [round(0.80 + 0.01 * i, 3) for i in range(20)] + [0.995]
    train_base = evaluate(train, presence, BASE_THRESHOLD)
    best = None
    for threshold in thresholds:
        got = evaluate(train, presence, threshold)
        if any(got[p]["edge_precision"] < train_base[p]["edge_precision"] - 0.01 or got[p]["typed_precision"] < train_base[p]["typed_precision"] - 0.01 for p in train_base):
            continue
        key = (sum(v["edge_recall"] for v in got.values()), sum(v["typed_recall"] for v in got.values()), threshold)
        if best is None or key > best["key"]:
            best = {"threshold": threshold, "key": key, "train": got}
    if best is None:
        best = {"threshold": BASE_THRESHOLD, "key": None, "train": train_base, "fallback": True}
    chosen = float(best["threshold"])
    hold_base = evaluate(hold, presence, BASE_THRESHOLD)
    hold_cal = evaluate(hold, presence, chosen)
    deltas = {p: {k: hold_cal[p][k] - hold_base[p][k] for k in ("edge_precision", "edge_recall", "typed_precision", "typed_recall")} for p in hold_base}
    binding = {"initial_checkpoint_sha256": SHA, "manifest_sha256": digest(COHORT), "train_keys": [r["sample_key"] for r in train_rows], "holdout_keys": [r["sample_key"] for r in hold_rows]}
    cache = {"binding": binding, "train": train, "holdout": hold}
    after = digest(base.INITIAL)
    report = {"schema_version": "chemvector-r17-presence-threshold-calibration-cpu-0.1", "complete": True, "training_performed": False,
              "training_scope": "train_only_threshold_selection_over_frozen_presence_sidecar", "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False,
              "cohort": str(COHORT.relative_to(ROOT)), "source": str(SOURCE.relative_to(ROOT)), "presence": str(PRESENCE.relative_to(ROOT)),
              "presence_sha256": digest(PRESENCE), "threshold_grid": thresholds, "base_threshold": BASE_THRESHOLD, "selected": best,
              "holdout_baseline": hold_base, "holdout_calibrated": hold_cal, "holdout_deltas": deltas,
              "cache_binding": binding, "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": after,
              "gate": {"checkpoint_unchanged": after == SHA, "sealed_test_read": False, "identity_disjoint": True,
                       "each_profile_edge_precision_drop_le_0.01": all(v["edge_precision"] >= -0.01 for v in deltas.values()),
                       "each_profile_typed_precision_drop_le_0.01": all(v["typed_precision"] >= -0.01 for v in deltas.values()),
                       "holdout_edge_recall_gain": any(v["edge_recall"] > 0 for v in deltas.values()), "production_go": False}}
    OUT.mkdir(parents=True); torch.save(cache, OUT / "feature-cache.pt"); (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "selected": best, "holdout_deltas": deltas, "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    run()
