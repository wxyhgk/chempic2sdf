#!/usr/bin/env python3
"""Read-only dev replay for the already-trained r9 direct-pair sidecar."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import (
    DATA, SHA, DirectPairResidual, _extract, _finish, _score_rows,
)

OUT = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-dev-r9"
STATE = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r9/scorer-state.pt"
REPORT = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r9/report.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != SHA:
        raise RuntimeError("Production initial checkpoint changed")
    prior = json.loads(REPORT.read_text())
    threshold = float(prior["scorer"]["threshold_selected_train_only"])
    values = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    rows = sorted([row for row in values if row.get("split") == "dev"], key=lambda row: row["sample_key"])
    if len(rows) != 96:
        raise RuntimeError(f"Expected 96 dev rows, got {len(rows)}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, config, _ = load_bundle_for_probe(INITIAL, SHA)
    model.eval().requires_grad_(False)
    state = torch.load(STATE, map_location="cpu", weights_only=False)
    scorer = DirectPairResidual(input_dim=262)
    scorer.load_state_dict(state["state_dict"], strict=True)
    scorer.eval().requires_grad_(False)
    data = [_extract(row, model, config, need_labels=True) for row in rows]
    baseline = _score_rows(data, None, 0.85)
    scored = _score_rows(data, scorer, threshold)
    profiles = {}
    for profile in sorted({row["profile_id"] for row in rows}):
        subset = [item for item in data if item["row"]["profile_id"] == profile]
        profiles[profile] = {"baseline": _finish(_score_rows(subset, None, 0.85)), "scored": _finish(_score_rows(subset, scorer, threshold))}
    report = {
        "schema_version": "chemvector-identity-aligned-direct-graph-pair-scorer-dev-r9-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "split": "dev", "images": len(rows), "threshold_reused_train_only": threshold,
        "baseline": _finish(baseline), "scored": _finish(scored), "profiles": profiles,
        "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": digest(INITIAL),
        "gate": {"checkpoint_unchanged": digest(INITIAL) == SHA, "sealed_test_read": False, "production_go": False},
    }
    OUT.mkdir(parents=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "baseline": report["baseline"], "scored": report["scored"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
