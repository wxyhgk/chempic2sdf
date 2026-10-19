"""Frozen r5 Hungarian/edge sidecar round-trip on the identity-disjoint extension."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_hungarian_roundtrip_r5 as base  # noqa: E402
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry  # noqa: E402
from tools import shadow_line_first_edge_bond_type_r5 as shadow  # noqa: E402
from tools import preflight_line_first_edge_features_r5 as features  # noqa: E402

EXT = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-valence-extension-r5"
CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-extension-r5/candidates.jsonl"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-extension-roundtrip-r5/report.json"


def candidate_rows() -> dict[str, list[dict]]:
    rows = {}
    for line in CANDIDATES.read_text().splitlines():
        row = json.loads(line)
        if row.get("source") != "shadow" or set(row) != {"sample_key", "source", "candidates"}:
            raise ValueError("candidate contract failed")
        rows[row["sample_key"]] = row["candidates"]
    if len(rows) != 216:
        raise ValueError("extension candidate count changed")
    return rows


def main() -> dict:
    # Route every proposal/data lookup in the imported diagnostic modules to
    # the extension, while retaining the reviewed frozen sidecar state.
    base.DATA = EXT
    chemistry.DATA = EXT
    shadow.DATA = EXT
    features.DATA = EXT
    base.CANDIDATES = CANDIDATES
    chemistry.CANDIDATES = CANDIDATES
    chemistry._candidate_rows = candidate_rows
    base.audit = chemistry
    rows = [json.loads(line) for line in (EXT / "samples.jsonl").read_text().splitlines() if line.strip()]
    candidates = candidate_rows()
    state = torch.load(chemistry.STATE, map_location="cpu", weights_only=False)
    model = chemistry.EdgeBondMLP(len(state["feature_names"]))
    model.load_state_dict(state["model"]); model.eval()
    import numpy as np
    from collections import Counter, defaultdict
    mean, std = np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32)
    threshold = float(state["threshold"])
    total, profiles = Counter(), defaultdict(Counter)
    for row in sorted(rows, key=lambda item: item["sample_key"]):
        value = base.evaluate(row, candidates, model, mean, std, threshold)
        total.update(value); profiles[row["profile_id"]].update(value)
    outputs = {"all": base.finish(total), "by_profile": {k: base.finish(v) for k, v in sorted(profiles.items())}}
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-valence-extension-roundtrip-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "proposal_contract": "LSD -> target-free Hungarian ownership -> frozen edge/type sidecar; labels post-hoc",
        "images": len(rows), "metrics": outputs,
        "checkpoint_unchanged": chemistry.file_digest(chemistry.INITIAL) == chemistry.INITIAL_SHA,
        "next": "extension is train-only evidence; compare its graph metrics to fixed full holdout, no production merge",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
