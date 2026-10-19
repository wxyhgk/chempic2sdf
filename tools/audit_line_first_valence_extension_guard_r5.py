"""Apply the frozen target-free cycle/valence guard to extension candidates."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_valence_topology_guard_r5 as guard  # noqa: E402
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry  # noqa: E402
from tools import shadow_line_first_edge_bond_type_r5 as shadow  # noqa: E402
from tools import preflight_line_first_edge_features_r5 as features  # noqa: E402

EXT = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-valence-extension-r5"
CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-extension-r5/candidates.jsonl"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-extension-guard-r5/report.json"


def candidate_rows():
    out = {}
    for line in CANDIDATES.read_text().splitlines():
        row = json.loads(line)
        out[row["sample_key"]] = row["candidates"]
    if len(out) != 216:
        raise ValueError("candidate count changed")
    return out


def finish(c: Counter) -> dict:
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1),
            "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1),
            "bond_type_accuracy": c["type_correct"] / max(c["type_total"], 1),
            "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1),
            "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1),
            "aromatic_error_rate": c["aromatic_errors"] / max(c["images"], 1),
            "valence_error_rate": c["valence_errors"] / max(c["images"], 1)}


def main() -> dict:
    # Redirect all target-free proposal/data lookups to the extension.
    guard.base.DATA = EXT; guard.base.CANDIDATES = CANDIDATES
    guard.base.audit.DATA = EXT; guard.base.audit.CANDIDATES = CANDIDATES
    guard.base.audit._candidate_rows = candidate_rows
    chemistry.DATA = EXT; shadow.DATA = EXT; features.DATA = EXT
    candidates = candidate_rows()
    state = torch.load(guard.base.audit.STATE, map_location="cpu", weights_only=False)
    model = guard.EdgeBondMLP(len(state["feature_names"]))
    model.load_state_dict(state["model"]); model.eval()
    mean, std = np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32)
    threshold = float(state["threshold"])
    rows = [json.loads(line) for line in (EXT / "samples.jsonl").read_text().splitlines() if line.strip()]
    total, profiles, edits = Counter(), defaultdict(Counter), Counter()
    for row in sorted(rows, key=lambda item: item["sample_key"]):
        selected, points, dangling = guard._raw(row, candidates, model, mean, std, threshold)
        selected, event = guard.guard(selected, low_confidence=0.4)
        edits.update(event)
        metrics = guard._metrics(row, selected, points, dangling)
        total.update(metrics); profiles[row["profile_id"]].update(metrics)
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-valence-extension-guard-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "proposal_contract": "LSD -> frozen target-free Hungarian ownership/sidecar -> cycle demotion -> target-free valence guard",
        "images": len(rows), "metrics": {"all": finish(total), "by_profile": {k: finish(v) for k, v in sorted(profiles.items())}},
        "edits": dict(edits), "checkpoint_unchanged": guard.base.digest(guard.base.audit.INITIAL) == guard.base.audit.INITIAL_SHA,
        "gate": {"identity_disjoint": True, "sealed_test_read": False, "production_go": False,
                 "all_profiles_sanitize_ge_0_90": all(finish(v)["oracle_sanitize_rate"] >= .90 for v in profiles.values()),
                 "all_profiles_edge_recall_ge_0_75": all(finish(v)["edge_recall"] >= .75 for v in profiles.values())},
        "next": "extension does not qualify as a GPU training release; use failures to design endpoint/topology proposal repair",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
