#!/usr/bin/env python3
"""Post-hoc m128 candidate endpoint/bond reachability audit."""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m128-t0p72-c1-panel128/feature-cache.pt"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/samples"
OUT = ROOT / os.environ.get("R17_REACH_OUT", "outputs/audits/chemvector-r17-m128-reachability")


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    groups = defaultdict(lambda: {"images": 0, "target_bonds": 0, "reachable_bonds": 0, "unreachable_bonds": 0})
    rows = []
    for split, records in cache["records"].items():
        for record in records:
            graph = json.loads((SOURCE / f"{record['sample_key']}.graph.json").read_text())
            truth = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))) for b in graph["bonds"]}
            mapping = record["mapping"]
            candidates = {tuple(sorted((int(mapping[a]), int(mapping[b]))))
                          for a, b, _ in record["pairs"]
                          if a in mapping and b in mapping and mapping[a] != mapping[b]}
            profile = record["sample_key"].rsplit("__", 1)[-1]
            group = groups[(split, profile)]
            group["images"] += 1
            group["target_bonds"] += len(truth)
            group["reachable_bonds"] += len(truth & candidates)
            group["unreachable_bonds"] += len(truth - candidates)
            rows.append({"split": split, "profile": profile, "sample_key": record["sample_key"],
                         "target_bonds": len(truth), "reachable_bonds": len(truth & candidates),
                         "unreachable_bonds": len(truth - candidates)})
    summary = {}
    for split in ("train", "dev", "fresh_holdout"):
        subset = [value for (key, profile), value in groups.items() if key == split]
        target = sum(x["target_bonds"] for x in subset)
        unreachable = sum(x["unreachable_bonds"] for x in subset)
        summary[split] = {"images": sum(x["images"] for x in subset), "target_bonds": target,
                          "reachable_bonds": target - unreachable, "unreachable_bonds": unreachable,
                          "unreachable_rate": unreachable / max(target, 1),
                          "profiles": {profile: groups[(split, profile)] for profile in ("oa00", "oa11", "oa20", "oa33")}}
    result = {"schema_version": "chemvector-r17-m128-reachability-0.1", "complete": True,
              "read_only": True, "training_performed": False, "gpu_used": False,
              "sealed_test_read": False, "production_checkpoint_written": False,
              "candidate_edge_set_frozen": True, "summary": summary, "rows": rows,
              "gate": {"fresh_target_positive": summary["fresh_holdout"]["target_bonds"] > 0,
                       "fresh_unreachable_rate_le_0.05": summary["fresh_holdout"]["unreachable_rate"] <= 0.05,
                       "production_go": False}}
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "summary": summary, "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
