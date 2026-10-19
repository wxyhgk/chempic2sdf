#!/usr/bin/env python3
"""Read-only candidate-pair capacity diagnostic for the r5 shadow.

The frozen model and scored candidate artifact are reused for every capacity.
Only the predeclared per-endpoint candidate cap (24/48/96) changes.  No MLP,
threshold, checkpoint, graph merge, or production tensor is trained/written.
Labels are used in memory solely to report target-pair reachability.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from collections import defaultdict
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tools.preflight_endpoint_pair_compatibility_r5 as pair
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe

OUT = ROOT / "outputs/audits/chemvector-identity-aligned-endpoint-pair-capacity-r5/report.json"
CAPS = (24, 48, 96)


def _empty():
    return {"images": 0, "pairs": 0, "positive_pairs": 0, "negative_pairs": 0,
            "target_bonds": 0, "mapped_target_bonds": 0, "reachable_target_bonds": 0,
            "mapped_target_reachability": 0.0, "target_pair_reachability": 0.0}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if file_digest(INITIAL) != pair.INITIAL_SHA:
        raise RuntimeError("production checkpoint digest changed")
    candidates = pair._candidates()
    rows = [json.loads(x) for x in (pair.DATA / "samples.jsonl").read_text().splitlines() if x]
    rows = [x for x in rows if x.get("render_status") == "accepted"]
    if len(rows) != 1170:
        raise RuntimeError(f"r5 accepted row count changed: {len(rows)}")
    selected_rows = {
        "train": sorted([x for x in rows if x["split"] == "train"], key=lambda x: x["sample_key"])[:pair.TRAIN_LIMIT],
        "holdout": sorted([x for x in rows if x["split"] == "holdout"], key=lambda x: x["sample_key"])[:pair.HOLDOUT_LIMIT],
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, config, payload = load_bundle_for_probe(INITIAL, pair.INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
    result = {"train": {}, "holdout": {}}
    old_cap = pair.MAX_PER_SIDE
    try:
        for cap in CAPS:
            pair.MAX_PER_SIDE = cap
            cap_result = {}
            for split, chosen in selected_rows.items():
                total = _empty()
                profile = defaultdict(_empty)
                for row in chosen:
                    sample = load_sample(pair.DATA, row["sample_key"], purpose="diagnostic")
                    with torch.inference_mode():
                        out = structure_forward(model, image_tensor(sample["image"])[None].to(device), config, payload["provenance"]["atom_threshold"])
                    bonds = {k: v.detach().cpu() for k, v in out[4].items()}
                    selected = {k: v.detach().cpu() for k, v in out[6].items()}
                    feats, meta, labels = pair._candidate_feature_rows(sample, candidates[row["sample_key"]], bonds, selected)
                    points, target_bonds = pair._target(sample)
                    mapped = {}
                    for qi in torch.where(bonds["bond_mask"][0].bool())[0].tolist():
                        hit = pair._proposal_target(bonds["bond_endpoints"][0, qi], points, target_bonds)
                        if hit is not None:
                            mapped[hit[0]] = hit[0]
                    positive_q = {m[0] for m, y in zip(meta, labels) if y}
                    # Map positive proposal queries to target bond identities;
                    # this is diagnostic only and is not put in any artifact.
                    reachable = set()
                    for qi in positive_q:
                        hit = pair._proposal_target(bonds["bond_endpoints"][0, qi], points, target_bonds)
                        if hit is not None:
                            reachable.add(hit[0])
                    vals = {"images": 1, "pairs": len(labels), "positive_pairs": int(sum(labels)),
                            "negative_pairs": int(len(labels) - sum(labels)), "target_bonds": len(target_bonds),
                            "mapped_target_bonds": len(mapped), "reachable_target_bonds": len(reachable)}
                    for key, value in vals.items():
                        total[key] += value
                        profile[row["profile_id"]][key] += value
                total["mapped_target_reachability"] = total["mapped_target_bonds"] / max(total["target_bonds"], 1)
                total["target_pair_reachability"] = total["reachable_target_bonds"] / max(total["target_bonds"], 1)
                by_profile = {}
                for name, value in profile.items():
                    value["mapped_target_reachability"] = value["mapped_target_bonds"] / max(value["target_bonds"], 1)
                    value["target_pair_reachability"] = value["reachable_target_bonds"] / max(value["target_bonds"], 1)
                    by_profile[name] = dict(value)
                cap_result[split] = {"aggregate": total, "by_profile": by_profile}
            result["train"][str(cap)] = cap_result["train"]
            result["holdout"][str(cap)] = cap_result["holdout"]
    finally:
        pair.MAX_PER_SIDE = old_cap
    report = {"schema_version": "chemvector-identity-aligned-endpoint-pair-capacity-r5-0.1", "complete": True,
              "training_performed": False, "threshold_swept": False, "sealed_test_read": False,
              "deployment_authorized": False, "production_checkpoint_written": False,
              "initial_sha256": pair.INITIAL_SHA, "candidate_artifact_sha256": pair.digest(pair.CANDIDATES),
              "device": str(device), "fixed_slice": {"train_images": len(selected_rows["train"]), "holdout_images": len(selected_rows["holdout"]), "caps": list(CAPS)},
              "results": result, "gate": {"target_independent_generation": True, "no_training": True, "no_holdout_threshold": True, "go": False,
              "next": "capacity diagnostic only; choose cap before any separate sidecar/integration review"}}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)); return report


if __name__ == "__main__":
    run()
