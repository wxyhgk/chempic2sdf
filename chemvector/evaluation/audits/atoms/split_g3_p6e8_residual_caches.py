#!/usr/bin/env python3
"""Physically split P6-E8 caches and seal holdout files."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from chemvector.evaluation.audits.atoms import build_g3_p6e8_residual_cycle_data as p6e8
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json, _atomic_torch_save
from chemvector.g3_structural_data import file_sha256


OUTPUT = p6e8.OUTPUT / "splits"
AUDIT = p6e8.OUTPUT / "split_audit.json"
HARD_SHA256 = "c9495efb304212c850f448d255f4db96052ef93737e95fa71e4a4192dd88016b"
EASY_SHA256 = "4931a5e39ec6d3492a7e75920f20f8d62ea00a2ee2e95eb87acca7ded46972cf"
SPLITS = ("train", "calibration", "holdout")
TENSOR_FIELDS = ("image_crops", "score_crops", "candidate_offsets", "candidate_peak_logits", "candidate_mask", "offset_targets")
LIST_FIELDS = ("sample_key", "molecule_id", "scaffold", "split", "variant", "target_role", "degree_bucket", "target_index", "required_residual_cap")


def split_cache(cache: dict, split: str) -> dict:
    rows = torch.tensor([index for index, value in enumerate(cache["split"]) if value == split], dtype=torch.long)
    result = {name: cache[name][rows] for name in TENSOR_FIELDS}
    result.update({name: [cache[name][index] for index in rows.tolist()] for name in LIST_FIELDS})
    for name in ("p6e1_checkpoint_sha256", "eligible_manifest_sha256", "vision_checkpoint_sha256"):
        result[name] = cache[name]
    return result


def main() -> None:
    if OUTPUT.exists() or AUDIT.exists():
        raise RuntimeError("P6-E8 split cache output already exists")
    for path, expected in ((p6e8.HARD_CACHE, HARD_SHA256), (p6e8.EASY_CACHE, EASY_SHA256)):
        if file_sha256(path) != expected:
            raise ValueError(f"P6-E8 source cache SHA256 differs: {path}")
    hard = torch.load(p6e8.HARD_CACHE, map_location="cpu", weights_only=False)
    easy = torch.load(p6e8.EASY_CACHE, map_location="cpu", weights_only=False)
    OUTPUT.mkdir(parents=True)
    summary = {}
    for kind, cache in (("hard", hard), ("easy", easy)):
        summary[kind] = {}
        for split in SPLITS:
            payload = split_cache(cache, split)
            path = OUTPUT / f"{kind}_{split}.pt"
            _atomic_torch_save(path, payload)
            summary[kind][split] = {"rows": len(payload["offset_targets"]), "path": str(path.resolve()), "sha256": file_sha256(path)}
    for path in (OUTPUT / "hard_holdout.pt", OUTPUT / "easy_holdout.pt"):
        os.chmod(path, 0)
    audit = {"schema_version":"g3-p6e8-residual-cache-splits-0.1","status":"passed","summary":summary,"checks":{"hard_total_6552":sum(value["rows"] for value in summary["hard"].values())==6552,"easy_total_27504":sum(value["rows"] for value in summary["easy"].values())==27504,"holdout_mode_000":all((OUTPUT/name).stat().st_mode & 0o777 == 0 for name in ("hard_holdout.pt","easy_holdout.pt"))},"source":{"hard_sha256":HARD_SHA256,"easy_sha256":EASY_SHA256}}
    if not all(audit["checks"].values()):
        raise RuntimeError("P6-E8 split cache validation failed")
    _atomic_json(AUDIT, audit); print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
