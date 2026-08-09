#!/usr/bin/env python3
"""Build target-labelled pair indices without copying P6 source rasters."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import tempfile

import torch

from chemvector.components.recovery.pairwise_structural_counterfactual import (
    IMAGE_SIZE,
    PAIR_EQUIVALENT,
    PAIR_HARMFUL,
    PAIR_RESCUE,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9


AUDIT_NAME = "chemvector-g3-p7e0-pairwise-counterfactual-data"
OUTPUT = Path("outputs/audits") / AUDIT_NAME
TRAIN_PAIRS = OUTPUT / "train_pairs.pt"
CALIBRATION_PAIRS = OUTPUT / "calibration_pairs.pt"
SCHEMA_VERSION = "g3-p7e0-pairwise-counterfactual-indices-0.1"
SOURCE_HARD = 0
SOURCE_EASY = 1


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def classify_pair(prior_distance: float, alternative_distance: float) -> int | None:
    prior_good = prior_distance <= 6.0
    alternative_good = alternative_distance <= 6.0
    if not prior_good and alternative_good:
        return PAIR_RESCUE
    if prior_good and not alternative_good:
        return PAIR_HARMFUL
    if prior_good and alternative_good:
        return PAIR_EQUIVALENT
    return None


def build_pair_rows(cache: dict, source_kind: int) -> tuple[dict, Counter]:
    offsets = cache["candidate_offsets"].float()
    targets = cache["offset_targets"].float()
    masks = cache["candidate_mask"]
    logits = cache["candidate_peak_logits"].float().masked_fill(~masks, -torch.inf)
    prior_indices = logits.argmax(dim=1)
    distances = torch.linalg.vector_norm(
        (offsets - targets[:, None]) * IMAGE_SIZE, dim=-1
    )
    rows: dict[str, list] = {
        "source_kind": [],
        "source_row": [],
        "prior_index": [],
        "alternative_index": [],
        "label": [],
        "scaffold": [],
        "variant": [],
        "target_role": [],
        "degree_bucket": [],
        "pair_distance_bucket": [],
    }
    counts: Counter = Counter()
    for row in range(len(offsets)):
        prior = int(prior_indices[row])
        for alternative in torch.where(masks[row])[0].tolist():
            if alternative == prior:
                continue
            label = classify_pair(
                float(distances[row, prior]), float(distances[row, alternative])
            )
            counts["all_non_prior"] += 1
            if label is None:
                counts["both_bad_excluded"] += 1
                continue
            separation = float(
                torch.linalg.vector_norm(
                    (offsets[row, alternative] - offsets[row, prior]) * IMAGE_SIZE
                )
            )
            rows["source_kind"].append(source_kind)
            rows["source_row"].append(row)
            rows["prior_index"].append(prior)
            rows["alternative_index"].append(alternative)
            rows["label"].append(label)
            rows["scaffold"].append(str(cache["scaffold"][row]))
            rows["variant"].append(str(cache["variant"][row]))
            rows["target_role"].append(str(cache["target_role"][row]))
            rows["degree_bucket"].append(str(cache["degree_bucket"][row]))
            rows["pair_distance_bucket"].append(int(separation // 4.0))
            counts[{PAIR_RESCUE: "rescue", PAIR_HARMFUL: "harmful", PAIR_EQUIVALENT: "equivalent"}[label]] += 1
    tensors = {
        name: torch.tensor(value, dtype=torch.long)
        for name, value in rows.items()
        if name in {
            "source_kind",
            "source_row",
            "prior_index",
            "alternative_index",
            "label",
            "pair_distance_bucket",
        }
    }
    tensors.update(
        {name: value for name, value in rows.items() if name not in tensors}
    )
    return tensors, counts


def _combine(first: dict, second: dict) -> dict:
    output = {}
    for name in first:
        if isinstance(first[name], torch.Tensor):
            output[name] = torch.cat((first[name], second[name]))
        else:
            output[name] = first[name] + second[name]
    return output


def _build_split(hard_path: Path, easy_path: Path, output: Path) -> dict:
    hard = torch.load(hard_path, map_location="cpu", weights_only=False)
    easy = torch.load(easy_path, map_location="cpu", weights_only=False)
    hard_rows, hard_counts = build_pair_rows(hard, SOURCE_HARD)
    easy_rows, easy_counts = build_pair_rows(easy, SOURCE_EASY)
    pairs = _combine(hard_rows, easy_rows)
    pairs.update(
        {
            "schema_version": SCHEMA_VERSION,
            "source_cache_sha256": {
                "hard": file_sha256(hard_path),
                "easy": file_sha256(easy_path),
            },
        }
    )
    _atomic_torch_save(output, pairs)
    return {
        "pairs": len(pairs["label"]),
        "hard_source": dict(sorted(hard_counts.items())),
        "easy_source": dict(sorted(easy_counts.items())),
        "labels": {
            "rescue": int((pairs["label"] == PAIR_RESCUE).sum()),
            "harmful": int((pairs["label"] == PAIR_HARMFUL).sum()),
            "equivalent": int((pairs["label"] == PAIR_EQUIVALENT).sum()),
        },
        "scaffolds": len(set(pairs["scaffold"])),
        "variants": dict(sorted(Counter(pairs["variant"]).items())),
        "roles": dict(sorted(Counter(pairs["target_role"]).items())),
        "output": str(output.resolve()),
        "sha256": file_sha256(output),
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P7-E0 output already exists: {OUTPUT}")
    for path, expected in p6e9.SHA256.items():
        if file_sha256(path) != expected:
            raise ValueError(f"P7-E0 source cache SHA256 differs: {path}")
    if any((path.stat().st_mode & 0o777) != 0 for path in p6e9.HOLDOUT_FILES):
        raise RuntimeError("P7-E0 holdout cache is not sealed")
    train = _build_split(p6e9.TRAIN_HARD, p6e9.TRAIN_EASY, TRAIN_PAIRS)
    calibration = _build_split(
        p6e9.CALIBRATION_HARD,
        p6e9.CALIBRATION_EASY,
        CALIBRATION_PAIRS,
    )
    train_scaffolds = set(
        torch.load(TRAIN_PAIRS, map_location="cpu", weights_only=False)["scaffold"]
    )
    calibration_scaffolds = set(
        torch.load(
            CALIBRATION_PAIRS, map_location="cpu", weights_only=False
        )["scaffold"]
    )
    checks = {
        "train_rescue_7376": train["labels"]["rescue"] == 7376,
        "train_harmful_136562": train["labels"]["harmful"] == 136562,
        "calibration_rescue_921": calibration["labels"]["rescue"] == 921,
        "calibration_harmful_17416": calibration["labels"]["harmful"] == 17416,
        "scaffold_overlap_zero": not (train_scaffolds & calibration_scaffolds),
        "holdout_mode_000": all(
            (path.stat().st_mode & 0o777) == 0 for path in p6e9.HOLDOUT_FILES
        ),
    }
    audit = {
        "schema_version": "g3-p7e0-pairwise-counterfactual-data-audit-0.1",
        "decision": "pairwise_counterfactual_data_go"
        if all(checks.values())
        else "pairwise_counterfactual_data_no_go",
        "checks": checks,
        "train": train,
        "calibration": calibration,
        "protocol": {
            "distance_threshold_pixels": 6.0,
            "both_good": "equivalent_zero_delta",
            "both_bad": "audit_only_excluded",
            "source_rasters_copied": False,
            "holdout_read": False,
        },
        "holdout_read": False,
        "dev_confirm_external_sealed_read": False,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
