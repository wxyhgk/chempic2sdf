"""CPU-only rank/hard-negative preflight for the visual group cache.

This is a read-only admission diagnostic.  It does not load ``initial.pt``,
open image/layer/graph files, run an optimizer, or read sealed identities.  A
strict-positive group is a raw candidate rejected by the frozen selector;
confirmed-negative groups contain only raw-negative, selector-rejected
candidates.  The score is the frozen atom score in the released cache.  The
purpose is to establish whether a rank-aware objective has a meaningful
hard-negative signal before authorizing another GPU shadow.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_VISUAL_RANK_PREFLIGHT_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-targeted-selector-visual-pool-r2/report.json"),
))
TOP_K = (32, 64, 128, 256, 512)
HARD_NEGATIVES_PER_POSITIVE = 8


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _groups(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"groups", "purpose"}:
        raise ValueError(f"Unexpected cache schema: {path}")
    return payload["groups"]


def _rows(groups: list[dict]) -> tuple[list[dict], list[dict]]:
    positives: list[dict] = []
    negatives: list[dict] = []
    for group in groups:
        if group.get("protected"):
            continue
        members = group.get("members", [])
        if not members:
            raise ValueError("Empty unprotected group")
        if group.get("strict_positive"):
            # Keep only candidates that are definitely positive under the
            # released raw/selected contract; unknown members are excluded.
            scores = [float(m["feature"][0]) for m in members
                      if int(m["raw_target"]) >= 0 and int(m["selected_target"]) < 0]
            if not scores:
                raise ValueError("Strict-positive group has no strict member")
            positives.append({"score": max(scores), "group": group})
        elif group.get("confirmed_negative"):
            if any(int(m["raw_target"]) >= 0 or int(m["selected_target"]) >= 0
                   for m in members):
                raise ValueError("Confirmed negative contains a positive/selected member")
            negatives.append({"score": max(float(m["feature"][0]) for m in members),
                              "group": group})
        else:
            raise ValueError("Unprotected group is neither strict-positive nor confirmed-negative")
    return positives, negatives


def _auc(positives: list[dict], negatives: list[dict]) -> float | None:
    if not positives or not negatives:
        return None
    wins = ties = 0
    for positive in positives:
        score = positive["score"]
        for negative in negatives:
            if score > negative["score"]:
                wins += 1
            elif score == negative["score"]:
                ties += 1
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def _top_k(positives: list[dict], negatives: list[dict]) -> dict[str, dict]:
    rows = sorted(
        [(float(item["score"]), 1) for item in positives]
        + [(float(item["score"]), 0) for item in negatives],
        key=lambda item: item[0], reverse=True,
    )
    result = {}
    for k in TOP_K:
        chosen = rows[: min(k, len(rows))]
        result[str(k)] = {
            "selected": len(chosen),
            "strict_positive": sum(label for _, label in chosen),
            "precision": sum(label for _, label in chosen) / max(len(chosen), 1),
        }
    return result


def _hard_negative(positives: list[dict], negatives: list[dict]) -> dict:
    by_image: dict[str, list[float]] = defaultdict(list)
    for row in negatives:
        by_image[row["group"]["sample_key"]].append(row["score"])
    pair_count = violations = 0
    ranks = []
    for positive in positives:
        score = positive["score"]
        image_negatives = sorted(by_image[positive["group"]["sample_key"]], reverse=True)
        hard = image_negatives[:HARD_NEGATIVES_PER_POSITIVE]
        pair_count += len(hard)
        violations += sum(negative >= score for negative in hard)
        ranks.append(1 + sum(negative > score for negative in image_negatives))
    return {
        "per_positive_cap": HARD_NEGATIVES_PER_POSITIVE,
        "pairs": pair_count,
        "violations": violations,
        "violation_rate": violations / max(pair_count, 1),
        "positive_count_with_image_negatives": len(ranks),
        "median_positive_rank": median(ranks) if ranks else None,
        "top1_fraction": sum(rank <= 1 for rank in ranks) / max(len(ranks), 1),
        "top8_fraction": sum(rank <= 8 for rank in ranks) / max(len(ranks), 1),
    }


def _split(path: Path) -> dict:
    positives, negatives = _rows(_groups(path))
    return {
        "groups": len(positives) + len(negatives),
        "strict_positive_groups": len(positives),
        "confirmed_negative_groups": len(negatives),
        "pairwise_auc": _auc(positives, negatives),
        "top_k": _top_k(positives, negatives),
        "hard_negative": _hard_negative(positives, negatives),
    }


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    manifest = RELEASE / "manifest.json"
    train_cache = RELEASE / "train-group-cache.json"
    holdout_cache = RELEASE / "holdout-group-cache.json"
    for path in (manifest, train_cache, holdout_cache):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_payload.get("sealed_test_read") is not False:
        raise ValueError("Release manifest is not sealed-safe")
    if manifest_payload.get("training_run_authorized") is not False:
        raise ValueError("This preflight requires an un-authorized release")
    train_groups = _groups(train_cache)
    holdout_groups = _groups(holdout_cache)
    train_ids = {
        "sample_key": {g["sample_key"] for g in train_groups},
        "molecule_id": {g["molecule_id"] for g in train_groups},
    }
    holdout_ids = {
        "sample_key": {g["sample_key"] for g in holdout_groups},
        "molecule_id": {g["molecule_id"] for g in holdout_groups},
    }
    identity_overlap = {
        key: sorted(train_ids[key] & holdout_ids[key])
        for key in train_ids
    }
    if any(identity_overlap.values()):
        raise ValueError(f"Train/holdout identity overlap: {identity_overlap}")
    result = {
        "schema_version": "chemvector-targeted-selector-visual-pool-rank-preflight-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "checkpoint_read": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "source": {
            "release": str(RELEASE.relative_to(ROOT)),
            "manifest_sha256": digest(manifest),
            "train_cache_sha256": digest(train_cache),
            "holdout_cache_sha256": digest(holdout_cache),
            "score": "max frozen atom-score feature[0] per group; no image/GT feature added",
        },
        "splits": {"train": _split(train_cache), "holdout": _split(holdout_cache)},
        "identity_split": {
            "train_groups": len(train_groups),
            "holdout_groups": len(holdout_groups),
            "sample_key_overlap": len(identity_overlap["sample_key"]),
            "molecule_id_overlap": len(identity_overlap["molecule_id"]),
        },
        "contract": {
            "protected_groups_excluded": True,
            "unknown_members_excluded": True,
            "hard_negatives_same_image_only": True,
            "hard_negative_cap": HARD_NEGATIVES_PER_POSITIVE,
            "baseline_prefix_replayed": True,
            "sealed_access": False,
        },
        "gate": {
            "identity_split_from_release": all(not values for values in identity_overlap.values()),
            "train_positive_min300": True,
            "holdout_positive_min100": True,
            "train_hard_pairs_min1000": True,
            "holdout_hard_pairs_min300": True,
            "holdout_auc_min080": False,
            "holdout_top128_precision_min085": False,
            "training_ready": False,
            "reason": "step0 hard-negative ranking signal is not precision-admissible; no GPU authorization",
        },
    }
    # The positive-count and pair-count checks are expressed explicitly rather
    # than inferred from future reports, so a changed cache fails closed.
    train = result["splits"]["train"]
    holdout = result["splits"]["holdout"]
    result["gate"]["train_positive_min300"] = train["strict_positive_groups"] >= 300
    result["gate"]["holdout_positive_min100"] = holdout["strict_positive_groups"] >= 100
    result["gate"]["train_hard_pairs_min1000"] = train["hard_negative"]["pairs"] >= 1000
    result["gate"]["holdout_hard_pairs_min300"] = holdout["hard_negative"]["pairs"] >= 300
    result["gate"]["holdout_auc_min080"] = bool(holdout["pairwise_auc"] is not None and holdout["pairwise_auc"] >= 0.80)
    result["gate"]["holdout_top128_precision_min085"] = holdout["top_k"]["128"]["precision"] >= 0.85
    result["gate"]["training_ready"] = all(result["gate"][key] for key in (
        "identity_split_from_release", "train_positive_min300", "holdout_positive_min100",
        "train_hard_pairs_min1000", "holdout_hard_pairs_min300", "holdout_auc_min080",
        "holdout_top128_precision_min085",
    ))
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
