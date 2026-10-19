"""Validate the candidate-level cache emitted by the frozen replay.

The replay process can be memory-heavy because it keeps aggregate rows while
streaming a large JSONL cache.  This audit is deliberately a second, read-only
pass over the JSONL: it validates every record, reconstructs split/style
strata, and binds the result to the already completed aggregate replay.  It
does not load a checkpoint, images, sealed data, or run training.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
AGGREGATE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r1/report.json"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/cache-audit.json"
CORE = ("coordgen", "original", "rdkit")
LABELS = ("strict_positive", "protected", "near_unknown", "hard_negative", "selected_unmatched")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not CACHE.is_file() or not AGGREGATE.is_file():
        raise FileNotFoundError("candidate cache and r1 aggregate are required")
    expected = json.loads(AGGREGATE.read_text(encoding="utf-8"))["by_split_variant"]
    counts: dict[str, Counter] = defaultdict(Counter)
    samples: dict[str, set[str]] = defaultdict(set)
    rows = 0
    with CACHE.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            row = json.loads(line)
            required = ("sample_key", "molecule_id", "replay_split", "variant", "profile",
                        "slot", "label", "raw_score", "presence_logit", "x_norm", "y_norm",
                        "x_px", "y_px", "nearest_distance_px")
            if any(name not in row for name in required):
                raise ValueError(f"missing field at line {line_number}")
            split, variant, label = row["replay_split"], row["variant"], row["label"]
            if split not in ("train_core", "dev_core", "dev_stress"):
                raise ValueError(f"unexpected split at line {line_number}: {split}")
            if label not in LABELS:
                raise ValueError(f"unexpected label at line {line_number}: {label}")
            if split != "dev_stress" and variant not in CORE:
                raise ValueError(f"unexpected core variant at line {line_number}: {variant}")
            if split == "dev_stress" and variant != "stress":
                raise ValueError(f"unexpected stress variant at line {line_number}")
            if not 0.0 <= float(row["x_norm"]) <= 1.0 or not 0.0 <= float(row["y_norm"]) <= 1.0:
                raise ValueError(f"candidate coordinate outside image at line {line_number}")
            if float(row["nearest_distance_px"]) < 0.0:
                raise ValueError(f"negative distance at line {line_number}")
            counts[split, variant][label] += 1
            samples[split].add(row["sample_key"])
            rows += 1

    reconstructed = {split: {variant: dict(counter) for (s, variant), counter in counts.items() if s == split}
                     for split in ("train_core", "dev_core", "dev_stress")}
    # The cache must be exactly the same forward replay as r1, not a silently
    # relabelled or partially written second dataset.
    for split, variants in expected.items():
        for variant, aggregate in variants.items():
            got = counts.get((split, variant), Counter())
            mapped = {
                "strict_positive": int(aggregate["strict_positive"]),
                "protected": int(aggregate["protected"]),
                "near_unknown": int(aggregate["near_unknown"]),
                "hard_negative": int(aggregate["hard_negative"]),
                "selected_unmatched": int(aggregate["selected_unmatched"]),
            }
            if any(got[name] != value for name, value in mapped.items()):
                raise ValueError(f"cache aggregate mismatch: {split}/{variant}: {dict(got)} != {mapped}")

    result = {
        "schema_version": "chemvector-style-factorial-candidate-cache-audit-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": False, "checkpoint_read": False, "sealed_test_read": False,
        "cache_sha256": digest(CACHE), "aggregate_sha256": digest(AGGREGATE),
        "cache_rows": rows, "samples_by_split": {k: len(v) for k, v in sorted(samples.items())},
        "by_split_variant": reconstructed,
        "gate": {
            "jsonl_complete_and_valid": True,
            "matches_forward_aggregate": True,
            "near_unknown_present_and_separate": any(counts[k]["near_unknown"] for k in counts),
            "stress_separate": "stress" in reconstructed.get("dev_stress", {}),
            "training_admissible": False,
            "next": "fit one fixed CPU gate on train_core strict positives plus hard negatives; report each core variant and stress separately",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "cache_rows": rows,
                      "samples_by_split": result["samples_by_split"], "gate": result["gate"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
