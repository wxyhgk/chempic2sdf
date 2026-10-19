"""Finalize a completed candidate JSONL cache without another model replay."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r1/report.json"
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/report.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    by_sv: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    by_split: dict[str, Counter] = defaultdict(Counter)
    sample_keys: dict[str, set[str]] = defaultdict(set)
    lines = 0
    fields = None
    with CACHE.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            if fields is None:
                fields = sorted(row)
            if sorted(row) != fields:
                raise ValueError(f"cache schema changed at line {line_no}")
            split, variant, label = row["replay_split"], row["variant"], row["label"]
            by_sv[split][variant][label] += 1
            by_split[split][label] += 1
            sample_keys[split].add(row["sample_key"])
            lines += 1
    expected_total = sum(x["raw_candidates"] for groups in source["by_split_variant"].values() for x in groups.values())
    if lines != expected_total:
        raise ValueError(f"cache line count {lines} != frozen replay total {expected_total}")
    # Exact strata binding guards against truncation or accidental cache mixing.
    for split, groups in source["by_split_variant"].items():
        for variant, expected in groups.items():
            got = by_sv[split][variant]
            for label in ("strict_positive", "protected", "near_unknown", "hard_negative", "selected_unmatched"):
                if int(got[label]) != int(expected[label]):
                    raise ValueError(f"strata mismatch {split}/{variant}/{label}: {got[label]} != {expected[label]}")
    report = {**source,
              "candidate_cache": {"path": str(CACHE.relative_to(ROOT)), "sha256": digest(CACHE),
                                  "lines": lines, "fields": fields, "sample_counts": {k: len(v) for k, v in sample_keys.items()}},
              "by_split_variant": {s: {v: dict(c) for v, c in sorted(groups.items())}
                                   for s, groups in sorted(by_sv.items())},
              "by_split": {s: dict(c) for s, c in sorted(by_split.items())},
              "cache_finalized_without_replay": True,
              "cache_complete": True}
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "cache": report["candidate_cache"],
                      "by_split_variant": report["by_split_variant"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
