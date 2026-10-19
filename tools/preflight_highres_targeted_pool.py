"""Read-only deterministic pool preflight for the high-resolution shadow.

This command only validates the already-admitted 12,985 train-row manifest
returned by :func:`source_checks` and writes two identity manifests (400 and
800 rows).  It does not load a model, open image bytes, read validation/sealed
media, or authorize training/deployment.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from chemvector.training.curriculum.pair_scale_training import source_checks


ROOT = Path(__file__).resolve().parents[1]
FIXED_REPORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_HIGHRES_POOL_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-highres-targeted-pool-preflight-20261006"),
))
EXPECTED_ROWS = 12_985
PROFILE_ORDER = (
    "font_large", "serif", "line_thick", "font_small", "reference",
    "bond_short", "rotated", "line_thin", "bond_long",
)
TARGET_SIZES = (400, 800)


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _quota(total: int) -> dict[str, int]:
    if total == 400:
        main, rest = 100, 100
    elif total == 800:
        main, rest = 200, 200
    else:
        raise ValueError(f"unsupported pool size: {total}")
    targeted = ("font_large", "serif", "line_thick")
    others = tuple(profile for profile in PROFILE_ORDER if profile not in targeted)
    result = {profile: main for profile in targeted}
    base, remainder = divmod(rest, len(others))
    for index, profile in enumerate(others):
        result[profile] = base + (index < remainder)
    if sum(result.values()) != total:
        raise AssertionError("quota total mismatch")
    return result


def _identity(row: dict) -> dict:
    # Keep enough provenance to replay the exact row, without copying image or
    # graph payloads into this preflight artifact.
    fields = (
        "sample_key", "molecule_id", "profile_id", "split", "source",
        "pixel_sha256", "png_sha256", "graph_sha256", "meta_sha256",
    )
    return {field: row.get(field) for field in fields}


def _pool(rows: list[dict], size: int) -> tuple[dict[str, list[dict]], dict[str, int]]:
    quota = _quota(size)
    grouped: dict[str, list[dict]] = {profile: [] for profile in PROFILE_ORDER}
    for row in rows:
        grouped.setdefault(str(row["profile_id"]), []).append(row)
    selected: dict[str, list[dict]] = {}
    for profile in PROFILE_ORDER:
        ranked = sorted(grouped[profile], key=lambda row: hashlib.sha256(
            str(row["sample_key"]).encode("utf-8")).hexdigest())
        if len(ranked) < quota[profile]:
            raise ValueError(f"profile {profile} has {len(ranked)} rows, need {quota[profile]}")
        selected[profile] = [_identity(row) for row in ranked[:quota[profile]]]
    return selected, quota


def _flatten(selected: dict[str, list[dict]]) -> list[dict]:
    rows = [row for profile in PROFILE_ORDER for row in selected[profile]]
    rows.sort(key=lambda row: row["sample_key"])
    if len({row["sample_key"] for row in rows}) != len(rows):
        raise ValueError("duplicate sample_key in selected pool")
    return rows


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    sources = source_checks()
    fixed = json.loads(FIXED_REPORT.read_text())
    fixed_keys = {row["sample_key"] for row in fixed["per_sample"]}
    rows = [row for row in sources["rows"] if row["sample_key"] not in fixed_keys]
    if len(rows) != EXPECTED_ROWS - len(fixed_keys) or any(row.get("split") != "train" for row in rows):
        raise ValueError("source_checks did not return the expected train rows after fixed-cohort exclusion")
    profiles = Counter(str(row["profile_id"]) for row in rows)
    if set(profiles) != set(PROFILE_ORDER):
        raise ValueError(f"unexpected profiles: {sorted(profiles)}")
    pools = {}
    for size in TARGET_SIZES:
        selected, quota = _pool(rows, size)
        flattened = _flatten(selected)
        if len(flattened) != size:
            raise AssertionError("selected pool size mismatch")
        pools[str(size)] = {
            "size": size,
            "quota": quota,
            "rows": flattened,
            "rows_sha256": _digest(flattened),
            "profile_counts": dict(Counter(row["profile_id"] for row in flattened)),
        }
    # Nestedness is useful for a bounded follow-up: the 400 pool must be the
    # deterministic prefix of the 800 pool within each profile.
    nested = bool(pools["400"]["rows"]) and {
        row["sample_key"] for row in pools["400"]["rows"]
    }.issubset({row["sample_key"] for row in pools["800"]["rows"]})
    result = {
        "schema_version": "chemvector-highres-targeted-pool-preflight-0.1",
        "complete": True,
        "read_only": True,
        "model_loaded": False,
        "image_bytes_read": False,
        "sealed_test_read": False,
        "training_performed": False,
        "deployment_authorized": False,
        "source_rows": len(rows),
        "excluded_fixed_cohort": len(fixed_keys),
        "source_rows_sha256": _digest(rows),
        "source_preparation_report_sha256": sources["preparation_report_sha256"],
        "available_profile_counts": dict(profiles),
        "pool_nested_400_in_800": nested,
        "pools": {size: {key: value for key, value in pool.items() if key != "rows"}
                  for size, pool in pools.items()},
    }
    OUTPUT.mkdir(parents=True)
    for size, pool in pools.items():
        (OUTPUT / f"manifest-{size}.json").write_text(
            json.dumps({"schema_version": result["schema_version"], **pool}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "source_rows": len(rows),
                      "profile_counts": dict(profiles), "pools": result["pools"],
                      "nested": nested}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
