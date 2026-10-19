#!/usr/bin/env python3
"""Read-only overlap audit for the release slice80 and r17 sidecar cohorts.

The release slice is public metadata.  This audit compares molecule IDs from
that manifest with (a) the r17 specialist plan and (b) the actually materialized
public r17 feature caches.  It does not read sealed tests, infer labels, or
write any model state.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SLICE_REPORT = ROOT / "outputs/audits/chemvector-r17-selector-graph-replay-10k-slice80-aromatic/report.json"
RELEASE_MANIFEST = ROOT / "outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/manifest.json"
R17_PLAN = ROOT / "outputs/audits/chemvector-aromatic-specialist-plan-r17-v2/rows.jsonl"
MOLECULES = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/molecules.json"
OUT = ROOT / os.environ.get("R17_SLICE80_OVERLAP_OUT", "outputs/audits/chemvector-r17-release-slice80-identity-overlap-v1")
CACHE_GLOB = "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-*/feature-cache.pt"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pair_hash(values: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def matrix(left: dict[str, set[str]], right: dict[str, set[str]]) -> dict[str, dict[str, int]]:
    return {a: {b: len(xs & right[b]) for b in right} for a, xs in left.items()}


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    slice_report = json.loads(SLICE_REPORT.read_text())
    manifest = json.loads(RELEASE_MANIFEST.read_text())
    molecules = {str(row["molecule_id"]): row for row in json.loads(MOLECULES.read_text())}
    release_rows = {row["sample_key"]: row for row in manifest["rows"]}

    # slice80 rows are already public replay metadata; resolve IDs through the
    # release manifest rather than deriving identity from a filename.
    release: dict[str, set[str]] = defaultdict(set)
    missing_manifest = []
    for row in slice_report["rows"]:
        key = row["sample_key"]
        source = release_rows.get(key)
        if source is None:
            missing_manifest.append(key)
            continue
        split = "holdout" if row["split"] == "fresh_holdout" else row["split"]
        release[split].add(str(source["molecule_id"]))

    planned: dict[str, set[str]] = defaultdict(set)
    for line in R17_PLAN.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        planned[str(row["split"])].add(str(row["molecule_id"]))

    # Materialized cache cohort: only identities present in actual feature
    # caches count here.  A plan row without a cache is not execution evidence.
    cached: dict[str, set[str]] = defaultdict(set)
    cache_files = sorted(ROOT.glob(CACHE_GLOB))
    cache_counts = {}
    for path in cache_files:
        data = torch.load(path, map_location="cpu", weights_only=False)
        local = defaultdict(set)
        for split, records in data.get("records", {}).items():
            for row in records:
                key = row.get("sample_key")
                if key:
                    local[split].add(str(key).rsplit("__", 1)[0])
                    cached[split].add(str(key).rsplit("__", 1)[0])
        cache_counts[str(path.relative_to(ROOT))] = {k: len(v) for k, v in local.items()}

    release_overlap = {split: len(release[split] & set().union(*planned.values())) for split in release}
    release_cached_overlap = {split: len(release[split] & set().union(*cached.values())) for split in release}
    identity_fields = ("canonical_smiles", "connection_identity_without_isotope_stereo", "scaffold")
    def field_sets(cohort: dict[str, set[str]]) -> dict[str, set[str]]:
        output = {field: set() for field in identity_fields}
        for ids in cohort.values():
            for molecule_id in ids:
                row = molecules.get(molecule_id)
                if row is None:
                    continue
                for field in identity_fields:
                    value = row.get(field)
                    if value:
                        output[field].add(str(value))
        return output
    release_fields = field_sets(release)
    planned_fields = field_sets(planned)
    cached_fields = field_sets(cached)
    release_fields_by = {split: field_sets({split: ids}) for split, ids in release.items()}
    planned_fields_by = {split: field_sets({split: ids}) for split, ids in planned.items()}
    cached_fields_by = {split: field_sets({split: ids}) for split, ids in cached.items()}
    field_overlap_plan = {field: len(release_fields[field] & planned_fields[field]) for field in identity_fields}
    field_overlap_cache = {field: len(release_fields[field] & cached_fields[field]) for field in identity_fields}
    field_matrix_plan = {
        split: {other: {field: len(release_fields_by[split][field] & planned_fields_by[other][field]) for field in identity_fields} for other in planned_fields_by}
        for split in release_fields_by
    }
    field_matrix_cache = {
        split: {other: {field: len(release_fields_by[split][field] & cached_fields_by[other][field]) for field in identity_fields} for other in cached_fields_by}
        for split in release_fields_by
    }
    def internally_disjoint(fields_by: dict[str, dict[str, set[str]]]) -> bool:
        names = sorted(fields_by)
        return all(not (fields_by[a][field] & fields_by[b][field])
                   for i, a in enumerate(names) for b in names[i + 1:]
                   for field in identity_fields)
    def cross_disjoint(left: dict[str, dict[str, set[str]]], right: dict[str, dict[str, set[str]]]) -> bool:
        return all(not (xs[field] & ys[field])
                   for xs in left.values() for ys in right.values() for field in identity_fields)
    gate = {
        "slice_manifest_complete": not missing_manifest and len(slice_report["rows"]) == 1280,
        "release_slice_internal_disjoint": not (release["train"] & release["holdout"]),
        "r17_plan_internal_disjoint": not (planned["train"] & planned["dev"] or planned["train"] & planned["fresh_holdout"] or planned["dev"] & planned["fresh_holdout"]),
        "cached_internal_disjoint": not (cached["train"] & cached["dev"] or cached["train"] & cached["fresh_holdout"] or cached["dev"] & cached["fresh_holdout"]),
        "release_slice_identity_fields_internal_disjoint": internally_disjoint(release_fields_by),
        "r17_plan_identity_fields_internal_disjoint": internally_disjoint(planned_fields_by),
        "cached_identity_fields_internal_disjoint": internally_disjoint(cached_fields_by),
        "release_slice_disjoint_from_r17_plan": not any(release_overlap.values()),
        "release_slice_disjoint_from_materialized_r17_cache": not any(release_cached_overlap.values()),
        "release_identity_fields_disjoint_from_r17_plan": cross_disjoint(release_fields_by, planned_fields_by),
        "release_identity_fields_disjoint_from_materialized_cache": cross_disjoint(release_fields_by, cached_fields_by),
        "production_go": False,
    }
    result = {
        "schema_version": "chemvector-r17-release-slice80-identity-overlap-0.1",
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "inputs": {
            "slice_report": str(SLICE_REPORT.relative_to(ROOT)),
            "slice_report_sha256": digest(SLICE_REPORT),
            "release_manifest": str(RELEASE_MANIFEST.relative_to(ROOT)),
            "release_manifest_sha256": digest(RELEASE_MANIFEST),
            "r17_plan": str(R17_PLAN.relative_to(ROOT)),
            "r17_plan_sha256": digest(R17_PLAN),
            "molecules_sha256": digest(MOLECULES),
            "cache_files": cache_counts,
        },
        "cohorts": {
            "release_slice80": {k: {"molecules": len(v), "id_set_sha256": pair_hash(v)} for k, v in release.items()},
            "r17_plan": {k: {"molecules": len(v), "id_set_sha256": pair_hash(v)} for k, v in planned.items()},
            "r17_materialized_cache": {k: {"molecules": len(v), "id_set_sha256": pair_hash(v)} for k, v in cached.items()},
        },
        "overlap_matrix_release_vs_r17_plan": matrix(release, planned),
        "overlap_matrix_release_vs_materialized_cache": matrix(release, cached),
        "release_overlap_union_r17_plan": release_overlap,
        "release_overlap_union_materialized_cache": release_cached_overlap,
        "identity_field_overlap_union_r17_plan": field_overlap_plan,
        "identity_field_overlap_union_materialized_cache": field_overlap_cache,
        "identity_field_overlap_matrix_r17_plan": field_matrix_plan,
        "identity_field_overlap_matrix_materialized_cache": field_matrix_cache,
        "missing_manifest_sample_keys": len(missing_manifest),
        "gate": gate,
        "interpretation": "slice80_reuses_r17_identities" if any(release_cached_overlap.values()) else "slice80_identity_disjoint_from_materialized_cache",
    }
    OUT.mkdir(parents=False)
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "release_overlap_union_r17_plan": release_overlap, "release_overlap_union_materialized_cache": release_cached_overlap, "gate": gate}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
