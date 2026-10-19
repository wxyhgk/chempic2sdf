"""Prepare an identity-clean oa13/oa33 stress manifest for graph replay.

The known-clean selector sidecar was fit on the r17 m128 cache.  This helper
selects only release holdout molecules absent from every cache split, keeps
the oa13/oa33 profiles, and writes metadata only.  It does not load images,
labels, sealed test content, or model weights.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/manifest.json"
CACHE = ROOT / "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m128-t0p72-c1-panel128/feature-cache.pt"
KNOWN_ARTIFACT = ROOT / "outputs/audits/chemvector-r17-known-clean-graph-artifact-v1/graph-artifact.jsonl"
SLICE80 = ROOT / "outputs/audits/chemvector-r17-selector-graph-replay-10k-slice80-aromatic/report.json"
OUT = ROOT / "outputs/audits/chemvector-r17-oa13-oa33-known-clean-stress-manifest-v2"
PROFILES = ("oa13", "oa33")
COUNT = 16
SEED = "r17-known-clean-oa13-oa33-stress-v1:"


def digest(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    release = json.loads(RELEASE.read_text(encoding="utf-8"))["rows"]
    import torch
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    cached_ids = {str(row["sample_key"]).split("__", 1)[0]
                  for rows in cache["records"].values() for row in rows}
    artifact_ids = {str(json.loads(line)["sample_key"]).split("__", 1)[0]
                    for line in KNOWN_ARTIFACT.read_text().splitlines() if line.strip()}
    slice_ids = {str(row["sample_key"]).split("__", 1)[0]
                 for row in json.loads(SLICE80.read_text())["rows"]}
    excluded_ids = cached_ids | artifact_ids | slice_ids
    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in release:
        if row["split"] == "holdout" and row["profile_id"] in PROFILES:
            by_id[str(row["molecule_id"])].append(row)
    eligible = [rows for molecule_id, rows in by_id.items()
                if molecule_id not in excluded_ids and len(rows) == len(PROFILES)]
    eligible.sort(key=lambda rows: hashlib.sha256(
        (SEED + str(rows[0]["molecule_id"])).encode()).hexdigest())
    chosen = eligible[:COUNT]
    if len(chosen) != COUNT:
        raise ValueError(f"expected {COUNT} prior-artifact-disjoint identities, got {len(chosen)}")
    rows = sorted((row for molecule in chosen for row in molecule),
                  key=lambda row: row["sample_key"])
    ids = {str(row["molecule_id"]) for row in rows}
    if ids & excluded_ids or len(ids) != COUNT:
        raise RuntimeError("stress identities overlap a prior known-clean artifact")
    manifest = {"schema_version": "chemvector-r17-known-clean-oa13-oa33-stress-manifest-0.1",
                "rows": rows}
    report = {
        "schema_version": manifest["schema_version"], "complete": True,
        "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "release_manifest_sha256": digest(RELEASE), "cache_sha256": digest(CACHE),
        "known_artifact_sha256": digest(KNOWN_ARTIFACT), "slice80_sha256": digest(SLICE80),
        "profiles": list(PROFILES), "holdout_molecules": COUNT,
        "images": len(rows), "cache_seen_molecule_ids": len(cached_ids),
        "artifact_seen_molecule_ids": len(artifact_ids), "slice80_seen_molecule_ids": len(slice_ids),
        "excluded_molecule_ids": len(excluded_ids),
        "selected_molecule_ids": sorted(ids),
        "identity": {"cache_overlap": len(ids & cached_ids),
                     "artifact_overlap": len(ids & artifact_ids),
                     "slice80_overlap": len(ids & slice_ids), "split": "holdout",
                     "profile_complete": all(len(molecule) == len(PROFILES) for molecule in chosen)},
        "gate": {"metadata_only": True, "cache_disjoint": not bool(ids & cached_ids),
                 "artifact_disjoint": not bool(ids & artifact_ids),
                 "slice80_disjoint": not bool(ids & slice_ids),
                 "sealed_test_read": False, "training_admissible": False,
                 "next": "run frozen selector graph replay only; do not train or deploy"},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    report["manifest_sha256"] = digest(OUT / "manifest.json")
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "report": report}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
