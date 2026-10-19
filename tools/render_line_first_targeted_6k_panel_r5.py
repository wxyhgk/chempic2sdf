"""Render the fixed 6k targeted manifest into a quarantine dataset.

The renderer consumes only the metadata manifest and the public curriculum
catalog.  It never reads sealed test content, runs a model, or writes a
checkpoint.  Completion is marked by ``summary.json`` only after every file
hash and row has been written.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile, render
from Data.sdf.chemvector_curriculum_v0_2.catalog import entries

MANIFEST = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-targeted-6k-manifest-r5/manifest.json"
SOURCE = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-6k-r5"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads(MANIFEST.read_text())
    if not manifest.get("gate", {}).get("identity_disjoint") or manifest.get("planned_images") != 6000:
        raise ValueError("manifest gate/accounting failed")
    source_rows = [json.loads(line) for line in (SOURCE / "samples.jsonl").read_text().splitlines() if line.strip()]
    source_ids = {str(row.get("molecule_id")) for row in source_rows}
    by_id = {entry["molecule_id"]: entry for entry in entries()}
    items = manifest["items"]
    if len(items) != 1000 or source_ids & {item["molecule_id"] for item in items}:
        raise ValueError("fixed source overlap or item count mismatch")
    OUT.mkdir(parents=True)
    (OUT / "samples").mkdir()
    accepted, rejected, file_hashes = [], [], {}
    profiles = {}
    for style in manifest["styles"]:
        profiles[style["profile_id"]] = Profile(
            profile_id=style["profile_id"], font_size=int(style["font_size"]),
            bond_length_cap=float(style["bond_length_cap"]),
            bond_line_width=float(style["bond_line_width"]),
            rotation_degrees=float(style["rotation_degrees"]),
            font_family="DejaVu Serif",
        )
    for item in items:
        entry = by_id.get(item["molecule_id"])
        if entry is None or entry["canonical_smiles"] != item["canonical_smiles"]:
            raise ValueError(f"catalog mismatch: {item['molecule_id']}")
        for profile_id, profile in profiles.items():
            sample_key = item["molecule_id"] + "__" + profile_id
            row = {
                "sample_key": sample_key,
                "molecule_id": item["molecule_id"],
                "canonical_smiles": item["canonical_smiles"],
                "connection_identity": item["connection_identity"],
                "profile_id": profile_id,
                "split": item["split"],
                "selection_stratum": item["selection_stratum"],
                "target_strata": item["tags"],
                "render_status": "accepted",
                "sha256": {},
            }
            try:
                files, meta = render(entry, profile, typography="stereo-aware-v1")
            except ValueError as exc:
                rejected.append({**row, "render_status": "quality_rejected", "reason": str(exc)})
                continue
            row["measured"] = meta["measured"]
            for suffix, payload in files.items():
                path = OUT / "samples" / (sample_key + suffix)
                path.write_bytes(payload)
                file_hash = hashlib.sha256(payload).hexdigest()
                row["sha256"][suffix] = file_hash
                file_hashes[str(path.relative_to(OUT))] = file_hash
            accepted.append(row)
        if len(accepted) and len(accepted) % 600 == 0:
            print(json.dumps({"accepted_images": len(accepted), "rejected_images": len(rejected)}), flush=True)
    planned = len(items) * len(profiles)
    summary = {
        "schema_version": "chemvector-identity-aligned-line-first-targeted-6k-r5-render-0.1",
        "complete": True,
        "status": "quarantine_train_dev_fresh_holdout",
        "training_authorized": False,
        "gpu_used": False,
        "model_forwards": 0,
        "optimizer_steps": 0,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "manifest_sha256": digest(MANIFEST),
        "source_summary_sha256": digest(SOURCE / "summary.json"),
        "planned_images": planned,
        "accepted_images": len(accepted),
        "rejected_images": len(rejected),
        "accepted_by_split": dict(Counter(row["split"] for row in accepted)),
        "accepted_by_profile": dict(Counter(row["profile_id"] for row in accepted)),
        "selection_counts": dict(Counter(row["selection_stratum"] for row in accepted)),
        "molecule_count": len({row["molecule_id"] for row in accepted}),
        "source_molecule_overlap": len({row["molecule_id"] for row in accepted} & source_ids),
        "rejection_reasons": dict(Counter(row["reason"] for row in rejected)),
        "verified_sample_file_hashes": len(file_hashes),
        "next": "run exact-byte/style/roundtrip and endpoint reachability audits; no training admission",
    }
    (OUT / "samples.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in accepted))
    (OUT / "rejected.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rejected))
    (OUT / "sample-sha256.json").write_text(json.dumps(file_hashes, indent=2, sort_keys=True) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    main()
