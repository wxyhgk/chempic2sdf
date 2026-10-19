"""Render the identity-disjoint valence/topology extension panel.

This is deliberately a quarantine-only operation.  It consumes only the
metadata manifest and the public curriculum catalog, writes no checkpoint,
and does not read the fixed holdout beyond an overlap check against its
metadata identities.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile, render  # noqa: E402
from Data.sdf.chemvector_curriculum_v0_2.catalog import entries  # noqa: E402

MANIFEST = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-valence-targeted-manifest-r5/manifest.json"
SOURCE = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-valence-extension-r5"
TARGET_PROFILES = {
    "cov13_base48": Profile("cov13_base48", font_size=32, bond_length_cap=32.0,
                             bond_line_width=3.2, rotation_degrees=180.0,
                             font_family="DejaVu Serif"),
    "cov13_cap52": Profile("cov13_cap52", font_size=32, bond_length_cap=52.0,
                            bond_line_width=3.2, rotation_degrees=180.0,
                            font_family="DejaVu Serif"),
    "cov13_stroke16_48": Profile("cov13_stroke16_48", font_size=32, bond_length_cap=32.0,
                                  bond_line_width=1.6, rotation_degrees=180.0,
                                  font_family="DejaVu Serif"),
}


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
    extension = manifest["catalog_extension"]
    if not extension["identity_disjoint_from_fixed_source"] or extension["rendered"]:
        raise ValueError("extension is not a fresh identity-disjoint metadata plan")
    source_rows = [json.loads(line) for line in (SOURCE / "samples.jsonl").read_text().splitlines() if line.strip()]
    source_train = {row["molecule_id"] for row in source_rows if row.get("split") == "train"}
    source_holdout = {row["molecule_id"] for row in source_rows if row.get("split") == "holdout"}
    selected = extension["items"]
    ids = {item["molecule_id"] for item in selected}
    if ids & (source_train | source_holdout):
        raise ValueError("extension overlaps fixed source identities")
    if len(selected) != 72 or extension["planned_images"] != 216:
        raise ValueError("unexpected extension accounting")
    by_id = {entry["molecule_id"]: entry for entry in entries()}
    OUT.mkdir(parents=True)
    (OUT / "samples").mkdir()
    accepted, rejected, hashes = [], [], {}
    for item in sorted(selected, key=lambda x: (x["molecule_id"], x["selection_stratum"])):
        entry = by_id.get(item["molecule_id"])
        if entry is None or entry["canonical_smiles"] != item["canonical_smiles"]:
            raise ValueError(f"catalog mismatch: {item['molecule_id']}")
        for profile_id, profile in TARGET_PROFILES.items():
            sample_key = item["molecule_id"] + "__" + profile_id
            row = {"sample_key": sample_key, "molecule_id": item["molecule_id"],
                   "canonical_smiles": item["canonical_smiles"], "connection_identity": item["connection_identity"],
                   "profile_id": profile_id, "split": "train", "selection_stratum": item["selection_stratum"],
                   "target_strata": item["tags"], "render_status": "accepted", "sha256": {}}
            try:
                files, meta = render(entry, profile, typography="stereo-aware-v1")
            except ValueError as exc:
                rejected.append({**row, "render_status": "quality_rejected", "reason": str(exc)})
                continue
            row["measured"] = meta["measured"]
            for suffix, payload in files.items():
                path = OUT / "samples" / (sample_key + suffix)
                path.write_bytes(payload)
                row["sha256"][suffix] = hashlib.sha256(payload).hexdigest()
                hashes[str(path.relative_to(OUT))] = row["sha256"][suffix]
            accepted.append(row)
    summary = {
        "schema_version": "chemvector-identity-aligned-line-first-valence-extension-r5-render-0.1",
        "complete": True, "status": "quarantine_train_only", "training_authorized": False,
        "gpu_used": False, "model_forwards": 0, "optimizer_steps": 0,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "manifest_sha256": digest(MANIFEST), "source_summary_sha256": digest(SOURCE / "summary.json"),
        "planned_images": 216, "accepted_images": len(accepted), "rejected_images": len(rejected),
        "accepted_by_profile": dict(Counter(row["profile_id"] for row in accepted)),
        "selection_counts": dict(Counter(row["selection_stratum"] for row in accepted)),
        "rejection_reasons": dict(Counter(row["reason"] for row in rejected)),
        "molecule_count": len({row["molecule_id"] for row in accepted}),
        "fixed_source_overlap": len({row["molecule_id"] for row in accepted} & (source_train | source_holdout)),
        "next": "CPU loader/coverage/line-first round-trip gate; no GPU training admission",
    }
    (OUT / "samples.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in accepted))
    (OUT / "rejected.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rejected))
    (OUT / "sample-sha256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
