"""Parallel quarantine renderer for the fixed 6k targeted manifest."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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
OUT = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-6k-r5-parallel"


def _job(payload):
    item, style, entry_data, out_root = payload
    entry = {"molecule_id": entry_data["molecule_id"], "canonical_smiles": entry_data["canonical_smiles"]}
    profile = Profile(profile_id=style["profile_id"], font_size=int(style["font_size"]),
                      bond_length_cap=float(style["bond_length_cap"]),
                      bond_line_width=float(style["bond_line_width"]),
                      rotation_degrees=float(style["rotation_degrees"]), font_family="DejaVu Serif")
    sample_key = item["molecule_id"] + "__" + style["profile_id"]
    row = {"sample_key": sample_key, "molecule_id": item["molecule_id"],
           "canonical_smiles": item["canonical_smiles"],
           "connection_identity": item["connection_identity"],
           "profile_id": style["profile_id"], "split": item["split"],
           "selection_stratum": item["selection_stratum"], "target_strata": item["tags"],
           "render_status": "accepted", "sha256": {}}
    try:
        files, meta = render(entry, profile, typography="stereo-aware-v1")
    except ValueError as exc:
        row["render_status"] = "quality_rejected"
        row["reason"] = str(exc)
        return row
    row["measured"] = meta["measured"]
    sample_dir = Path(out_root) / "samples"
    for suffix, data in files.items():
        path = sample_dir / (sample_key + suffix)
        path.write_bytes(data)
        row["sha256"][suffix] = hashlib.sha256(data).hexdigest()
    return row


def main(workers: int = 16) -> dict:
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
    jobs = []
    for item in items:
        entry = by_id.get(item["molecule_id"])
        if entry is None or entry["canonical_smiles"] != item["canonical_smiles"]:
            raise ValueError(f"catalog mismatch: {item['molecule_id']}")
        entry_data = {"molecule_id": entry["molecule_id"], "canonical_smiles": entry["canonical_smiles"]}
        jobs.extend((item, style, entry_data, str(OUT)) for style in manifest["styles"])
    accepted, rejected = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, payload) for payload in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row["render_status"] == "accepted":
                accepted.append(row)
            else:
                rejected.append(row)
            if index % 600 == 0:
                print(json.dumps({"completed_jobs": index, "planned_jobs": len(jobs),
                                  "accepted": len(accepted), "rejected": len(rejected)}), flush=True)
    accepted.sort(key=lambda row: row["sample_key"])
    rejected.sort(key=lambda row: row["sample_key"])
    file_hashes = {str(path.relative_to(OUT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (OUT / "samples").iterdir() if path.is_file()}
    summary = {
        "schema_version": "chemvector-identity-aligned-line-first-targeted-6k-r5-render-parallel-0.1",
        "complete": True, "status": "quarantine_train_dev_fresh_holdout",
        "training_authorized": False, "gpu_used": False, "model_forwards": 0,
        "optimizer_steps": 0, "sealed_test_read": False,
        "production_checkpoint_written": False,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "source_summary_sha256": hashlib.sha256((SOURCE / "summary.json").read_bytes()).hexdigest(),
        "planned_images": len(jobs), "accepted_images": len(accepted), "rejected_images": len(rejected),
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
