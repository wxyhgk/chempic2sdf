"""Render the line-first cov13 targeted train-only panel.

This is a quarantine-only CPU operation.  It deliberately re-renders only the
fixed r5 train identities in the three weak cov13 profiles; no holdout rows,
model weights, or sealed identities are read.  The resulting artifact is a
diagnostic panel for endpoint/round-trip gates, not a training release.
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

SOURCE = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-train-r5"
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


def tags(graph: dict) -> list[str]:
    kinds = {str(b.get("chemical_type", "OTHER")).upper() for b in graph.get("bonds", [])}
    elements = {str(a.get("element", "OTHER")).upper() for a in graph.get("atoms", [])}
    degree = Counter()
    for bond in graph.get("bonds", []):
        degree[int(bond["begin_atom"])] += 1
        degree[int(bond["end_atom"])] += 1
    out = []
    if "AROMATIC" in kinds:
        out.append("aromatic_ring")
    if {"DOUBLE", "TRIPLE"} & kinds:
        out.append("multiple_bond")
    if any(value >= 3 for value in degree.values()):
        out.append("junction_degree3plus")
    if {"O", "N"} & elements:
        out.append("visible_heteroatom_candidate")
    return out


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    source_rows = [json.loads(line) for line in (SOURCE / "samples.jsonl").read_text().splitlines() if line.strip()]
    rows = [row for row in source_rows if row.get("split") == "train" and row.get("profile_id") in TARGET_PROFILES]
    if len(rows) != 116 * 3 or len({row["molecule_id"] for row in rows}) != 116:
        raise ValueError("unexpected fixed train target accounting")
    holdout_ids = {row["molecule_id"] for row in source_rows if row.get("split") == "holdout"}
    if holdout_ids & {row["molecule_id"] for row in rows}:
        raise ValueError("target panel overlaps fixed holdout identities")
    by_id = {entry["molecule_id"]: entry for entry in entries()}
    OUT.mkdir(parents=True)
    (OUT / "samples").mkdir()
    accepted, rejected, hashes = [], [], {}
    strata = Counter()
    for row in sorted(rows, key=lambda item: item["sample_key"]):
        entry = by_id.get(row["molecule_id"])
        profile = TARGET_PROFILES[row["profile_id"]]
        if entry is None:
            raise ValueError(f"missing catalog entry: {row['molecule_id']}")
        try:
            files, meta = render(entry, profile, typography="stereo-aware-v1")
        except ValueError as exc:
            rejected.append({**row, "render_status": "quality_rejected", "reason": str(exc)})
            continue
        graph = json.loads(files[".graph.json"])
        row_out = {**row, "render_status": "accepted", "measured": meta["measured"],
                   "target_strata": tags(graph), "sha256": {}}
        for suffix, payload in files.items():
            path = OUT / "samples" / (row["sample_key"] + suffix)
            path.write_bytes(payload)
            row_out["sha256"][suffix] = hashlib.sha256(payload).hexdigest()
            hashes[str(path.relative_to(OUT))] = row_out["sha256"][suffix]
        for tag in row_out["target_strata"]:
            strata[tag] += 1
        accepted.append(row_out)
    summary = {
        "schema_version": "chemvector-identity-aligned-line-first-targeted-train-r5-render-0.1",
        "complete": True, "status": "quarantine_train_only", "training_authorized": False,
        "gpu_used": False, "model_forwards": 0, "optimizer_steps": 0,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "source_summary_sha256": digest(SOURCE / "summary.json"),
        "planned_images": len(rows), "accepted_images": len(accepted), "rejected_images": len(rejected),
        "accepted_by_profile": dict(Counter(row["profile_id"] for row in accepted)),
        "rejection_reasons": dict(Counter(row["reason"] for row in rejected)),
        "target_strata_counts": dict(sorted(strata.items())),
        "molecule_count": len({row["molecule_id"] for row in accepted}),
        "holdout_overlap": len({row["molecule_id"] for row in accepted} & holdout_ids),
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
