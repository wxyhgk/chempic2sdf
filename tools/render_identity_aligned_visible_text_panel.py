"""Render the reviewed O/N panel into a fresh quarantine dataset.

This is bounded CPU data construction only.  It never loads the model, reads
evaluation data, or changes training admission.  A known renderer-quality
ValueError is recorded as a rejection; unexpected exceptions abort the job.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import fields
import hashlib
import json
from pathlib import Path

from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile, render
from Data.sdf.chemvector_curriculum_v0_2.catalog import entries
from tools.prepare_identity_aligned_visible_text_panel import build_plan

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r4"
QUALITY = {
    "Overlapping glyph ink is outside the reference contract",
    "Overlapping text/geometry ink is outside the reference contract",
    "Recorded glyph touches canvas edge; possible clipping",
    "Recorded bond touches canvas edge; possible clipping",
    "Recorded auxiliary junction touches canvas edge; possible clipping",
    "Typography clearance would erase a bond segment",
    "Degenerate auxiliary junction",
}
STRICT_MOLECULES = False


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    # Build the plan from the currently frozen 120/80 constants.  Do not read
    # an older r1 plan: it used the superseded 180/100 split (1,680 images).
    plan = build_plan()
    if plan["selection"]["planned_images"] != 1200:
        raise ValueError("unexpected targeted panel size; expected 1,200 images")
    # The reviewed r1 plan contains a larger 180/100 quarantine selection.
    # Freeze a deterministic 120/80 subset for this bounded render so the
    # release-scale target is exactly 1,200 images.
    train_ids = sorted({r["molecule_id"] for r in plan["rows"] if r["split"] == "train"})[:120]
    holdout_ids = sorted({r["molecule_id"] for r in plan["rows"] if r["split"] == "holdout"})[:80]
    chosen_ids = set(train_ids) | set(holdout_ids)
    rows = [r for r in plan["rows"] if r["molecule_id"] in chosen_ids]
    plan["rows"] = rows
    plan["selection"]["train_molecules"] = len(train_ids)
    plan["selection"]["holdout_molecules"] = len(holdout_ids)
    plan["selection"]["planned_images"] = len(rows)
    by_id = {e["molecule_id"]: e for e in entries()}
    profile_by_id = {}
    for row in plan["selection"]["profiles"]:
        allowed = {f.name for f in fields(Profile)}
        profile_by_id[row["profile_id"]] = Profile(**{k: row[k] for k in allowed if k in row})
    if len(rows) != plan["selection"]["planned_images"] or len(profile_by_id) != 6:
        raise ValueError("panel plan accounting is invalid")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "samples").mkdir()
    plan_path = OUTPUT / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    accepted, rejected, sha = [], [], {}
    groups = defaultdict(list)
    for row in rows:
        groups[row["molecule_id"]].append(row)
    work = list(groups.values()) if STRICT_MOLECULES else [[row] for row in rows]
    processed = 0
    for group in work:
        staged = []
        failure = None
        for row in group:
            entry = by_id.get(row["molecule_id"])
            profile = profile_by_id.get(row["profile_id"])
            if entry is None or profile is None:
                raise ValueError("plan identity/profile is not in frozen source")
            try:
                files, meta = render(entry, profile, typography="stereo-aware-v1")
            except ValueError as exc:
                if str(exc) not in QUALITY:
                    raise
                failure = str(exc)
                break
            expected_key = row["sample_key"]
            if meta.get("sample_key") != expected_key:
                raise ValueError("renderer sample key differs from frozen plan")
            staged.append((row, files, meta))
        if failure is not None:
            for row in group:
                rejected.append({**row, "render_status": "quality_rejected",
                                 "reason": failure if not STRICT_MOLECULES else "molecule_quality_rejected: " + failure})
        else:
            for row, files, meta in staged:
                expected_key = row["sample_key"]
                for suffix, payload in files.items():
                    path = OUTPUT / "samples" / (expected_key + suffix)
                    path.write_bytes(payload)
                    sha[str(path.relative_to(OUTPUT))] = hashlib.sha256(payload).hexdigest()
                accepted.append({**row, "render_status": "accepted", "measured": meta["measured"],
                                 "sha256": meta["sha256"]})
        processed += len(group)
        if processed % 100 == 0 or processed == len(rows):
            print(json.dumps({"images": processed, "accepted": len(accepted), "rejected": len(rejected)}), flush=True)
    if len(accepted) + len(rejected) != len(rows):
        raise ValueError("render accounting is incomplete")
    summary = {
        "schema_version": "chemvector-identity-aligned-visible-text-panel-render-0.1",
        "complete": True, "status": "quarantine_cases_only", "training_authorized": False,
        "gpu_used": False, "model_forwards": 0, "optimizer_steps": 0,
        "sealed_test_read": False, "plan_sha256": digest(plan_path),
        "planned_images": len(rows), "accepted_images": len(accepted), "rejected_images": len(rejected),
        "accepted_by_split": dict(Counter(r["split"] for r in accepted)),
        "accepted_by_profile": dict(Counter(r["profile_id"] for r in accepted)),
        "rejection_reasons": dict(Counter(r["reason"] for r in rejected)),
        "molecule_overlap": len({r["molecule_id"] for r in accepted if r["split"] == "train"} &
                                {r["molecule_id"] for r in accepted if r["split"] == "holdout"}),
        "next": "run target adapter and frozen initial identity-aligned replay; no training admission",
    }
    (OUTPUT / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in accepted))
    (OUTPUT / "rejected.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rejected))
    (OUTPUT / "sample-sha256.json").write_text(json.dumps(sha, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run()
