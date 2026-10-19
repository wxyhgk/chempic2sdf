"""Read-only deterministic retry audit for the 88 missing oa13 rows.

This does not write dataset samples, does not touch the sealed/evaluation
identities, and never changes the source quarantine batch.  It establishes
whether the missing rows can actually be repaired with the exact original
oa13 renderer tuple.
"""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from Data.sdf.structure_expansion_v1.batch import load_fixed_candidates
from Data.sdf.case_diversity_v1.build import molecule_job
from Data.sdf.case_diversity_v1.profiles import profiles


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
OUTPUT = ROOT / "outputs/audits/chemvector-oa13-missing-repair-20261017/report.json"


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    samples = json.loads((SOURCE / "samples.json").read_bytes())
    rejected = json.loads((SOURCE / "rejections.json").read_bytes())
    missing = [row for row in rejected if row["profile_id"] == "oa13"]
    if len(missing) != 88 or len({row["molecule_id"] for row in missing}) != 88:
        raise ValueError("source no longer contains exactly 88 missing oa13 rows")
    present = {(row["molecule_id"], row["profile_id"]) for row in samples}
    if any((row["molecule_id"], "oa13") in present for row in missing):
        raise ValueError("a supposedly missing oa13 row is already present")

    entries, _ = load_fixed_candidates()
    by_id = {entry["molecule_id"]: entry for entry in entries}
    oa13 = next(profile for profile in profiles() if profile.profile_id == "oa13")
    attempts = []
    for row in sorted(missing, key=lambda value: value["molecule_id"]):
        entry = by_id[row["molecule_id"]]
        result = molecule_job((entry, [oa13]))[0]
        if result["failure"] is not None:
            attempts.append({**row, "retry_status": "quality_rejected",
                             "retry_reason": result["failure"]["reason"]})
        else:
            # A successful retry is deliberately not persisted by this audit.
            meta = result["meta"]
            attempts.append({**row, "retry_status": "accepted",
                             "retry_reason": None, "retry_sample_key": meta["sample_key"]})

    counts = Counter(item["retry_status"] for item in attempts)
    same_reason = sum(item["reason"] == item["retry_reason"]
                      for item in attempts if item["retry_status"] == "quality_rejected")
    report = {
        "schema_version": "chemvector-oa13-missing-repair-0.1",
        "complete": True,
        "read_only": True,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_summary_sha256": digest(SOURCE / "summary.json"),
        "source_samples_sha256": digest(SOURCE / "samples.json"),
        "source_rejections_sha256": digest(SOURCE / "rejections.json"),
        "profile_id": "oa13",
        "profile_tuple": {
            "font_size": oa13.font_size,
            "bond_length_cap": oa13.bond_length_cap,
            "bond_line_width": oa13.bond_line_width,
            "rotation_degrees": oa13.rotation_degrees,
            "font_family": oa13.font_family,
        },
        "planned_retries": len(attempts),
        "retry_counts": dict(sorted(counts.items())),
        "same_quality_reason_as_original": same_reason,
        "attempts": attempts,
        "repair_gate": {
            "all_88_repaired": counts.get("accepted", 0) == 88,
            "profile_repair_status": "GO" if counts.get("accepted", 0) == 88 else "NO-GO",
            "next": ("persist repaired rows, then audit the bounded supplement"
                     if counts.get("accepted", 0) == 88 else
                     "stop; do not render extra oa13/oa33 identities or train"),
        },
        "training_authorized": False,
        "gpu_used": False,
        "model_forwards": 0,
        "sealed_test_read": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("planned_retries", "retry_counts", "same_quality_reason_as_original", "repair_gate")}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
