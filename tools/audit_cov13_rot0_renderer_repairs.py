"""Renderer-only repair probe for the ten cov13_rot0 quality rejections.

The probe never writes samples, loads a model, or changes a production profile.
Each variant changes exactly one renderer factor from cov13_rot0 and retries
the same fixed ten identities.
"""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from Data.sdf.case_diversity_v1.rendering import render
from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/datasets/chemvector-oa13-oa33-coverage-panel-20261017-r2"
OUTPUT = ROOT / "outputs/audits/chemvector-cov13-rot0-renderer-repair-20261017/report.json"


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = [row for row in json.loads((SOURCE / "rejections.json").read_bytes())
            if row["profile_id"] == "cov13_rot0"]
    if len(rows) != 10:
        raise ValueError("expected exactly ten cov13_rot0 failures")
    base = dict(profile_id="cov13_rot0", width=768, height=768, font_size=32,
                bond_length_cap=28.0, bond_line_width=3.2, rotation_degrees=0.0,
                font_family="DejaVu Serif")
    variants = {
        "base_control": {},
        "bond_cap_40": {"bond_length_cap": 40.0},
        "bond_cap_52": {"bond_length_cap": 52.0},
        "bond_cap_64": {"bond_length_cap": 64.0},
        "line_width_1_6": {"bond_line_width": 1.6},
        "line_width_2_4": {"bond_line_width": 2.4},
    }
    results = {}
    for name, changes in variants.items():
        profile = Profile(**{**base, **changes})
        accepted, reasons = 0, Counter()
        for row in rows:
            entry = {"molecule_id": row["molecule_id"],
                     "canonical_smiles": row["canonical_smiles"],
                     "split": "quarantine", "training_authorized": False,
                     "external_overlap_status": "unchecked"}
            try:
                render(entry, profile)
            except ValueError as error:
                reasons[str(error)] += 1
            else:
                accepted += 1
        results[name] = {
            "changed_factor": "none" if not changes else next(iter(changes)),
            "profile_tuple": {**base, **changes},
            "attempts": len(rows), "accepted": accepted,
            "rejected": len(rows) - accepted,
            "rejection_reasons": dict(sorted(reasons.items())),
            "complete_denominator": accepted == len(rows),
            "decision": "PASS" if accepted == len(rows) else "NO-GO",
        }
    report = {
        "schema_version": "chemvector-cov13-rot0-renderer-repair-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": False, "model_forwards": 0, "sealed_test_read": False,
        "source_dataset": str(SOURCE.relative_to(ROOT)),
        "source_summary_sha256": digest(SOURCE / "summary.json"),
        "source_rejections_sha256": digest(SOURCE / "rejections.json"),
        "fixed_failure_count": len(rows), "variants": results,
        "recommendation": {
            "pass_options": ["bond_cap_40", "bond_cap_52", "bond_cap_64"],
            "no_go_options": ["line_width_1_6", "line_width_2_4"],
            "production_changed": False,
            "training_authorized": False,
            "next": "If a repair is needed, prepare a separately named profile and rerender the full panel; do not relabel this control as cov13_rot0.",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
