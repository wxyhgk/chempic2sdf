"""Read-only CPU audit for the identity-aligned visible-text render.

This audit validates row accounting, identity/split ownership, loader
round-trips, O/N visibility and the geometric separation claim.  It does not
load a checkpoint, run inference, read sealed data or authorize training.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from chemvector.training.curriculum.data import load_sample

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r4"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-visible-text-20261017-r4-cpu-audit/report.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run() -> dict:
    if not DATA.is_dir() or not (DATA / "summary.json").is_file():
        raise FileNotFoundError(DATA)
    plan = json.loads((DATA / "plan.json").read_text(encoding="utf-8"))
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rejected = [json.loads(line) for line in (DATA / "rejected.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    planned = plan["rows"]
    if len(planned) != 1200 or len(rows) + len(rejected) != len(planned):
        raise ValueError("incomplete 1,200-row accounting")
    planned_keys = {r["sample_key"] for r in planned}
    observed_keys = {r["sample_key"] for r in rows + rejected}
    if planned_keys != observed_keys or len(observed_keys) != 1200:
        raise ValueError("plan/render sample-key mismatch")
    profile_ids = [p["profile_id"] for p in plan["selection"]["profiles"]]
    profile_rows = Counter(r["profile_id"] for r in rows + rejected)
    if profile_rows != Counter({p: 200 for p in profile_ids}):
        raise ValueError("profile denominator mismatch")
    train_ids = {r["molecule_id"] for r in planned if r["split"] == "train"}
    holdout_ids = {r["molecule_id"] for r in planned if r["split"] == "holdout"}
    scaffold_train = {r["scaffold"] for r in planned if r["split"] == "train"}
    scaffold_holdout = {r["scaffold"] for r in planned if r["split"] == "holdout"}
    if train_ids & holdout_ids or scaffold_train & scaffold_holdout:
        raise ValueError("identity/scaffold split overlap")
    loaded = 0
    visible = Counter()
    min_separation = float("inf")
    for row in rows:
        sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
        loaded += 1
        atoms = sample["layers"]["atoms"]
        for index, atom in enumerate(atoms):
            if atom.get("element") in {"O", "N"} and atom.get("visible_text"):
                visible[(row["split"], row["profile_id"], atom["element"])] += 1
            point = atom["center"]
            for other_index, other in enumerate(atoms):
                if index != other_index:
                    dx = point[0] - other["center"][0]
                    dy = point[1] - other["center"][1]
                    min_separation = min(min_separation, (dx * dx + dy * dy) ** 0.5)
    result = {
        "schema_version": "chemvector-identity-aligned-visible-text-render-cpu-audit-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": False, "model_forwards": 0, "optimizer_steps": 0,
        "sealed_test_read": False, "training_authorized": False,
        "dataset": str(DATA), "plan_sha256": digest(DATA / "plan.json"),
        "summary_sha256": digest(DATA / "summary.json"),
        "planned_images": len(planned), "accepted_images": len(rows),
        "rejected_images": len(rejected), "loaded_accepted_images": loaded,
        "accepted_by_split": dict(Counter(r["split"] for r in rows)),
        "accepted_by_profile": dict(Counter(r["profile_id"] for r in rows)),
        "rejection_reasons": dict(Counter(r["reason"] for r in rejected)),
        "visible_on_by_stratum": {"|".join(k): v for k, v in sorted(visible.items())},
        "identity": {"train_molecules": len(train_ids), "holdout_molecules": len(holdout_ids),
                     "molecule_overlap": len(train_ids & holdout_ids),
                     "scaffold_overlap": len(scaffold_train & scaffold_holdout)},
        "geometry": {"declared_min_anchor_separation_px": plan["identity_contract"]["target_anchor_separation_px"],
                     "observed_min_anchor_separation_px": min_separation},
        "gate": {"row_accounting": True, "profile_denominator": True,
                 "loader_roundtrip": loaded == len(rows), "identity_disjoint": not (train_ids & holdout_ids),
                 "scaffold_disjoint": not (scaffold_train & scaffold_holdout),
                 "declared_anchor_separation_met": min_separation >= plan["identity_contract"]["target_anchor_separation_px"],
                 "gpu_training_admissible": False},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
