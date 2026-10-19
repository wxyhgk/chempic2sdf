#!/usr/bin/env python3
"""Read-only audit of r5 residual-ownership coverage.

The reviewed vertex artifact has excellent all-candidate atom coverage, but
production integration protects atoms already owned by the frozen baseline.
This panel therefore separates all candidates from candidates that remain
after the fixed duplicate/ownership exclusion.  It reports an upper bound for
residual-only bond completion and does not train, sweep a threshold, or write
any model state.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
import tools.preflight_endpoint_pair_compatibility_r5 as pair

INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-r5-residual-ownership-coverage/report.json"
IMAGE = 768.0
TARGET_RADIUS = 6.0
DUPLICATE_RADIUS = 6.0
HOLDOUT_LIMIT = 80


def _finish(c: Counter) -> dict:
    out = dict(c)
    atoms = max(c["target_atoms"], 1)
    bonds = max(c["target_bonds"], 1)
    for key in ("all_atom_coverage", "baseline_atom_coverage", "residual_atom_coverage",
                "residual_bond_upper_bound", "one_residual_bond_upper_bound",
                "baseline_residual_bond_upper_bound", "all_bond_upper_bound"):
        denom = atoms if "atom_coverage" in key else bonds
        out[key] = c[key] / denom
    return out


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint digest changed")
    rows = [json.loads(line) for line in (pair.DATA / "samples.jsonl").read_text().splitlines() if line]
    rows = sorted([row for row in rows if row.get("render_status") == "accepted" and row["split"] == "holdout"], key=lambda x: x["sample_key"])
    rows = rows[:HOLDOUT_LIMIT]
    if len(rows) != HOLDOUT_LIMIT:
        raise RuntimeError(f"expected {HOLDOUT_LIMIT} fixed holdout rows, got {len(rows)}")
    candidates = pair._candidates()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
    total = Counter()
    by_profile: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        sample = load_sample(pair.DATA, row["sample_key"], purpose="diagnostic")
        with torch.inference_mode():
            out = structure_forward(model, image_tensor(sample["image"])[None].to(device), config, payload["provenance"]["atom_threshold"])
        selected = {key: value.detach().cpu() for key, value in out[6].items()}
        values = candidates[row["sample_key"]]
        candidate_points = torch.tensor([[value["x"], value["y"]] for value in values], dtype=torch.float32)
        baseline_points = selected["atom_geometry"][0, selected["atom_mask"][0].bool(), :2]
        if len(baseline_points):
            candidate_to_baseline = torch.cdist(candidate_points * IMAGE, baseline_points * IMAGE).amin(dim=1)
            residual_mask = candidate_to_baseline > DUPLICATE_RADIUS
        else:
            residual_mask = torch.ones(len(candidate_points), dtype=torch.bool)
        atoms = sorted(sample["graph"]["atoms"], key=lambda value: value["atom_id"])
        atom_index = {value["atom_id"]: index for index, value in enumerate(atoms)}
        layer_atoms = {value["atom_id"]: value for value in sample["layers"]["atoms"]}
        target_points = torch.tensor([layer_atoms[value["atom_id"]]["center"] for value in atoms], dtype=torch.float32)
        target_bonds = [(atom_index[int(value["begin_atom"])], atom_index[int(value["end_atom"])]) for value in sample["graph"]["bonds"]]
        all_dist = torch.cdist(candidate_points * IMAGE, target_points)
        residual_points = candidate_points[residual_mask]
        residual_dist = torch.cdist(residual_points * IMAGE, target_points) if len(residual_points) else torch.full((0, len(target_points)), float("inf"))
        baseline_dist = torch.cdist(baseline_points * IMAGE, target_points) if len(baseline_points) else torch.full((0, len(target_points)), float("inf"))
        all_owned = all_dist.min(dim=0).values <= TARGET_RADIUS
        residual_owned = residual_dist.min(dim=0).values <= TARGET_RADIUS
        baseline_owned = baseline_dist.min(dim=0).values <= TARGET_RADIUS
        current = Counter({"images": 1, "target_atoms": len(target_points), "target_bonds": len(target_bonds), "candidate_points": len(candidate_points), "residual_candidate_points": int(residual_mask.sum()), "all_atom_coverage": int(all_owned.sum()), "baseline_atom_coverage": int(baseline_owned.sum()), "residual_atom_coverage": int(residual_owned.sum()), "residual_bond_upper_bound": sum(bool(residual_owned[a] and residual_owned[b]) for a, b in target_bonds), "one_residual_bond_upper_bound": sum(bool(residual_owned[a] or residual_owned[b]) for a, b in target_bonds), "baseline_residual_bond_upper_bound": sum(bool((baseline_owned[a] and residual_owned[b]) or (baseline_owned[b] and residual_owned[a])) for a, b in target_bonds), "all_bond_upper_bound": sum(bool(all_owned[a] and all_owned[b]) for a, b in target_bonds), "both_missing_after_split": sum(bool((not baseline_owned[a]) and (not residual_owned[a]) and (not baseline_owned[b]) and (not residual_owned[b])) for a, b in target_bonds)})
        for key, value in current.items():
            total[key] += value
        for key, value in current.items():
            by_profile[row["profile_id"]][key] += value
    report = {
        "schema_version": "chemvector-identity-aligned-r5-residual-ownership-coverage-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "production_checkpoint_written": False,
        "initial_sha256": INITIAL_SHA,
        "candidate_artifact_sha256": pair.digest(pair.CANDIDATES),
        "fixed_slice": {"split": "holdout", "images": len(rows), "target_radius_px": TARGET_RADIUS, "duplicate_radius_px": DUPLICATE_RADIUS},
        "all": _finish(total),
        "by_profile": {key: _finish(value) for key, value in sorted(by_profile.items())},
        "interpretation": {
            "all_candidate_atom_coverage": "upper bound if baseline ownership protection were disabled",
            "residual_atom_coverage": "coverage available to a residual-only sidecar after baseline duplicate exclusion",
            "residual_bond_upper_bound": "upper bound for residual-residual edges; no scorer can exceed it",
            "one_residual_bond_upper_bound": "upper bound when one endpoint is baseline-owned and the other is residual",
        },
        "gate": {
            "checkpoint_unchanged": file_digest(INITIAL) == INITIAL_SHA,
            "training_admissible": False,
            "production_go": False,
            "next": "train a residual-only vertex proposal head with baseline-owned negatives and hard collision/edge supervision; do not repeat scalar pair scoring",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "all": report["all"], "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
