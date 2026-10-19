#!/usr/bin/env python3
"""Label-only residual proposal reachability gate for r5.

This audit answers whether a residual-only proposal head is worth training.
The frozen production baseline owns any atom within 6 px; only candidates
outside that protected field may claim residual atoms.  Ground-truth labels
are used after frozen inference and remain in memory.  No model is trained,
no threshold is swept, and no target identity is written to an artifact.
"""
from __future__ import annotations

from collections import Counter, defaultdict
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
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-r5-residual-only-reachability/report.json"
IMAGE = 768.0
TARGET_RADIUS = 6.0
DUPLICATE_RADIUS = 6.0
TRAIN_LIMIT = 160
HOLDOUT_LIMIT = 80


def _empty() -> Counter:
    return Counter()


def _finish(c: Counter) -> dict:
    out = dict(c)
    atoms = max(c["target_atoms"], 1)
    residual_targets = max(c["residual_targets"], 1)
    bonds = max(c["target_bonds"], 1)
    residual_candidates = max(c["residual_candidates"], 1)
    for key in ("baseline_atom_coverage", "all_atom_coverage", "residual_atom_reachability"):
        out[key] = c[key] / (atoms if key != "residual_atom_reachability" else residual_targets)
    for key in ("residual_residual_bond_coverage", "one_residual_bond_coverage", "baseline_residual_bond_coverage", "combined_bond_coverage"):
        out[key] = c[key] / bonds
    out["residual_candidate_hard_collision_rate"] = c["hard_collision_candidates"] / residual_candidates
    out["residual_candidate_target_hit_rate"] = c["target_hit_candidates"] / residual_candidates
    out["residual_candidate_count_per_image"] = c["residual_candidates"] / max(c["images"], 1)
    out["residual_target_count_per_image"] = c["residual_targets"] / max(c["images"], 1)
    return out


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint digest changed")
    rows = [json.loads(line) for line in (pair.DATA / "samples.jsonl").read_text().splitlines() if line]
    accepted = [row for row in rows if row.get("render_status") == "accepted"]
    candidates = pair._candidates()
    splits = {
        "train": sorted([row for row in accepted if row["split"] == "train"], key=lambda x: x["sample_key"])[:TRAIN_LIMIT],
        "holdout": sorted([row for row in accepted if row["split"] == "holdout"], key=lambda x: x["sample_key"])[:HOLDOUT_LIMIT],
    }
    if len(splits["train"]) != TRAIN_LIMIT or len(splits["holdout"]) != HOLDOUT_LIMIT:
        raise RuntimeError("fixed r5 train/holdout slice changed")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
    reports: dict[str, dict] = {}
    for split, chosen in splits.items():
        total = _empty()
        by_profile: dict[str, Counter] = defaultdict(_empty)
        by_element: dict[str, Counter] = defaultdict(_empty)
        for row in chosen:
            sample = load_sample(pair.DATA, row["sample_key"], purpose="diagnostic")
            with torch.inference_mode():
                out = structure_forward(model, image_tensor(sample["image"])[None].to(device), config, payload["provenance"]["atom_threshold"])
            selected = {key: value.detach().cpu() for key, value in out[6].items()}
            baseline_points = selected["atom_geometry"][0, selected["atom_mask"][0].bool(), :2]
            values = candidates[row["sample_key"]]
            candidate_points = torch.tensor([[value["x"], value["y"]] for value in values], dtype=torch.float32)
            if len(baseline_points):
                candidate_to_baseline = torch.cdist(candidate_points * IMAGE, baseline_points * IMAGE).amin(dim=1)
                residual_mask = candidate_to_baseline > DUPLICATE_RADIUS
            else:
                residual_mask = torch.ones(len(candidate_points), dtype=torch.bool)
            residual_points = candidate_points[residual_mask]
            atoms = sorted(sample["graph"]["atoms"], key=lambda value: value["atom_id"])
            atom_index = {value["atom_id"]: index for index, value in enumerate(atoms)}
            layer_atoms = {value["atom_id"]: value for value in sample["layers"]["atoms"]}
            target_points = torch.tensor([layer_atoms[value["atom_id"]]["center"] for value in atoms], dtype=torch.float32)
            target_bonds = [(atom_index[int(value["begin_atom"])], atom_index[int(value["end_atom"])]) for value in sample["graph"]["bonds"]]
            baseline_dist = torch.cdist(baseline_points * IMAGE, target_points) if len(baseline_points) else torch.full((0, len(target_points)), float("inf"))
            all_dist = torch.cdist(candidate_points * IMAGE, target_points) if len(candidate_points) else torch.full((0, len(target_points)), float("inf"))
            residual_dist = torch.cdist(residual_points * IMAGE, target_points) if len(residual_points) else torch.full((0, len(target_points)), float("inf"))
            baseline_owned = baseline_dist.min(dim=0).values <= TARGET_RADIUS
            all_owned = all_dist.min(dim=0).values <= TARGET_RADIUS
            residual_owned = residual_dist.min(dim=0).values <= TARGET_RADIUS
            residual_targets = ~baseline_owned
            # Candidate-level hard collisions are residual proposals that land
            # on an already baseline-owned target.  A candidate can be a
            # target hit only when it is uniquely within TARGET_RADIUS of one
            # target; ambiguity is counted as a non-hit.
            target_hit = torch.zeros(len(residual_points), dtype=torch.bool)
            hard_collision = torch.zeros(len(residual_points), dtype=torch.bool)
            if len(residual_points):
                for ci in range(len(residual_points)):
                    close = torch.where(residual_dist[ci] <= TARGET_RADIUS)[0]
                    if len(close) == 1:
                        target_hit[ci] = True
                        hard_collision[ci] = bool(baseline_owned[int(close[0])])
            current = Counter({
                "images": 1,
                "target_atoms": len(target_points),
                "residual_targets": int(residual_targets.sum()),
                "target_bonds": len(target_bonds),
                "candidate_points": len(candidate_points),
                "residual_candidates": len(residual_points),
                "baseline_owned_atoms": int(baseline_owned.sum()),
                "all_atom_coverage": int(all_owned.sum()),
                "baseline_atom_coverage": int(baseline_owned.sum()),
                "residual_atom_reachability": int((residual_owned & residual_targets).sum()),
                "target_hit_candidates": int(target_hit.sum()),
                "hard_collision_candidates": int(hard_collision.sum()),
                "residual_residual_bond_coverage": sum(bool(residual_owned[a] and residual_owned[b]) for a, b in target_bonds),
                "one_residual_bond_coverage": sum(bool(residual_owned[a] or residual_owned[b]) for a, b in target_bonds),
                "baseline_residual_bond_coverage": sum(bool((baseline_owned[a] and residual_owned[b]) or (baseline_owned[b] and residual_owned[a])) for a, b in target_bonds),
                "combined_bond_coverage": sum(bool((baseline_owned[a] or residual_owned[a]) and (baseline_owned[b] or residual_owned[b])) for a, b in target_bonds),
                "both_uncovered_bonds": sum(bool((not baseline_owned[a]) and (not residual_owned[a]) and (not baseline_owned[b]) and (not residual_owned[b])) for a, b in target_bonds),
            })
            for key, value in current.items():
                total[key] += value
            for key, value in current.items():
                by_profile[row["profile_id"]][key] += value
            for index, atom in enumerate(atoms):
                element = atom.get("element", "other")
                if element == "C" and not layer_atoms[atom["atom_id"]].get("visible_text", False):
                    element = "implicit_C"
                if element not in {"O", "N", "implicit_C"}:
                    element = "other"
                by_element[element]["target_atoms"] += 1
                by_element[element]["residual_targets"] += int(residual_targets[index])
                by_element[element]["residual_reached"] += int(residual_targets[index] and residual_owned[index])
        reports[split] = {
            "images": len(chosen),
            "all": _finish(total),
            "by_profile": {key: _finish(value) for key, value in sorted(by_profile.items())},
            "by_element": {
                key: {
                    **dict(value),
                    "residual_atom_reachability": value["residual_reached"] / max(value["residual_targets"], 1),
                    "residual_target_fraction": value["residual_targets"] / max(value["target_atoms"], 1),
                }
                for key, value in sorted(by_element.items())
            },
        }
    report = {
        "schema_version": "chemvector-identity-aligned-r5-residual-only-reachability-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "label_only": True,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "production_checkpoint_written": False,
        "initial_sha256": INITIAL_SHA,
        "candidate_artifact_sha256": pair.digest(pair.CANDIDATES),
        "fixed_slice": {"train_images": TRAIN_LIMIT, "holdout_images": HOLDOUT_LIMIT, "target_radius_px": TARGET_RADIUS, "duplicate_radius_px": DUPLICATE_RADIUS, "device": str(device)},
        "by_split": reports,
        "gate": {
            "checkpoint_unchanged": file_digest(INITIAL) == INITIAL_SHA,
            "candidate_artifact_valid": True,
            "training_admissible": False,
            "shadow_authorized": False,
            "go": False,
            "next": "review residual-only labels and hard-collision rate; only then preregister one <=800-step residual proposal shadow",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "train": reports["train"]["all"], "holdout": reports["holdout"]["all"], "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
