#!/usr/bin/env python3
"""Frozen-bond endpoint reachability panel for the r5 holdout.

This is a forward-only diagnostic.  It does not train, change candidates, or
feed labels into inference.  Ground-truth geometry is used only after frozen
bond proposals have been produced, to distinguish endpoint geometry ceilings
from candidate/compatibility errors.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
DEFAULT_OUTPUT = ROOT / "outputs/audits/chemvector-identity-aligned-bond-endpoint-reachability-20261017-r5/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
RADII = (6.0, 12.0, 16.0, 20.0, 24.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target(sample: dict) -> tuple[torch.Tensor, list[dict]]:
    atoms = sorted(sample["graph"]["atoms"], key=lambda row: row["atom_id"])
    visible = {row["atom_id"]: row for row in sample["layers"]["atoms"]}
    points = torch.tensor([visible[row["atom_id"]]["center"] for row in atoms], dtype=torch.float32)
    return points, list(sample["graph"]["bonds"])


def _bond_match_counts(predicted: torch.Tensor, targets: torch.Tensor, radii: tuple[float, ...]) -> dict[str, int]:
    if not len(predicted) or not len(targets):
        return {f"r{int(radius)}": 0 for radius in radii}
    forward = torch.linalg.vector_norm((predicted[:, None] - targets[None]) * 1.0, dim=-1).amax(-1)
    reverse = torch.linalg.vector_norm((predicted[:, None] - targets[None].flip(2)) * 1.0, dim=-1).amax(-1)
    best = torch.minimum(forward, reverse)
    return {f"r{int(radius)}": int((best.min(dim=0).values <= radius).sum()) for radius in radii}


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("initial checkpoint digest changed")
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line]
    rows = [row for row in rows if row["split"] == "holdout"]
    if len(rows) != 474:
        raise RuntimeError(f"expected fixed r5 holdout of 474, got {len(rows)}")
    device = torch.device("cpu" if args.cpu else "cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --cpu for a slow diagnostic")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
    total = Counter(); profiles: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
        target_points_px, target_bonds = _target(sample)
        target_bond_points = target_points_px[torch.tensor([[b["begin_atom"], b["end_atom"]] for b in target_bonds], dtype=torch.long)]
        with torch.inference_mode():
            pixels = image_tensor(sample["image"])[None].to(device)
            _, _, _, _, bonds, _, _ = structure_forward(model, pixels, config, payload["provenance"]["atom_threshold"])
        endpoints = bonds["bond_endpoints"][0, bonds["bond_mask"][0]].detach().cpu() * 768.0
        target_points = target_points_px
        distances = torch.cdist(endpoints.reshape(-1, 2), target_points)
        nearest = distances.min(dim=1).values
        target_bond_points = target_bond_points.to(endpoints)
        bond_counts = _bond_match_counts(endpoints, target_bond_points, RADII)
        profile = profiles[row["profile_id"]]
        for counter in (total, profile):
            counter["images"] += 1
            counter["predicted_bonds"] += len(endpoints)
            counter["predicted_endpoints"] += endpoints.numel() // 2
            counter["target_bonds"] += len(target_bond_points)
            counter["target_endpoints"] += len(target_points) * 0 + len(target_bond_points) * 2
            counter["endpoint_distance_sum"] += float(nearest.sum())
            counter["endpoint_count"] += len(nearest)
            for radius in RADII:
                key = f"r{int(radius)}"
                counter[f"endpoint_within_{key}"] += int((nearest <= radius).sum())
                counter[f"target_bonds_reached_{key}"] += bond_counts[key]
        # Complete target-bond recall is reported separately from endpoint
        # contamination: the former asks whether each GT bond has any proposal.
    def finish(counter: Counter) -> dict:
        result = dict(counter)
        result["mean_endpoint_distance_px"] = counter["endpoint_distance_sum"] / max(counter["endpoint_count"], 1)
        for radius in RADII:
            key = f"r{int(radius)}"
            result[f"endpoint_reachability_{key}"] = counter[f"endpoint_within_{key}"] / max(counter["predicted_endpoints"], 1)
            result[f"target_bond_recall_{key}"] = counter[f"target_bonds_reached_{key}"] / max(counter["target_bonds"], 1)
        return result
    report = {
        "schema_version": "chemvector-identity-aligned-bond-endpoint-reachability-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "deployment_authorized": False, "sealed_test_read": False,
        "initial_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": digest(INITIAL),
        "split": "fixed_r5_holdout", "images": len(rows), "radii_px": list(RADII),
        "all": finish(total), "by_profile": {key: finish(value) for key, value in sorted(profiles.items())},
        "gate": {"production_unchanged": digest(INITIAL) == INITIAL_SHA,
                 "training_admissible": False,
                 "next": "use endpoint reachability to decide compatibility sidecar vs frozen bond-head repair"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "all": report["all"], "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())
