#!/usr/bin/env python3
"""Audit P1-G dynamic assignment turnover across all five checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from chemvector.components.atoms.g3_equivalent_candidate_matching import (
    build_score_matched_equivalent_targets,
)
from chemvector.components.atoms.g3_top128_set_ranker import extract_frozen_top128
from chemvector.evaluation.audits.atoms.audit_g3_p1c_final import (
    _ranker_from_payload,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.atoms import train_g3_p1g_equivalent_set_matching as p1g
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _autocast_context,
    load_frozen_vision_checkpoint,
)


DEFAULT_OUTPUT = Path(
    "outputs/audits/chemvector-g3-p1h-dynamic-assignment-stability-train5k"
)
EXPECTED_SHA256 = {
    1_250: "d548e6bf7d370174889c438968539d411df77b57a04c8f27b9fde78f9d8a2544",
    2_500: "a227eba3cf5c06041a91718842a61bb4d13c414484f1bc8a23604ba981c20e19",
    3_750: "576f6d92c34822131e667256ca4105a44352473a2b27b65ad27eb450b414805b",
    5_000: "d58a6c7300b471c94789efe7fb43eaa31734bf8252bc51c99f6ba56b7cdb8143",
    6_250: "1416b06d1cac1cddf6248d5d99d2b8883f26a96f8f20362c7619b98807a9f9a1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assignment(dynamic, candidate_ids, index) -> dict[int, int]:
    mask = dynamic["positive_mask"][index]
    targets = dynamic["matched_target_indices"][index, mask].cpu().tolist()
    candidates = candidate_ids[index, mask].cpu().tolist()
    result = dict(zip(targets, candidates))
    if len(result) != len(targets):
        raise RuntimeError("P1-H dynamic assignment is not one-to-one")
    return result


def _empty_geometry():
    return p1g._empty_geometry_totals()


def _finalize_turnover(value: dict[str, int]) -> dict:
    return {
        **value,
        "atom_turnover_rate": value["changed_atoms"] / value["atoms"],
        "image_turnover_rate": value["changed_images"] / value["images"],
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P1-H requires CUDA BF16 replay")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("P1-H output directory is not empty")
    device = torch.device("cuda")
    rankers = {}
    actual_sha = {}
    for step, expected_sha in EXPECTED_SHA256.items():
        path = p1g.DEFAULT_OUTPUT / f"step-{step:06d}.pt"
        actual_sha[step] = _sha256(path)
        if actual_sha[step] != expected_sha:
            raise RuntimeError(f"P1-H checkpoint SHA differs at step {step}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["global_step"] != step or payload["model_type"] != p1g.MODEL_TYPE:
            raise RuntimeError(f"P1-H checkpoint identity differs at step {step}")
        _, ranker = _ranker_from_payload(payload, device)
        rankers[step] = ranker.eval()

    atom_head, _, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    manifest, manifest_audit = p1a.load_train_manifest()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    steps = list(EXPECTED_SHA256)
    geometry = {step: _empty_geometry() for step in steps}
    assignment_digests = {step: hashlib.sha256() for step in steps}
    transitions = {
        f"{left}->{right}": {
            "images": 0,
            "atoms": 0,
            "changed_images": 0,
            "changed_atoms": 0,
        }
        for left, right in zip(steps, steps[1:])
    }
    stable_atoms = stable_images = images = atoms = 0
    sample_digest = hashlib.sha256()
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
        for key in batch["keys"]:
            sample_digest.update(f"{key}\n".encode())
        pixels = batch["pixel_values"].to(device)
        targets = {
            "atom_anchors": batch["atom_anchors"].to(device),
            "atom_mask": batch["atom_mask"].to(device),
        }
        with _autocast_context(device, "bf16"):
            features = vision.forward_stroke_resolution(pixels)
            dense = atom_head(features.detach())
            candidates = extract_frozen_top128(dense)
            ranked = {step: rankers[step](candidates) for step in steps}
        fixed = p1d.build_hybrid_top128_targets(candidates, targets)
        dynamic = {
            step: build_score_matched_equivalent_targets(
                ranked[step], fixed, targets["atom_anchors"]
            )
            for step in steps
        }
        for index, key in enumerate(batch["keys"]):
            assignments = {}
            for step in steps:
                current = p1g._geometry_sample(
                    ranked[step], fixed, targets["atom_anchors"], index
                )
                for name, value in current.items():
                    geometry[step][name] += value
                assignments[step] = _assignment(
                    dynamic[step], candidates["candidate_ids"], index
                )
                for target, candidate in sorted(assignments[step].items()):
                    assignment_digests[step].update(
                        f"{key}\t{target}\t{candidate}\n".encode()
                    )
            target_keys = set(assignments[steps[0]])
            if any(set(assignments[step]) != target_keys for step in steps[1:]):
                raise RuntimeError("P1-H representable target set changed by checkpoint")
            images += 1
            atoms += len(target_keys)
            stable_count = sum(
                len({assignments[step][target] for step in steps}) == 1
                for target in target_keys
            )
            stable_atoms += stable_count
            stable_images += int(stable_count == len(target_keys))
            for left, right in zip(steps, steps[1:]):
                value = transitions[f"{left}->{right}"]
                changed = sum(
                    assignments[left][target] != assignments[right][target]
                    for target in target_keys
                )
                value["images"] += 1
                value["atoms"] += len(target_keys)
                value["changed_images"] += int(changed > 0)
                value["changed_atoms"] += changed

    transition_report = {
        key: _finalize_turnover(value) for key, value in transitions.items()
    }
    last_rate = transition_report["5000->6250"]["atom_turnover_rate"]
    if last_rate >= 0.10:
        decision = "late_assignment_instability_major"
    elif last_rate >= 0.03:
        decision = "late_assignment_instability_secondary"
    else:
        decision = "late_assignment_instability_not_primary"
    report = {
        "schema_version": "g3-p1h-assignment-stability-audit-0.1",
        "scope": "fixed-train5k-five-p1g-checkpoints",
        "manifest_sha256": manifest_audit["sha256"],
        "sample_keys_sha256": sample_digest.hexdigest(),
        "vision_checkpoint_sha256": vision_sha,
        "checkpoint_sha256": {str(key): value for key, value in actual_sha.items()},
        "assignment_sha256": {
            str(step): assignment_digests[step].hexdigest() for step in steps
        },
        "geometry": {
            str(step): p1g._finalize_geometry(geometry[step]) for step in steps
        },
        "transitions": transition_report,
        "all_five_stability": {
            "images": images,
            "atoms": atoms,
            "stable_images": stable_images,
            "stable_atoms": stable_atoms,
            "stable_image_rate": stable_images / images,
            "stable_atom_rate": stable_atoms / atoms,
        },
        "updates": 0,
        "forbidden_data_read": False,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"decision": decision, "last_turnover": last_rate}))


if __name__ == "__main__":
    main()
