#!/usr/bin/env python3
"""Audit fixed candidate identity against 6px-equivalent Atom sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from chemvector.components.atoms.g3_strong_top128_set_ranker import (
    align_candidate_patches,
    cap_zero_candidate_patches,
    extract_predicted_anchor_stroke_patches,
    sham_candidate_patches,
)
from chemvector.components.atoms.g3_top128_set_ranker import (
    build_hybrid_top128_targets,
    extract_frozen_top128,
    select_ranked_top128,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1c_final import (
    _ranker_from_payload,
)
from chemvector.evaluation.audits.atoms.audit_g3_p1e_final import _load_arm
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.atoms import (
    train_g3_p1e_aligned_stroke_feature_sufficiency as p1e,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _autocast_context,
    load_frozen_vision_checkpoint,
)


DEFAULT_OUTPUT = Path(
    "outputs/audits/chemvector-g3-p1f-equivalent-candidate-supervision-train5k"
)
P1D_CHECKPOINT = p1d.DEFAULT_OUTPUT / "step-006250.pt"
P1E_CHECKPOINTS = {
    arm: p1e.DEFAULT_OUTPUT / arm / "step-006250.pt" for arm in p1e.ARMS
}
EXPECTED_CHECKPOINT_SHA256 = {
    "p1d": "a8fd94c36434ecb977c406b574e3d2c43d48f98d512aae491bd85a6eb3afd97c",
    "cap": "5c31296978672f35af4bd2a60a6d219f1ff5b1c1be230c73097c761c046dc393",
    "sham": "573d45254f80eff116fce2ceeea8d003b519ecae85d65753e5e0a54d2cf75039",
    "align": "445d75917ee44ebceeef752084d66c5b6a5234d3c48d965e877d05b403429aa4",
}
SHAM_SEED = 31_033
RADIUS = 6.0 / 768.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _empty_model_totals() -> dict[str, int]:
    return {
        "images": 0,
        "representable_atoms": 0,
        "selected_atoms": 0,
        "fixed_identity_matched_atoms": 0,
        "geometry_matched_atoms": 0,
        "fixed_identity_exact_images": 0,
        "geometry_equivalent_exact_images": 0,
        "identity_artifact_rescue_images": 0,
        "fixed_negative_selected_atoms": 0,
        "fixed_negative_selected_near_gt_atoms": 0,
    }


def sample_equivalence_statistics(
    ranked: dict,
    hybrid: dict,
    target_anchors: torch.Tensor,
    index: int,
) -> dict[str, int]:
    positive = hybrid["positive_mask"][index].bool()
    count = int(positive.sum())
    selected = select_ranked_top128(
        {key: value[index : index + 1] if isinstance(value, torch.Tensor) and value.ndim else value
         for key, value in ranked.items()},
        torch.tensor([count], device=positive.device),
    )
    selected_mask = selected["atom_mask"][0]
    selected_ids = selected["atom_flat_ids"][0, selected_mask]
    selected_anchors = selected["atom_anchors"][0, selected_mask].float()
    positive_ids = ranked["candidate_ids"][index, positive]
    target_indices = hybrid["matched_target_indices"][index, positive]
    if count != len(target_indices.unique()) or bool((target_indices < 0).any()):
        raise RuntimeError("P1-F hybrid positives do not map one-to-one to targets")
    representable_targets = target_anchors[target_indices].float()
    distances = torch.cdist(selected_anchors, representable_targets)
    compatible = distances <= RADIUS
    geometry_matched = p1a.maximum_cardinality_radius_match_count(
        selected_anchors.cpu(), representable_targets.cpu(), radius_pixels=6, image_size=768
    )
    selected_set = set(int(value) for value in selected_ids.cpu().tolist())
    positive_set = set(int(value) for value in positive_ids.cpu().tolist())
    fixed_matched = len(selected_set & positive_set)
    fixed_exact = selected_set == positive_set
    geometry_exact = geometry_matched == count
    fixed_negative = torch.tensor(
        [value not in positive_set for value in selected_ids.cpu().tolist()],
        device=compatible.device,
        dtype=torch.bool,
    )
    near = compatible.any(dim=1) if count else torch.zeros_like(fixed_negative)
    return {
        "images": 1,
        "representable_atoms": count,
        "selected_atoms": len(selected_ids),
        "fixed_identity_matched_atoms": fixed_matched,
        "geometry_matched_atoms": geometry_matched,
        "fixed_identity_exact_images": int(fixed_exact),
        "geometry_equivalent_exact_images": int(geometry_exact),
        "identity_artifact_rescue_images": int(geometry_exact and not fixed_exact),
        "fixed_negative_selected_atoms": int(fixed_negative.sum()),
        "fixed_negative_selected_near_gt_atoms": int((fixed_negative & near).sum()),
    }


def _finalize_model(totals: dict[str, int]) -> dict:
    images = totals["images"]
    atoms = totals["representable_atoms"]
    return {
        **totals,
        "fixed_identity_atom_recall": totals["fixed_identity_matched_atoms"] / atoms,
        "geometry_equivalent_atom_recall": totals["geometry_matched_atoms"] / atoms,
        "fixed_identity_exact_rate": totals["fixed_identity_exact_images"] / images,
        "geometry_equivalent_exact_rate": totals[
            "geometry_equivalent_exact_images"
        ]
        / images,
        "identity_artifact_rescue_rate": totals["identity_artifact_rescue_images"]
        / images,
        "geometry_minus_fixed_exact_pp": 100.0
        * (
            totals["geometry_equivalent_exact_images"]
            - totals["fixed_identity_exact_images"]
        )
        / images,
    }


@torch.no_grad()
def evaluate_manifest(atom_head, rankers, vision, manifest, collator, *, device):
    totals = {name: _empty_model_totals() for name in rankers}
    multiplicities = []
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
            stroke = vision.forward_stroke_resolution(pixels)
            dense = atom_head(stroke.detach())
            candidates = extract_frozen_top128(dense)
            aligned = extract_predicted_anchor_stroke_patches(stroke, candidates)
            sham = sham_candidate_patches(aligned, batch["keys"], seed=SHAM_SEED)[
                "candidate_patches"
            ]
            ranked = {
                "p1d": rankers["p1d"](candidates),
                "cap": rankers["cap"](candidates, cap_zero_candidate_patches(aligned)),
                "sham": rankers["sham"](candidates, sham),
                "align": rankers["align"](
                    candidates, align_candidate_patches(aligned)
                ),
            }
        hybrid = build_hybrid_top128_targets(candidates, targets)
        for index in range(len(batch["keys"])):
            positive = hybrid["positive_mask"][index].bool()
            target_indices = hybrid["matched_target_indices"][index, positive]
            representable = targets["atom_anchors"][index, target_indices].float()
            if len(representable):
                counts = (
                    torch.cdist(
                        candidates["candidate_anchors"][index].float(), representable
                    )
                    <= RADIUS
                ).sum(dim=0)
                multiplicities.extend(int(value) for value in counts.cpu().tolist())
            for name in rankers:
                current = sample_equivalence_statistics(
                    ranked[name], hybrid, targets["atom_anchors"][index], index
                )
                for key, value in current.items():
                    totals[name][key] += value
    ordered = sorted(multiplicities)
    count = len(ordered)
    buckets = {
        "1": sum(value == 1 for value in ordered),
        "2": sum(value == 2 for value in ordered),
        "3": sum(value == 3 for value in ordered),
        "4": sum(value == 4 for value in ordered),
        "5+": sum(value >= 5 for value in ordered),
    }
    return {
        "images": len(manifest),
        "models": {name: _finalize_model(value) for name, value in totals.items()},
        "candidate_multiplicity": {
            "representable_atoms": count,
            "buckets": buckets,
            "mean": sum(ordered) / count,
            "p50": ordered[int(0.50 * (count - 1))],
            "p90": ordered[int(0.90 * (count - 1))],
            "max": ordered[-1],
            "multiple_candidate_atoms": sum(value >= 2 for value in ordered),
            "multiple_candidate_rate": sum(value >= 2 for value in ordered) / count,
        },
        "sample_keys_sha256": sample_digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P1-F requires CUDA BF16 fixed-checkpoint replay")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("P1-F output directory is not empty")
    actual = {"p1d": _sha256(P1D_CHECKPOINT)}
    actual.update({arm: _sha256(path) for arm, path in P1E_CHECKPOINTS.items()})
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("P1-F checkpoint SHA differs")
    device = torch.device("cuda")
    p1d_payload = torch.load(P1D_CHECKPOINT, map_location="cpu", weights_only=False)
    _, p1d_ranker = _ranker_from_payload(p1d_payload, device)
    p1e_payloads = {
        arm: torch.load(path, map_location="cpu", weights_only=False)
        for arm, path in P1E_CHECKPOINTS.items()
    }
    rankers = {"p1d": p1d_ranker.eval()}
    rankers.update({arm: _load_arm(payload, device) for arm, payload in p1e_payloads.items()})
    atom_head, p1a_payload, _ = p1e.load_p1a_frozen_atom_head(device)
    vision, _, _, _ = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    train, _ = p1a.load_train_manifest()
    route, _ = p1a.load_route_manifest()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    report = {
        "schema_version": "g3-p1f-equivalent-candidate-supervision-audit-0.1",
        "scope": "fixed-identity-versus-6px-equivalent-set",
        "checkpoint_sha256": actual,
        "sham_seed": SHAM_SEED,
        "train5k": evaluate_manifest(
            atom_head, rankers, vision, train, collator, device=device
        ),
        "route100": evaluate_manifest(
            atom_head, rankers, vision, route, collator, device=device
        ),
        "updates": 0,
        "forbidden_data_read": False,
    }
    deltas = [
        value["geometry_minus_fixed_exact_pp"]
        for value in report["train5k"]["models"].values()
    ]
    multiple = report["train5k"]["candidate_multiplicity"][
        "multiple_candidate_rate"
    ]
    maximum_delta = max(deltas)
    if maximum_delta >= 10.0 and multiple >= 0.25:
        decision = "fixed_identity_major_bottleneck"
    elif maximum_delta >= 3.0:
        decision = "fixed_identity_secondary_bottleneck"
    else:
        decision = "fixed_identity_not_primary"
    report["decision"] = decision
    args.output_dir.mkdir(parents=True, exist_ok=True)
    p1e._atomic_json(args.output_dir / "audit.json", report)
    print(json.dumps({"output": str(args.output_dir / 'audit.json'), "decision": decision}))


if __name__ == "__main__":
    main()
