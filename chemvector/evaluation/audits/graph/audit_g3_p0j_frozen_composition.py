#!/usr/bin/env python3
"""Audit frozen P0-I Atom and P0-G relation composition on overfit32."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.audit_g3_p0i_canonical_cell_slot import (
    config_from_payload,
    model_from_checkpoint as p0i_model_from_checkpoint,
)
from chemvector.g3_hierarchical_graph import (
    G3AtomPairBondIdentityPredictor,
    build_g3_bond_targets,
)
from chemvector.g3_structural_data import (
    G3StructuralCollator,
    OVERFIT32_MANIFEST,
    OVERFIT32_MANIFEST_SHA256,
    file_sha256,
    g3_targets_to_device,
    load_g3_manifest_samples,
    load_g3_overfit32_manifest,
)
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.train_g3_hierarchical_graph import (
    DEFAULT_VISION_CHECKPOINT_SHA256,
    _autocast_context,
    cache_frozen_vision_features,
    evaluate_g3_count_modes,
    load_frozen_vision_checkpoint,
)


P0I_CHECKPOINT = Path(
    "outputs/smoke/chemvector-g3-p0i-canonical-cell-slot-overfit32/last.pt"
)
P0I_CHECKPOINT_SHA256 = (
    "452a95c84cbb5bfb4a0e185f9955a963e5c126f77663fc8888052fdae619fe60"
)
P0G_CHECKPOINT = Path(
    "outputs/smoke/chemvector-g3-p0g-gt-atom-pair-bond-capacity/last.pt"
)
P0G_CHECKPOINT_SHA256 = (
    "f4f6f6030e1dc0802a143b960a83ed3aff3163b2853e648c9a058b9150b08ee0"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/smoke/chemvector-g3-p0j-frozen-relation-composition"
)
EXPECTED_IMAGES = 32
EXPECTED_ATOMS = 1424
EXPECTED_BONDS = 1508
EXPECTED_PAIRS = 37380
IMAGE_SIZE = 768.0
MATCH_RADIUS_PIXELS = 6.0


def load_frozen_p0_models(
    p0i_path: Path,
    p0g_path: Path,
    device: torch.device,
):
    """Strictly restore and freeze the two committed capacity winners."""

    if file_sha256(p0i_path) != P0I_CHECKPOINT_SHA256:
        raise ValueError("P0-J P0-I checkpoint SHA differs from the protocol")
    if file_sha256(p0g_path) != P0G_CHECKPOINT_SHA256:
        raise ValueError("P0-J P0-G checkpoint SHA differs from the protocol")
    p0i_payload = torch.load(p0i_path, map_location="cpu", weights_only=False)
    p0g_payload = torch.load(p0g_path, map_location="cpu", weights_only=False)
    p0i_config = config_from_payload(p0i_payload)
    p0g_config = config_from_payload(p0g_payload)
    if p0i_config != p0g_config:
        raise ValueError("P0-J P0-I and P0-G configs differ")
    p0i = p0i_model_from_checkpoint(p0i_payload, p0i_config, device)
    if p0g_payload.get("model_type") != G3AtomPairBondIdentityPredictor.model_type:
        raise ValueError("P0-J P0-G checkpoint has an unexpected model type")
    p0g = G3AtomPairBondIdentityPredictor(p0g_config).to(device)
    p0g.load_state_dict(p0g_payload["model"], strict=True)
    for model in (p0i, p0g):
        model.requires_grad_(False)
        model.eval()
    return p0i, p0g, p0i_payload, p0g_payload, p0i_config


def maximum_radius_atom_mapping(
    predicted_anchors: torch.Tensor,
    predicted_indices: torch.Tensor,
    expected_anchors: torch.Tensor,
    expected_indices: torch.Tensor,
    *,
    radius_pixels: float = MATCH_RADIUS_PIXELS,
    image_size: float = IMAGE_SIZE,
) -> dict[int, int]:
    """Maximize one-to-one in-radius Atom matches, then minimize distance."""

    predicted = predicted_anchors[predicted_indices].float()
    expected = expected_anchors[expected_indices].float()
    if not len(predicted) or not len(expected):
        return {}
    distances = torch.cdist(expected, predicted) * image_size
    eligible_columns = (distances <= radius_pixels).any(dim=0)
    eligible = distances[:, eligible_columns]
    eligible_predicted = predicted_indices[eligible_columns]
    unmatched_cost = float(len(expected) + 1)
    real_cost = torch.where(
        eligible <= radius_pixels,
        eligible / radius_pixels,
        torch.full_like(eligible, 2.0 * unmatched_cost),
    )
    dummy = torch.full(
        (len(expected), len(expected)),
        unmatched_cost,
        device=distances.device,
        dtype=distances.dtype,
    )
    rows, columns = linear_sum_assignment(
        torch.cat((real_cost, dummy), dim=1).detach().cpu().numpy()
    )
    mapping = {}
    for row, column in zip(rows, columns):
        if column >= eligible.size(1):
            continue
        if float(eligible[row, column]) > radius_pixels:
            continue
        mapping[int(eligible_predicted[column])] = int(expected_indices[row])
    return mapping


def gather_slot_features(
    dense: dict[str, torch.Tensor],
    slot_indices: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather P0-G features at P0-I predicted dense identities without GT."""

    source = dense["slot_features"]
    if slot_indices.shape != (*atom_mask.shape, 3):
        raise ValueError("P0-J Atom slot identities have an incompatible shape")
    result = source.new_zeros(atom_mask.size(0), atom_mask.size(1), source.size(2))
    for batch_index, query_index in atom_mask.nonzero(as_tuple=False).tolist():
        slot, y_index, x_index = (
            int(value) for value in slot_indices[batch_index, query_index].tolist()
        )
        if not (
            0 <= slot < source.size(1)
            and 0 <= y_index < source.size(-2)
            and 0 <= x_index < source.size(-1)
        ):
            raise ValueError("P0-J selected Atom has an invalid dense identity")
        result[batch_index, query_index] = source[
            batch_index, slot, :, y_index, x_index
        ]
    return result


def compose_pair_output(
    p0g: G3AtomPairBondIdentityPredictor,
    visual_features: torch.Tensor,
    atoms: dict[str, torch.Tensor],
    atom_features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run the frozen P0-G pair/count heads over a supplied predicted Atom set."""

    pair_indices, sample_indices = enumerate_unordered_atom_pairs(atoms["atom_mask"])
    flat = p0g.pair_bond(
        visual_features,
        atom_features,
        atoms["atom_anchors"],
        atoms["atom_scores"],
        pair_indices,
        sample_indices,
    )
    weights = atoms["atom_mask"][..., None].to(atom_features.dtype)
    pooled_atoms = (atom_features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    pooled_visual = visual_features.mean(dim=(-2, -1))
    count_logits = p0g.bond_count(torch.cat((pooled_visual, pooled_atoms), dim=-1))
    return {
        "pair_indices": pair_indices,
        "sample_indices": sample_indices,
        "presence_logits": flat["presence_logits"],
        "chemical_logits": flat["chemical_logits"],
        "bond_count_logits": count_logits,
    }


def empty_mode_totals() -> dict[str, float]:
    totals = {
        "images": 0.0,
        "atom_targets": 0.0,
        "atom_matched": 0.0,
        "atom_set_exact_images": 0.0,
        "target_bonds": 0.0,
        "representable_bonds": 0.0,
        "positive_type_correct": 0.0,
        "pair_universe": 0.0,
    }
    for mode in ("gt_bond_count", "deploy_bond_count"):
        totals.update(
            {
                f"{mode}_selected": 0.0,
                f"{mode}_true_positive": 0.0,
                f"{mode}_count_exact_images": 0.0,
                f"{mode}_count_absolute_error": 0.0,
                f"{mode}_skeleton_exact_images": 0.0,
                f"{mode}_typed_graph_exact_images": 0.0,
            }
        )
    return totals


def composition_batch_statistics(
    pair_output: dict[str, torch.Tensor],
    atoms: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    normalized_bonds: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Score composition after forward; GT mapping never changes pair predictions."""

    totals = empty_mode_totals()
    for batch_index in range(atoms["atom_mask"].size(0)):
        predicted_indices = torch.where(atoms["atom_mask"][batch_index])[0]
        expected_indices = torch.where(targets["atom_mask"][batch_index].bool())[0]
        mapping = maximum_radius_atom_mapping(
            atoms["atom_anchors"][batch_index],
            predicted_indices,
            targets["atom_anchors"][batch_index],
            expected_indices,
        )
        atom_exact = (
            len(predicted_indices) == len(expected_indices) == len(mapping)
        )
        target_count = int(normalized_bonds["mask"][batch_index].sum())
        expected_graph = {}
        for index in range(target_count):
            begin, end = normalized_bonds["atom_refs"][batch_index, index].tolist()
            expected_graph[tuple(sorted((int(begin), int(end))))] = int(
                normalized_bonds["chemical"][batch_index, index]
            )
        mapped_targets = set(mapping.values())
        representable = {
            pair: bond_type
            for pair, bond_type in expected_graph.items()
            if pair[0] in mapped_targets and pair[1] in mapped_targets
        }

        rows = torch.where(pair_output["sample_indices"] == batch_index)[0]
        logits = pair_output["presence_logits"][rows].float()
        pair_keys: list[tuple[int, int] | None] = []
        for row in rows.tolist():
            begin, end = pair_output["pair_indices"][row].tolist()
            if int(begin) not in mapping or int(end) not in mapping:
                pair_keys.append(None)
            else:
                pair_keys.append(tuple(sorted((mapping[int(begin)], mapping[int(end)]))))
        for local_index, key in enumerate(pair_keys):
            if key in expected_graph:
                predicted_type = int(
                    pair_output["chemical_logits"][rows[local_index]].argmax()
                )
                totals["positive_type_correct"] += float(
                    predicted_type == expected_graph[key]
                )

        deploy_count = min(
            int(pair_output["bond_count_logits"][batch_index].argmax()), len(rows)
        )
        for mode, selected_count in (
            ("gt_bond_count", target_count),
            ("deploy_bond_count", deploy_count),
        ):
            selected_local = logits.argsort(descending=True)[:selected_count].tolist()
            predicted_graph = {}
            valid_graph = atom_exact and len(representable) == len(expected_graph)
            true_positive = 0
            for local_index in selected_local:
                key = pair_keys[local_index]
                if key is None or key in predicted_graph:
                    valid_graph = False
                    continue
                row = rows[local_index]
                predicted_graph[key] = int(pair_output["chemical_logits"][row].argmax())
                true_positive += int(key in expected_graph)
            count_exact = selected_count == target_count
            skeleton_exact = (
                valid_graph
                and count_exact
                and set(predicted_graph) == set(expected_graph)
            )
            typed_exact = skeleton_exact and predicted_graph == expected_graph
            totals[f"{mode}_selected"] += selected_count
            totals[f"{mode}_true_positive"] += true_positive
            totals[f"{mode}_count_exact_images"] += int(count_exact)
            totals[f"{mode}_count_absolute_error"] += abs(selected_count - target_count)
            totals[f"{mode}_skeleton_exact_images"] += int(skeleton_exact)
            totals[f"{mode}_typed_graph_exact_images"] += int(typed_exact)

        totals["images"] += 1
        totals["atom_targets"] += len(expected_indices)
        totals["atom_matched"] += len(mapping)
        totals["atom_set_exact_images"] += int(atom_exact)
        totals["target_bonds"] += len(expected_graph)
        totals["representable_bonds"] += len(representable)
        totals["pair_universe"] += len(rows)
    return totals


def merge_totals(destination: dict[str, float], source: dict[str, float]) -> None:
    for name, value in source.items():
        destination[name] = destination.get(name, 0.0) + value


def finalize_mode_totals(totals: dict[str, float]) -> dict:
    images = max(totals["images"], 1.0)
    target_bonds = max(totals["target_bonds"], 1.0)
    result = {
        "images": int(totals["images"]),
        "atom_targets": int(totals["atom_targets"]),
        "atom_matched": int(totals["atom_matched"]),
        "atom_recall": totals["atom_matched"] / max(totals["atom_targets"], 1.0),
        "atom_set_exact_images": int(totals["atom_set_exact_images"]),
        "target_bonds": int(totals["target_bonds"]),
        "representable_bonds": int(totals["representable_bonds"]),
        "bond_representable_recall": totals["representable_bonds"] / target_bonds,
        "pair_universe": int(totals["pair_universe"]),
        "positive_type_accuracy": totals["positive_type_correct"] / target_bonds,
    }
    for mode in ("gt_bond_count", "deploy_bond_count"):
        selected = totals[f"{mode}_selected"]
        true_positive = totals[f"{mode}_true_positive"]
        precision = true_positive / max(selected, 1.0)
        recall = true_positive / target_bonds
        result[mode] = {
            "selected_pairs": int(selected),
            "true_positive_pairs": int(true_positive),
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
            "count_exact_images": int(totals[f"{mode}_count_exact_images"]),
            "count_mae": totals[f"{mode}_count_absolute_error"] / images,
            "skeleton_exact_images": int(
                totals[f"{mode}_skeleton_exact_images"]
            ),
            "typed_graph_exact_images": int(
                totals[f"{mode}_typed_graph_exact_images"]
            ),
        }
    return result


def pair_identity_sha256(sample_keys: list[str], atom_masks: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for sample_key, mask in zip(sample_keys, atom_masks):
        valid = torch.where(mask)[0]
        for pair in torch.combinations(valid, r=2).tolist():
            digest.update(f"{sample_key}\0{pair[0]}\0{pair[1]}\n".encode("utf-8"))
    return digest.hexdigest()


@torch.no_grad()
def evaluate_composition_mode(
    p0i,
    p0g,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    sample_keys: list[str],
    *,
    feature_mode: str,
    batch_size: int,
    precision: str,
) -> dict:
    if feature_mode not in {"p0i_native", "p0g_slot_replay"}:
        raise ValueError(f"Unknown P0-J feature mode: {feature_mode}")
    totals = empty_mode_totals()
    identity_masks = []
    for start in range(0, len(features), batch_size):
        current_features = features[start : start + batch_size]
        current_targets = {
            name: value[start : start + batch_size] for name, value in targets.items()
        }
        with _autocast_context(features.device, precision):
            atoms = p0i(current_features)
            atom_features = atoms["atom_features"]
            if feature_mode == "p0g_slot_replay":
                p0g_dense = p0g.atom_head(current_features)
                atom_features = gather_slot_features(
                    p0g_dense, atoms["atom_slot_indices"], atoms["atom_mask"]
                )
            pair_output = compose_pair_output(
                p0g, current_features, atoms, atom_features
            )
        current = composition_batch_statistics(
            pair_output,
            atoms,
            current_targets,
            build_g3_bond_targets(current_targets),
        )
        merge_totals(totals, current)
        identity_masks.extend(mask.detach().cpu() for mask in atoms["atom_mask"])
    result = finalize_mode_totals(totals)
    result["feature_mode"] = feature_mode
    result["pair_identity_sha256"] = pair_identity_sha256(
        sample_keys, identity_masks
    )
    return result


def p0j_gate(oracle: dict, native: dict, replay: dict) -> dict:
    checks = {
        "image_count": replay["images"] == EXPECTED_IMAGES,
        "atom_mapping": (
            replay["atom_matched"] == replay["atom_targets"] == EXPECTED_ATOMS
            and replay["atom_set_exact_images"] == EXPECTED_IMAGES
        ),
        "bond_representable": (
            replay["representable_bonds"]
            == replay["target_bonds"]
            == EXPECTED_BONDS
        ),
        "pair_universe": replay["pair_universe"] == EXPECTED_PAIRS,
        "feature_mode_pair_identity": (
            native["pair_identity_sha256"] == replay["pair_identity_sha256"]
        ),
        "p0g_oracle_gt_graph": (
            oracle["gt_count"]["structural_graph_exact_images"] == EXPECTED_IMAGES
        ),
        "p0g_oracle_deploy_graph": (
            oracle["deploy_count"]["structural_graph_exact_images"]
            == EXPECTED_IMAGES
        ),
        "replay_gt_skeleton": (
            replay["gt_bond_count"]["skeleton_exact_images"] >= 30
        ),
        "replay_gt_typed_graph": (
            replay["gt_bond_count"]["typed_graph_exact_images"] >= 30
        ),
        "replay_deploy_count": (
            replay["deploy_bond_count"]["count_exact_images"] >= 30
        ),
        "replay_deploy_skeleton": (
            replay["deploy_bond_count"]["skeleton_exact_images"] >= 30
        ),
        "replay_deploy_typed_graph": (
            replay["deploy_bond_count"]["typed_graph_exact_images"] >= 30
        ),
    }
    native_direct = (
        native["deploy_bond_count"]["typed_graph_exact_images"] >= 30
        and native["gt_bond_count"]["typed_graph_exact_images"] >= 30
    )
    replay_passed = all(checks.values())
    interpretation = (
        "direct_composition_go"
        if replay_passed and native_direct
        else (
            "atom_feature_interface_mismatch"
            if replay_passed
            else "predicted_atom_conditioned_relation_mismatch"
        )
    )
    return {
        "status": "passed" if replay_passed else "failed",
        "checks": checks,
        "native_feature_direct_composition": (
            "passed" if native_direct else "failed"
        ),
        "interpretation": interpretation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0i-checkpoint", default=str(P0I_CHECKPOINT))
    parser.add_argument("--p0g-checkpoint", default=str(P0G_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vision-batch-size", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.batch_size, args.vision_batch_size) < 1:
        raise ValueError("P0-J batch sizes must be positive")
    if args.device == "cpu" and args.precision != "fp32":
        raise ValueError("P0-J CPU audit requires fp32")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P0-J CUDA audit requested but CUDA is unavailable")
    output_dir = Path(args.output_dir)
    if output_dir.resolve() != DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError(f"P0-J requires fixed output directory: {DEFAULT_OUTPUT_DIR}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"P0-J output directory is not empty: {output_dir}")

    device = torch.device(args.device)
    p0i, p0g, p0i_payload, p0g_payload, config = load_frozen_p0_models(
        Path(args.p0i_checkpoint), Path(args.p0g_checkpoint), device
    )
    manifest, manifest_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    if manifest_audit["sha256"] != OVERFIT32_MANIFEST_SHA256:
        raise ValueError("P0-J overfit32 manifest SHA differs from the protocol")
    records = load_g3_manifest_samples(manifest)
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        p0i_payload["source_vision_checkpoint"], device
    )
    if vision_sha != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-J vision SHA differs from the protocol")
    collator = G3StructuralCollator(
        maximum_atoms=config.maximum_atoms,
        maximum_bonds=config.maximum_bonds,
        image_size=768,
    )
    batch = collator(records)
    features = cache_frozen_vision_features(
        vision,
        batch["pixel_values"],
        device=device,
        batch_size=args.vision_batch_size,
        precision=args.precision,
    )
    targets = g3_targets_to_device(batch, device)
    sample_keys = list(batch["keys"])
    del vision, batch

    oracle = evaluate_g3_count_modes(
        p0g,
        features,
        targets,
        batch_size=args.batch_size,
        precision=args.precision,
        gt_atom_slot_oracle=True,
        direct_gt_atom_pair_bonds=True,
    )
    native = evaluate_composition_mode(
        p0i,
        p0g,
        features,
        targets,
        sample_keys,
        feature_mode="p0i_native",
        batch_size=args.batch_size,
        precision=args.precision,
    )
    replay = evaluate_composition_mode(
        p0i,
        p0g,
        features,
        targets,
        sample_keys,
        feature_mode="p0g_slot_replay",
        batch_size=args.batch_size,
        precision=args.precision,
    )
    report = {
        "schema_version": "g3-p0j-frozen-relation-composition-0.1",
        "scope": "train-only-overfit32-no-grad-composition",
        "manifest": manifest_audit,
        "vision": {
            "path": str(Path(p0i_payload["source_vision_checkpoint"]).resolve()),
            "sha256": vision_sha,
        },
        "checkpoints": {
            "p0i": {
                "path": str(Path(args.p0i_checkpoint).resolve()),
                "sha256": file_sha256(args.p0i_checkpoint),
                "model_type": p0i_payload["model_type"],
                "global_step": p0i_payload["global_step"],
            },
            "p0g": {
                "path": str(Path(args.p0g_checkpoint).resolve()),
                "sha256": file_sha256(args.p0g_checkpoint),
                "model_type": p0g_payload["model_type"],
                "global_step": p0g_payload["global_step"],
            },
        },
        "implementation": {
            "audit_script": str(Path(__file__).resolve()),
            "audit_script_sha256": file_sha256(__file__),
        },
        "p0g_gt_atom_oracle_replay": oracle,
        "p0i_native_feature": native,
        "p0g_slot_replay_feature": replay,
        "p0j_gate": p0j_gate(oracle, native, replay),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "composition_audit.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
