#!/usr/bin/env python3
"""Audit P0-G AtomPair identity and target coverage across fixed checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from chemvector.audit_g3_bond_slot_identity import EXPECTED_STEPS, _config_from_payload
from chemvector.g3_hierarchical_graph import (
    G3AtomPairBondIdentityPredictor,
    build_g3_atom_pair_targets,
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
from chemvector.train_g3_hierarchical_graph import (
    DEFAULT_VISION_CHECKPOINT_SHA256,
    _autocast_context,
    cache_frozen_vision_features,
    load_frozen_vision_checkpoint,
)


DEFAULT_ARTIFACT_DIR = Path(
    "outputs/smoke/chemvector-g3-p0g-gt-atom-pair-bond-capacity"
)


def validate_p0g_checkpoint(payload: dict, *, step: int) -> None:
    if payload.get("global_step") != step:
        raise ValueError("P0-G checkpoint filename and global step differ")
    scope = payload.get("structural_scope", {})
    if not scope.get("gt_atom_slot_oracle") or not scope.get(
        "direct_gt_atom_pair_bonds"
    ):
        raise ValueError("Audit accepts only P0-G GT AtomPair checkpoints")
    if any(
        scope.get(name)
        for name in (
            "fixed_learned_bond_slots",
            "set_learned_bond_slots",
            "gt_bond_slot_identity_oracle",
            "shared_confidence_aware_bond_assignment",
        )
    ):
        raise ValueError("P0-G checkpoint contains an incompatible Bond protocol")
    if payload.get("sample_manifest", {}).get("sha256") != OVERFIT32_MANIFEST_SHA256:
        raise ValueError("P0-G checkpoint is not bound to the committed overfit32")
    if payload.get("source_vision_checkpoint_sha256") != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-G checkpoint uses an unexpected vision source")


def atom_pair_identity_sha256(
    sample_keys: list[str],
    pair_indices: torch.Tensor,
    sample_indices: torch.Tensor,
) -> str:
    rows = [
        [sample_keys[int(sample)], int(pair[0]), int(pair[1])]
        for pair, sample in zip(pair_indices.tolist(), sample_indices.tolist())
    ]
    encoded = json.dumps(rows, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@torch.no_grad()
def audit_checkpoint(
    model: G3AtomPairBondIdentityPredictor,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    *,
    sample_keys: list[str],
    batch_size: int,
    precision: str,
) -> dict:
    model.eval()
    all_pairs = []
    all_samples = []
    target_bonds = covered_bonds = universe_pairs = 0
    for start in range(0, len(features), batch_size):
        current_features = features[start : start + batch_size]
        current_targets = {
            name: value[start : start + batch_size] for name, value in targets.items()
        }
        with _autocast_context(features.device, precision):
            output = model(
                current_features,
                atom_counts=current_targets["atom_mask"].sum(dim=1).long(),
                oracle_atom_anchors=current_targets["atom_anchors"],
                oracle_atom_mask=current_targets["atom_mask"],
            )
        pair_targets = build_g3_atom_pair_targets(output, current_targets)
        pairs = output["flat_atom_pair_indices"].detach().cpu()
        samples = output["flat_atom_pair_sample_indices"].detach().cpu() + start
        all_pairs.append(pairs)
        all_samples.append(samples)
        universe_pairs += len(pairs)
        target_bonds += int(pair_targets["target_bonds"])
        covered_bonds += int(pair_targets["covered_bonds"])
    pairs = torch.cat(all_pairs)
    samples = torch.cat(all_samples)
    return {
        "pair_universe_pairs": universe_pairs,
        "target_bonds": target_bonds,
        "covered_bonds": covered_bonds,
        "target_coverage": covered_bonds / max(target_bonds, 1),
        "pair_identity_sha256": atom_pair_identity_sha256(
            sample_keys, pairs, samples
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vision-batch-size", type=int, default=4)
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.batch_size, args.vision_batch_size) < 1:
        raise ValueError("P0-G audit batch sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P0-G audit requested CUDA but CUDA is unavailable")
    if args.device == "cpu" and args.precision != "fp32":
        raise ValueError("P0-G CPU audit requires fp32")
    artifact_dir = Path(args.artifact_dir)
    checkpoint_paths = [
        artifact_dir / f"step-{step:06d}.pt" for step in EXPECTED_STEPS
    ]
    if any(not path.is_file() for path in checkpoint_paths):
        raise FileNotFoundError("P0-G audit requires all six fixed checkpoints")

    device = torch.device(args.device)
    manifest, manifest_audit = load_g3_overfit32_manifest(OVERFIT32_MANIFEST)
    records = load_g3_manifest_samples(manifest)
    first_payload = torch.load(
        checkpoint_paths[0], map_location="cpu", weights_only=False
    )
    validate_p0g_checkpoint(first_payload, step=EXPECTED_STEPS[0])
    vision, _, _, vision_sha256 = load_frozen_vision_checkpoint(
        first_payload["source_vision_checkpoint"], device
    )
    if vision_sha256 != DEFAULT_VISION_CHECKPOINT_SHA256:
        raise ValueError("P0-G audit vision bytes differ from the frozen protocol")
    config = _config_from_payload(first_payload)
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

    reports = []
    identity_hashes = []
    for step, path in zip(EXPECTED_STEPS, checkpoint_paths):
        payload = (
            first_payload
            if step == EXPECTED_STEPS[0]
            else torch.load(path, map_location="cpu", weights_only=False)
        )
        validate_p0g_checkpoint(payload, step=step)
        if _config_from_payload(payload) != config:
            raise ValueError("P0-G checkpoint model configs differ")
        model = G3AtomPairBondIdentityPredictor(config).to(device)
        model.load_state_dict(payload["model"], strict=True)
        current = audit_checkpoint(
            model,
            features,
            targets,
            sample_keys=sample_keys,
            batch_size=args.batch_size,
            precision=args.precision,
        )
        current.update(
            {
                "step": step,
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": file_sha256(path),
                "stored_gate_status": payload["train_only_metrics"]["p0_gate"][
                    "status"
                ],
            }
        )
        reports.append(current)
        identity_hashes.append(current["pair_identity_sha256"])
        del model, payload

    report = {
        "schema_version": "g3-p0g-atom-pair-identity-audit-0.1",
        "scope": "train-only-overfit32-no-grad-replay",
        "manifest": manifest_audit,
        "manifest_sha256": hashlib.sha256(
            Path(OVERFIT32_MANIFEST).read_bytes()
        ).hexdigest(),
        "expected_steps": list(EXPECTED_STEPS),
        "checkpoints": reports,
        "pair_identity_unique_hashes": len(set(identity_hashes)),
        "pair_identity_checkpoint_churn": 0.0
        if len(set(identity_hashes)) == 1
        else 1.0,
        "all_target_coverage_exact": all(
            item["target_bonds"] == item["covered_bonds"] == 1508
            for item in reports
        ),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
