#!/usr/bin/env python3
"""Evaluate frozen P9 pairwise overrides on scaffold-disjoint whole images."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import torch

from chemvector.components.atoms.objects.g3_atom_center_field import decode_atom_center_field
from chemvector.components.recovery.pairwise_structural_counterfactual import (
    PairwiseStructuralCounterfactual,
    PairwiseStructuralCounterfactualConfig,
)
from chemvector.components.recovery.structural_context_peak import (
    StructuralContextPeakHead,
    anchor_centered_context_crops,
)
from chemvector.components.recovery.structural_peak_residual import normalized_peak_prior
from chemvector.evaluation.audits.atoms import audit_g3_p1n_regional_maxima as p1n
from chemvector.evaluation.audits.atoms import audit_g3_p4e6_missing_peak_sources as p4e6
from chemvector.evaluation.audits.atoms import build_g3_p3e21_shared_cache as p3e21
from chemvector.evaluation.audits.atoms import build_g3_p6e8_residual_cycle_data as p6e8
from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import g3_p4_local_recovery_data as recovery_data
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.atoms import train_g3_p6e3_structural_peak_selector as p6e3
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9
from chemvector.training.atoms import train_g3_p9_stratified_pairwise_anchor as p9
from chemvector.training.graph.scenes.g3_structural_data import load_g3_sample_manifest
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json, _autocast_context


AUDIT_NAME = "chemvector-g3-p10-pairwise-dev-shadow"
OUTPUT = Path("outputs/audits") / AUDIT_NAME
DEV_MANIFEST = p3e21.MANIFEST_ROOT / "dev.jsonl"
DEV_MANIFEST_SHA256 = p3e21.MANIFESTS["dev"][0]
P9_CHECKPOINT = p9.CHECKPOINT
P9_CHECKPOINT_SHA256 = "8bc96e42c2229c640967baa1cf4e12ed577230fae8a0465bb815e1c799423eae"
THRESHOLD = 1.6389398574829102
IMAGE_SIZE = 768.0
MATCH_RADIUS_PIXELS = 6.0
TOP_K = 128
BATCH_SIZE = 4


def choose_overrides(
    corrected_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if corrected_scores.shape != candidate_mask.shape or corrected_scores.ndim != 2:
        raise ValueError("P10 score and mask shapes differ")
    if candidate_mask.dtype != torch.bool:
        raise ValueError("P10 candidate mask must be boolean")
    scores = corrected_scores.masked_fill(~candidate_mask, -torch.inf)
    winners = scores.argmax(dim=1)
    values = scores.gather(1, winners[:, None])[:, 0]
    execute = values >= float(threshold)
    return winners, execute


def coverage_metrics(predicted: torch.Tensor, expected: torch.Tensor) -> tuple[int, bool]:
    assignments = p4e6.maximum_cardinality_assignments(
        predicted.cpu(), expected.cpu(), radius_pixels=MATCH_RADIUS_PIXELS
    )
    matched = sum(value is not None for value in assignments)
    return matched, matched == len(expected)


def summarize_records(records: list[dict]) -> dict:
    totals = Counter()
    variants: dict[str, Counter] = defaultdict(Counter)
    paired = Counter()
    for row in records:
        totals["images"] += 1
        totals["targets"] += int(row["targets"])
        totals["raw_matched"] += int(row["raw_matched"])
        totals["prior_matched"] += int(row["prior_matched"])
        totals["p9_matched"] += int(row["p9_matched"])
        totals["raw_exact"] += int(row["raw_exact"])
        totals["prior_exact"] += int(row["prior_exact"])
        totals["p9_exact"] += int(row["p9_exact"])
        totals["actions"] += int(row["actions"])
        variants[row["variant"]].update(
            {
                "images": 1,
                "targets": int(row["targets"]),
                "prior_matched": int(row["prior_matched"]),
                "p9_matched": int(row["p9_matched"]),
                "prior_exact": int(row["prior_exact"]),
                "p9_exact": int(row["p9_exact"]),
                "actions": int(row["actions"]),
            }
        )
        paired["exact_rescues"] += int(not row["prior_exact"] and row["p9_exact"])
        paired["exact_breaks"] += int(row["prior_exact"] and not row["p9_exact"])
        paired["matched_gain_images"] += int(row["p9_matched"] > row["prior_matched"])
        paired["matched_loss_images"] += int(row["p9_matched"] < row["prior_matched"])
    return {
        **dict(totals),
        "matched_net_gain": totals["p9_matched"] - totals["prior_matched"],
        "exact_net_gain": totals["p9_exact"] - totals["prior_exact"],
        "paired": dict(paired),
        "by_variant": {name: dict(value) for name, value in sorted(variants.items())},
    }


def _load_p9(device: torch.device) -> PairwiseStructuralCounterfactual:
    if file_sha256(P9_CHECKPOINT) != P9_CHECKPOINT_SHA256:
        raise ValueError("P10 P9 checkpoint SHA256 differs")
    payload = torch.load(P9_CHECKPOINT, map_location="cpu", weights_only=False)
    model = PairwiseStructuralCounterfactual(
        PairwiseStructuralCounterfactualConfig(arm="patch-only")
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval().requires_grad_(False)
    return model


def _refine_image(
    sources: torch.Tensor,
    darkness: torch.Tensor,
    center: torch.Tensor,
    peak_model: StructuralContextPeakHead,
    pair_model: PairwiseStructuralCounterfactual,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    image_crops = anchor_centered_context_crops(darkness[None], sources[None])[0]
    score_crops = anchor_centered_context_crops(center[None], sources[None])[0]
    zeros = torch.zeros(len(sources), device=sources.device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        heatmaps = peak_model(image_crops, score_crops, zeros, zeros)
    offsets, logits, mask = p6e3.candidate_tensors(heatmaps)
    prior_scores = normalized_peak_prior(logits.float(), mask)
    prior_indices = prior_scores.argmax(dim=1)
    rows = torch.arange(len(sources), device=sources.device)
    prior_offsets = offsets[rows, prior_indices]
    maximum = offsets.size(1)
    expanded_images = image_crops[:, None].expand(-1, maximum, -1, -1, -1).reshape(-1, 1, 96, 96)
    expanded_scores = score_crops[:, None].expand(-1, maximum, -1, -1, -1).reshape(-1, 1, 96, 96)
    expanded_prior = prior_offsets[:, None].expand(-1, maximum, -1).reshape(-1, 2)
    alternatives = offsets.reshape(-1, 2)
    flat_mask = mask.reshape(-1)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        delta = pair_model(
            expanded_images,
            expanded_scores,
            torch.zeros_like(expanded_prior),
            expanded_prior,
            alternatives,
            flat_mask,
        ).reshape(len(sources), maximum)
    base_margin = prior_scores - prior_scores.gather(1, prior_indices[:, None])
    corrected = base_margin + delta.float()
    corrected.scatter_(1, prior_indices[:, None], -torch.inf)
    alternative_mask = mask.clone()
    alternative_mask.scatter_(1, prior_indices[:, None], False)
    winners, execute = choose_overrides(corrected, alternative_mask, THRESHOLD)
    selected = torch.where(execute[:, None], offsets[rows, winners], prior_offsets)
    return (
        (sources + prior_offsets).clamp(0.0, 1.0),
        (sources + selected).clamp(0.0, 1.0),
        int(execute.sum()),
    )


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P10 output already exists: {OUTPUT}")
    if any((path.stat().st_mode & 0o777) != 0 for path in p6e9.HOLDOUT_FILES):
        raise RuntimeError("P10 historical holdout must remain sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("P10 requires CUDA")
    manifest, manifest_audit = load_g3_sample_manifest(
        DEV_MANIFEST,
        expected_sha256=DEV_MANIFEST_SHA256,
        expected_samples=640,
        expected_molecules=160,
        expected_variants=frozenset({"coordgen", "original", "rdkit", "stress"}),
    )
    seen_scaffolds = set()
    for path in (p6e9.TRAIN_HARD, p6e9.TRAIN_EASY, p6e9.CALIBRATION_HARD, p6e9.CALIBRATION_EASY):
        seen_scaffolds.update(torch.load(path, map_location="cpu", weights_only=False)["scaffold"])
    manifest_rows = [json.loads(line) for line in DEV_MANIFEST.read_text().splitlines() if line]
    if any(str(row["scaffold"]) in seen_scaffolds for row in manifest_rows):
        raise RuntimeError("P10 dev scaffold overlaps P6/P7")
    device = torch.device("cuda")
    pair_model = _load_p9(device)
    peak_payload = torch.load(p6e8.P6E1_CHECKPOINT, map_location="cpu", weights_only=False)
    peak_model = StructuralContextPeakHead().to(device)
    peak_model.load_state_dict(peak_payload["model"], strict=True)
    peak_model.eval().requires_grad_(False)
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    records = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=BATCH_SIZE, accepted_split="train"):
            objects, stroke = p1t.model_inputs(
                atom_head, vision, frozen_block, batch["pixel_values"], device=device, training=False
            )
            with _autocast_context(device, "bf16"):
                field_output = field(objects, stroke)
            peaks = p1n.extract_regional_maximum_objects(
                decode_atom_center_field(field_output), maximum_objects=TOP_K
            )
            darkness = recovery_data._grayscale_darkness(batch["pixel_values"].to(device))
            center = field_output["center_logits"].detach().float()
            for index, key in enumerate(batch["keys"]):
                raw = peaks[index]["anchors"].float().to(device)
                expected = batch["atom_anchors"][index][batch["atom_mask"][index]].float().to(device)
                prior, refined, actions = _refine_image(
                    raw, darkness[index], center[index], peak_model, pair_model
                )
                raw_matched, raw_exact = coverage_metrics(raw, expected)
                prior_matched, prior_exact = coverage_metrics(prior, expected)
                p9_matched, p9_exact = coverage_metrics(refined, expected)
                records.append(
                    {
                        "sample_key": str(key),
                        "variant": str(batch["variants"][index]),
                        "targets": len(expected),
                        "raw_matched": raw_matched,
                        "prior_matched": prior_matched,
                        "p9_matched": p9_matched,
                        "raw_exact": raw_exact,
                        "prior_exact": prior_exact,
                        "p9_exact": p9_exact,
                        "actions": actions,
                    }
                )
            if len(records) % 160 == 0:
                print(json.dumps({"images": len(records), "elapsed_seconds": time.perf_counter() - started}), flush=True)
    metrics = summarize_records(records)
    checks = {
        "dev_images_640": len(records) == 640,
        "dev_molecules_160": manifest_audit["molecule_count"] == 160,
        "four_variants_160_each": all(sum(row["variant"] == name for row in records) == 160 for name in ("coordgen", "original", "rdkit", "stress")),
        "p6_p7_scaffold_overlap_zero": True,
        "p9_actions_positive": metrics["actions"] > 0,
        "matched_net_gain_positive": metrics["matched_net_gain"] > 0,
        "exact_net_gain_positive": metrics["exact_net_gain"] > 0,
        "paired_exact_rescues_gt_breaks": metrics["paired"]["exact_rescues"] > metrics["paired"]["exact_breaks"],
        "historical_holdout_mode_000": all((path.stat().st_mode & 0o777) == 0 for path in p6e9.HOLDOUT_FILES),
    }
    audit = {
        "schema_version": "g3-p10-pairwise-dev-shadow-audit-0.1",
        "decision": "pairwise_dev_shadow_go" if all(checks.values()) else "pairwise_dev_shadow_no_go",
        "checks": checks,
        "metrics": metrics,
        "threshold": THRESHOLD,
        "lineage": {
            "p9_checkpoint_sha256": P9_CHECKPOINT_SHA256,
            "p6e1_checkpoint_sha256": p6e8.P6E1_CHECKPOINT_SHA256,
            "vision_checkpoint_sha256": vision_sha,
            "dev_manifest_sha256": DEV_MANIFEST_SHA256,
        },
        "protocol": {
            "baseline": "P6 first peak for every frozen P4-E3 top128 source",
            "treatment": "P9 alternative only above frozen full-calibration threshold",
            "metric": "maximum-cardinality 6px top128 coverage",
            "deployment": False,
            "confirm_external_sealed_read": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "historical_p7_holdout_read": False,
        "confirm_external_sealed_read": False,
    }
    OUTPUT.mkdir(parents=True)
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
