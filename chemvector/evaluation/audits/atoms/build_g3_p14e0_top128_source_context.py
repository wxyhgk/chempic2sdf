#!/usr/bin/env python3
"""Enrich the P11-E0 real-source pilot with its complete P4-E3 top128 set."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time

import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    decode_atom_center_field,
)
from chemvector.evaluation.audits.atoms import audit_g3_p1n_regional_maxima as p1n
from chemvector.evaluation.audits.atoms import audit_g3_p4e6_missing_peak_sources as p4e6
from chemvector.evaluation.audits.atoms import build_g3_p11e0_predicted_source_pilot as p11e0
from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1t_continuous_stride2_center_field as p1t
from chemvector.training.graph.scenes.g3_structural_data import load_g3_sample_manifest
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _autocast_context
from chemvector.training.atoms.train_g3_p7_pairwise_counterfactual import (
    _atomic_json,
    _atomic_torch_save,
)


AUDIT_NAME = "chemvector-g3-p14e0-top128-source-context-pilot"
OUTPUT = Path("outputs/audits") / AUDIT_NAME
CACHE = OUTPUT / "context.pt"
SOURCE_CACHE_SHA256 = (
    "b84e200c81c02e5f93b4a8c0e51f9d1eae5decae9b71c70742338a4437ea38ee"
)
TOP_K = 128
BATCH_SIZE = 8
EXPECTED_ROWS = 2_152
EXPECTED_IMAGES = 236
MAXIMUM_ALIGNMENT_ERROR_PIXELS = 1e-3


def source_context(
    anchors: torch.Tensor,
    scores: torch.Tensor,
    source_ids: torch.Tensor,
    base_source_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if anchors.shape != (TOP_K, 2) or scores.shape != (TOP_K,):
        raise ValueError("P14-E0 requires a complete top128 candidate set")
    if source_ids.shape != (TOP_K,) or source_ids.dtype != torch.long:
        raise ValueError("P14-E0 top128 source IDs differ")
    positions = torch.where(source_ids == int(base_source_id))[0]
    if len(positions) != 1:
        raise ValueError("P14-E0 base source is not unique in top128")
    source_index = int(positions[0])
    source = anchors[source_index]
    mask = torch.zeros(TOP_K, dtype=torch.bool, device=anchors.device)
    mask[source_index] = True
    return anchors - source, scores, mask, source_index


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P14-E0 output already exists: {OUTPUT}")
    if file_sha256(p11e0.CACHE) != SOURCE_CACHE_SHA256:
        raise ValueError("P14-E0 P11-E0 source cache SHA256 differs")
    if file_sha256(p11e0.PILOT_MANIFEST) != p11e0.PILOT_MANIFEST_SHA256:
        raise ValueError("P14-E0 pilot manifest SHA256 differs")
    if file_sha256(p4e6.P4E3_CHECKPOINT) != p4e6.P4E3_CHECKPOINT_SHA256:
        raise ValueError("P14-E0 P4-E3 checkpoint SHA256 differs")
    if not torch.cuda.is_available():
        raise RuntimeError("P14-E0 requires CUDA")

    source_cache = torch.load(p11e0.CACHE, map_location="cpu", weights_only=False)
    if len(source_cache["sample_key"]) != EXPECTED_ROWS:
        raise RuntimeError("P14-E0 source row count differs")
    rows_by_key: dict[str, list[int]] = defaultdict(list)
    for row, key in enumerate(source_cache["sample_key"]):
        rows_by_key[str(key)].append(row)
    if len(rows_by_key) != EXPECTED_IMAGES:
        raise RuntimeError("P14-E0 source image count differs")

    full_manifest, manifest_audit = load_g3_sample_manifest(
        p11e0.PILOT_MANIFEST,
        expected_sha256=p11e0.PILOT_MANIFEST_SHA256,
        expected_samples=20_000,
        expected_molecules=5_000,
        expected_variants=frozenset({"coordgen", "original", "rdkit", "stress"}),
    )
    manifest = [row for row in full_manifest if row.sample_key in rows_by_key]
    if len(manifest) != EXPECTED_IMAGES:
        raise RuntimeError("P14-E0 source manifest is incomplete")

    device = torch.device("cuda")
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p11e0.p6e8.ResidualCycleCollator(maximum_atoms=128, image_size=768)
    offsets = torch.empty(EXPECTED_ROWS, TOP_K, 2, dtype=torch.float16)
    scores = torch.empty(EXPECTED_ROWS, TOP_K, dtype=torch.float16)
    source_mask = torch.zeros(EXPECTED_ROWS, TOP_K, dtype=torch.bool)
    source_rank = torch.empty(EXPECTED_ROWS, dtype=torch.long)
    seen = torch.zeros(EXPECTED_ROWS, dtype=torch.bool)
    maximum_error = 0.0
    started = time.perf_counter()

    with torch.no_grad():
        for batch in p1a.iter_manifest_batches(
            manifest,
            collator,
            batch_size=BATCH_SIZE,
            accepted_split="train",
        ):
            objects, stroke = p1t.model_inputs(
                atom_head,
                vision,
                frozen_block,
                batch["pixel_values"],
                device=device,
                training=False,
            )
            with _autocast_context(device, "bf16"):
                field_output = field(objects, stroke)
            peaks = p1n.extract_regional_maximum_objects(
                decode_atom_center_field(field_output), maximum_objects=TOP_K
            )
            for batch_index, key_value in enumerate(batch["keys"]):
                key = str(key_value)
                peak = peaks[batch_index]
                anchors = peak["anchors"].float()
                peak_scores = peak["scores"].float()
                source_ids = peak["source_ids"].long()
                expected = batch["atom_anchors"][batch_index].float().cpu()
                for row in rows_by_key[key]:
                    current_offsets, current_scores, current_mask, rank = source_context(
                        anchors,
                        peak_scores,
                        source_ids,
                        int(source_cache["base_source_id"][row]),
                    )
                    target_index = int(source_cache["target_index"][row])
                    inferred_source = (
                        expected[target_index]
                        - source_cache["offset_targets"][row].float()
                    )
                    error = float(
                        torch.linalg.vector_norm(
                            (anchors[rank] - inferred_source) * 768.0
                        )
                    )
                    maximum_error = max(maximum_error, error)
                    if error > MAXIMUM_ALIGNMENT_ERROR_PIXELS:
                        raise RuntimeError(
                            f"P14-E0 source alignment differs at row {row}: {error} px"
                        )
                    offsets[row] = current_offsets.half()
                    scores[row] = current_scores.half()
                    source_mask[row] = current_mask
                    source_rank[row] = rank
                    seen[row] = True

    checks = {
        "rows_2152": int(seen.sum()) == EXPECTED_ROWS,
        "images_236": len(rows_by_key) == EXPECTED_IMAGES,
        "all_rows_seen_once": bool(seen.all()),
        "one_source_token_per_row": bool((source_mask.sum(dim=1) == 1).all()),
        "source_rank_matches_mask": bool(
            source_mask[torch.arange(EXPECTED_ROWS), source_rank].all()
        ),
        "alignment_error_max_le1e3px": maximum_error
        <= MAXIMUM_ALIGNMENT_ERROR_PIXELS,
    }
    decision = "top128_source_context_pilot_go" if all(checks.values()) else "top128_source_context_pilot_no_go"
    payload = {
        "schema_version": "g3-p14e0-top128-source-context-pilot-0.1",
        "source_cache_sha256": SOURCE_CACHE_SHA256,
        "p4e3_checkpoint_sha256": p4e6.P4E3_CHECKPOINT_SHA256,
        "vision_checkpoint_sha256": vision_sha,
        "sample_key": list(source_cache["sample_key"]),
        "base_source_id": list(source_cache["base_source_id"]),
        "set_offsets": offsets,
        "set_scores": scores,
        "set_source_mask": source_mask,
        "set_source_rank": source_rank,
    }
    OUTPUT.mkdir(parents=True)
    _atomic_torch_save(CACHE, payload)
    audit = {
        "schema_version": "g3-p14e0-top128-source-context-pilot-audit-0.1",
        "decision": decision,
        "checks": checks,
        "cache": str(CACHE.resolve()),
        "cache_sha256": file_sha256(CACHE),
        "rows": EXPECTED_ROWS,
        "images": EXPECTED_IMAGES,
        "maximum_alignment_error_pixels": maximum_error,
        "elapsed_seconds": time.perf_counter() - started,
        "manifest": manifest_audit,
        "holdout_read": False,
        "confirm_external_sealed_read": False,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
