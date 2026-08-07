#!/usr/bin/env python3
"""Compare P4-E3 offset anchors with their exact stride-1 cell centers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    decode_atom_center_field,
)
from chemvector.evaluation import g3_p3e21_cycle as p3_cycle
from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
    extract_regional_maximum_objects,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p4e6_missing_peak_sources as p4e6,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p3e21_shared_cache as p3e21,
)
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.atoms import (
    train_g3_p1t_continuous_stride2_center_field as p1t,
)
from chemvector.training.graph.scenes.g3_structural_data import (
    load_g3_sample_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p4e7-offset-radius-ab"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
P4E6_AUDIT = Path(
    "outputs/audits/chemvector-g3-p4e6-missing-peak-sources/audit.json"
)
P4E6_AUDIT_SHA256 = (
    "d689bc0980439a916c3cba02ca4f9c7d57b54694e1f408c0ba6e0cf93fb07db3"
)
BUDGETS = (128, 1024)
RADII = (6.0, 8.0, 10.0, 12.0)
ANCHOR_MODES = ("predicted_offset", "cell_center")
BATCH_SIZE = 24
IMAGE_SIZE = 768
SEED = 20260847


def cell_center_anchors(source_ids: torch.Tensor) -> torch.Tensor:
    if source_ids.ndim != 1 or source_ids.dtype != torch.long:
        raise ValueError("P4-E7 source IDs must be a rank-one long tensor")
    if bool(((source_ids < 0) | (source_ids >= IMAGE_SIZE * IMAGE_SIZE)).any()):
        raise ValueError("P4-E7 source IDs are outside the stride-1 field")
    x = source_ids.remainder(IMAGE_SIZE).float()
    y = torch.div(source_ids, IMAGE_SIZE, rounding_mode="floor").float()
    return torch.stack(
        ((x + 0.5) / IMAGE_SIZE, (y + 0.5) / IMAGE_SIZE), dim=-1
    )


def choose_decision(metrics: dict, paired: dict) -> tuple[str, dict[str, bool]]:
    baseline = metrics["predicted_offset"]["1024"]["6"]
    center = metrics["cell_center"]["1024"]["6"]
    checks = {
        "baseline_replays_p4e5": (
            baseline["matched"] == 32_342 and baseline["exact"] == 597
        ),
        "cell_center_matched_gain_min15": center["matched"] >= baseline["matched"] + 15,
        "cell_center_exact_no_regression": center["exact"] >= baseline["exact"],
        "cell_center_no_paired_exact_break": paired["cell_center_vs_offset_radius6"][
            "exact_break"
        ]
        == 0,
    }
    return (
        "offset_head_harmful_go"
        if all(checks.values())
        else "offset_head_not_primary",
        checks,
    )


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P4-E7 output is not empty: {OUTPUT}")
    for path, expected in (
        (P4E6_AUDIT, P4E6_AUDIT_SHA256),
        (p4e6.P4E3_CHECKPOINT, p4e6.P4E3_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P4-E7 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P4-E7 requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    dev, manifest_audit = load_g3_sample_manifest(
        p3e21.MANIFEST_ROOT / "dev.jsonl",
        expected_sha256=p3e21.MANIFESTS["dev"][0],
        expected_samples=p3e21.MANIFESTS["dev"][1],
        expected_molecules=p3e21.MANIFESTS["dev"][2],
        expected_variants=p3_cycle.VARIANTS,
    )
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=IMAGE_SIZE)
    totals = {
        mode: {
            str(budget): {
                f"{radius:g}": {"matched": 0, "targets": 0, "exact": 0}
                for radius in RADII
            }
            for budget in BUDGETS
        }
        for mode in ANCHOR_MODES
    }
    per_image = []
    started = time.perf_counter()

    with torch.no_grad():
        for batch in p1a.iter_manifest_batches(dev, collator, batch_size=BATCH_SIZE):
            objects, stroke = p1t.model_inputs(
                atom_head,
                vision,
                frozen_block,
                batch["pixel_values"],
                device=device,
                training=False,
            )
            with _autocast_context(device, "bf16"):
                output = field(objects, stroke)
            peaks = extract_regional_maximum_objects(
                decode_atom_center_field(output), maximum_objects=max(BUDGETS)
            )
            for index, sample_key_value in enumerate(batch["keys"]):
                expected = batch["atom_anchors"][index][batch["atom_mask"][index]]
                anchors = {
                    "predicted_offset": peaks[index]["anchors"],
                    "cell_center": cell_center_anchors(peaks[index]["source_ids"]),
                }
                image_metrics = {mode: {} for mode in ANCHOR_MODES}
                for mode in ANCHOR_MODES:
                    for budget in BUDGETS:
                        image_metrics[mode][str(budget)] = {}
                        for radius in RADII:
                            matched = sum(
                                value is not None
                                for value in p4e6.maximum_cardinality_assignments(
                                    anchors[mode][:budget],
                                    expected,
                                    radius_pixels=radius,
                                    image_size=IMAGE_SIZE,
                                )
                            )
                            exact = int(matched == len(expected))
                            current = totals[mode][str(budget)][f"{radius:g}"]
                            current["matched"] += matched
                            current["targets"] += len(expected)
                            current["exact"] += exact
                            image_metrics[mode][str(budget)][f"{radius:g}"] = {
                                "matched": matched,
                                "exact": exact,
                            }
                per_image.append(
                    {"sample_key": str(sample_key_value), "metrics": image_metrics}
                )

    metrics = {
        mode: {
            budget: {
                radius: {
                    **value,
                    "recall": value["matched"] / value["targets"],
                    "exact_rate": value["exact"] / len(dev),
                }
                for radius, value in radii.items()
            }
            for budget, radii in budgets.items()
        }
        for mode, budgets in totals.items()
    }
    cell_rescue = cell_break = exact_rescue = exact_break = 0
    radius_exact_rescue = radius_exact_break = 0
    for row in per_image:
        offset = row["metrics"]["predicted_offset"]["1024"]["6"]
        center = row["metrics"]["cell_center"]["1024"]["6"]
        radius10 = row["metrics"]["predicted_offset"]["1024"]["10"]
        cell_rescue += int(center["matched"] > offset["matched"])
        cell_break += int(center["matched"] < offset["matched"])
        exact_rescue += int(center["exact"] > offset["exact"])
        exact_break += int(center["exact"] < offset["exact"])
        radius_exact_rescue += int(radius10["exact"] > offset["exact"])
        radius_exact_break += int(radius10["exact"] < offset["exact"])
    paired = {
        "cell_center_vs_offset_radius6": {
            "matched_image_rescue": cell_rescue,
            "matched_image_break": cell_break,
            "exact_rescue": exact_rescue,
            "exact_break": exact_break,
        },
        "predicted_radius10_vs_radius6": {
            "exact_rescue": radius_exact_rescue,
            "exact_break": radius_exact_break,
        },
    }
    decision, checks = choose_decision(metrics, paired)
    audit = {
        "schema_version": "g3-p4e7-offset-radius-ab-audit-0.1",
        "decision": decision,
        "checks": checks,
        "protocol": {
            "anchor_modes": list(ANCHOR_MODES),
            "budgets": list(BUDGETS),
            "radii_pixels": list(RADII),
            "candidate_identity_and_order_fixed": True,
            "training_performed": False,
        },
        "metrics": metrics,
        "paired": paired,
        "manifest": manifest_audit,
        "lineage": {
            "checkpoint": str(p4e6.P4E3_CHECKPOINT.resolve()),
            "checkpoint_sha256": p4e6.P4E3_CHECKPOINT_SHA256,
            "vision_checkpoint_sha256": vision_sha,
            "p4e6_audit_sha256": P4E6_AUDIT_SHA256,
        },
        "data_access": {
            "dev_png_read": True,
            "pilot_train_read": False,
            "confirm_v2_read": False,
            "fresh_dev_read": False,
            "external_and_sealed_read": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
