#!/usr/bin/env python3
"""Replay the fixed P19-E1 Direct AtomPair checkpoint on fresh640."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.graph.joint.direct_atom_pair_bond import (
    DirectAtomPairBond,
    DirectAtomPairBondConfig,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p18e0_proposal_oracle_ladder as p18e0,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p19e0_direct_atom_pair_oracle_ladder as p19e0,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.bonds.relations.train_bond_presence_calibration import (
    load_components,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p19e2-direct-atom-pair-fresh640"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
DIRECT_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-g3-p19e1-direct-atom-pair-capacity64/best.pt"
)
DIRECT_CHECKPOINT_SHA256 = (
    "b22f23d1490e223a759a86dd230afd50d5e7f91e8036ecb45dbf22c6e11f9d47"
)
P19E1_RESULT = Path(
    "outputs/checkpoints/chemvector-g3-p19e1-direct-atom-pair-capacity64/result.json"
)
P19E1_RESULT_SHA256 = (
    "08c11a11c3521da93e98a347d9833bd39344631f29fc6a176f5146570ce72305"
)
OLD_ROWS = p19e0.OUTPUT / "rows.json"
OLD_ROWS_SHA256 = "6bba91215c28ef9c630d451eeab77f6f94a85bd6230a61b310ca95f926091e52"
THRESHOLD = 0.5
SEED = 20260810
GATES = {
    "gt_atom_overall_skeleton_exact_min": 512,
    "gt_atom_stress_skeleton_exact_min": 96,
    "gt_atom_crossed_skeleton_exact_min": 8,
    "gt_atom_near_nonbonded_skeleton_exact_min": 12,
    "gt_atom_nonstress_skeleton_exact_min": 405,
    "gt_atom_edge_recall_min": 0.97,
    "p4_atom_coverage_exact": 608,
}


def paired_summary(old_rows: list[dict], new_rows: list[dict]) -> dict:
    old_by_key = {str(row["sample_key"]): row for row in old_rows}
    new_by_key = {str(row["sample_key"]): row for row in new_rows}
    if len(old_by_key) != len(old_rows) or len(new_by_key) != len(new_rows):
        raise ValueError("P19-E2 paired rows contain duplicate keys")
    if set(old_by_key) != set(new_by_key):
        raise ValueError("P19-E2 paired sample keys differ")
    rows = []
    for key in sorted(old_by_key):
        old = bool(old_by_key[key]["skeleton_exact"])
        new = bool(new_by_key[key]["skeleton_exact"])
        rows.append(
            {
                "sample_key": key,
                "variant": str(new_by_key[key]["variant"]),
                "stress_operation": new_by_key[key].get("stress_operation"),
                "old_skeleton_exact": old,
                "new_skeleton_exact": new,
                "rescue": new and not old,
                "break": old and not new,
            }
        )

    def summary(current: list[dict]) -> dict:
        return {
            "images": len(current),
            "old_exact": sum(int(row["old_skeleton_exact"]) for row in current),
            "new_exact": sum(int(row["new_skeleton_exact"]) for row in current),
            "rescue": sum(int(row["rescue"]) for row in current),
            "break": sum(int(row["break"]) for row in current),
        }

    operations = sorted(
        {
            str(row["stress_operation"])
            for row in rows
            if row["stress_operation"] is not None
        }
    )
    return {
        "overall": summary(rows),
        "by_variant": {
            variant: summary([row for row in rows if row["variant"] == variant])
            for variant in p18e0.VARIANTS
        },
        "stress_by_operation": {
            operation: summary(
                [row for row in rows if row["stress_operation"] == operation]
            )
            for operation in operations
        },
        "rows": rows,
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P19-E2 output is not empty: {OUTPUT}")
    inputs = (
        (DIRECT_CHECKPOINT, DIRECT_CHECKPOINT_SHA256),
        (P19E1_RESULT, P19E1_RESULT_SHA256),
        (OLD_ROWS, OLD_ROWS_SHA256),
        (p19e0.SOURCE_CHECKPOINT, p19e0.SOURCE_CHECKPOINT_SHA256),
        (p19e0.VISION_CHECKPOINT, p19e0.VISION_CHECKPOINT_SHA256),
        (p19e0.E0_CACHE, p19e0.E0_CACHE_SHA256),
        (p19e0.E0_MANIFEST, p19e0.E0_MANIFEST_SHA256),
        (p19e0.E0A_CACHE, p19e0.E0A_CACHE_SHA256),
    )
    for path, expected in inputs:
        if file_sha256(path) != expected:
            raise ValueError(f"P19-E2 input SHA256 differs: {path}")
    result_e1 = json.loads(P19E1_RESULT.read_text(encoding="utf-8"))
    if result_e1["decision"] != "direct_atom_pair_capacity64_go":
        raise ValueError("P19-E2 requires the fixed P19-E1 capacity GO")
    if not torch.cuda.is_available():
        raise RuntimeError("P19-E2 requires CUDA BF16")
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
    started = time.perf_counter()

    source_payload = torch.load(
        p19e0.SOURCE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    graph_checkpoint = Path(
        source_payload["training_args"]["grounded_graph_checkpoint"]
    )
    if graph_checkpoint.resolve() != p19e0.VISION_CHECKPOINT.resolve():
        raise ValueError("P19-E2 source is bound to another vision checkpoint")
    vision, atom, bond, set_head, _ = load_components(
        str(p19e0.SOURCE_CHECKPOINT), str(p19e0.VISION_CHECKPOINT), device
    )
    for module in (vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    direct_payload = torch.load(
        DIRECT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    if direct_payload["model_type"] != result_e1["model_type"]:
        raise ValueError("P19-E2 checkpoint model type differs")
    if direct_payload["initialized_from_old_direct"]:
        raise ValueError("P19-E2 expected the scratch P19-E1 checkpoint")
    if float(direct_payload["threshold"]) != THRESHOLD:
        raise ValueError("P19-E2 fixed threshold differs")
    model = DirectAtomPairBond(
        DirectAtomPairBondConfig(**direct_payload["config"])
    ).to(device)
    model.load_state_dict(direct_payload["model"], strict=True)
    model.eval().requires_grad_(False)
    states_before = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "direct": module_state_sha256(model),
    }

    manifest_rows = p18e0._jsonl(p19e0.E0_MANIFEST)
    samples = list(
        p1a.iter_manifest_samples(
            p18e0._manifest_records(manifest_rows),
            accepted_split=p18e0.freshdev.manifest_protocol.EXPECTED_SPLIT,
        )
    )
    e0 = torch.load(p19e0.E0_CACHE, map_location="cpu", weights_only=False)
    p4_records = e0["records"]
    stress = torch.load(p19e0.E0A_CACHE, map_location="cpu", weights_only=False)
    operations = {
        str(record["sample_key"]): str(record["stress_operation"])
        for record in stress["records"]
    }
    if [str(sample["key"]) for sample in samples] != [
        str(record["sample_key"]) for record in p4_records
    ]:
        raise RuntimeError("P19-E2 sample/cache order differs")
    evaluated = p19e0.evaluate(
        model,
        vision,
        atom,
        samples,
        p4_records,
        operations,
        threshold=THRESHOLD,
        device=device,
    )
    states_after = {
        "vision": module_state_sha256(vision),
        "atom": module_state_sha256(atom),
        "direct": module_state_sha256(model),
    }
    old = json.loads(OLD_ROWS.read_text(encoding="utf-8"))
    paired = {
        route: paired_summary(old[route], evaluated["routes"][route]["rows"])
        for route in ("gt_atom", "p4_oracle_atom")
    }
    gt = evaluated["routes"]["gt_atom"]
    p4 = evaluated["routes"]["p4_oracle_atom"]
    nonstress_exact = sum(
        int(gt["by_variant"][variant]["skeleton_exact"])
        for variant in ("coordgen", "original", "rdkit")
    )
    checks = {
        "gt_atom_overall_skeleton_exact_min512": (
            gt["overall"]["skeleton_exact"]
            >= GATES["gt_atom_overall_skeleton_exact_min"]
        ),
        "gt_atom_stress_skeleton_exact_min96": (
            gt["by_variant"]["stress"]["skeleton_exact"]
            >= GATES["gt_atom_stress_skeleton_exact_min"]
        ),
        "gt_atom_crossed_skeleton_exact_min8": (
            gt["stress_by_operation"]["crossed_bonds"]["skeleton_exact"]
            >= GATES["gt_atom_crossed_skeleton_exact_min"]
        ),
        "gt_atom_near_nonbonded_skeleton_exact_min12": (
            gt["stress_by_operation"]["near_nonbonded"]["skeleton_exact"]
            >= GATES["gt_atom_near_nonbonded_skeleton_exact_min"]
        ),
        "gt_atom_nonstress_skeleton_exact_min405": (
            nonstress_exact >= GATES["gt_atom_nonstress_skeleton_exact_min"]
        ),
        "gt_atom_edge_recall_min97": (
            gt["overall"]["edge_recall"] >= GATES["gt_atom_edge_recall_min"]
        ),
        "gt_atom_net_paired_improvement": (
            paired["gt_atom"]["overall"]["rescue"]
            > paired["gt_atom"]["overall"]["break"]
        ),
        "p4_atom_coverage_matches_608": (
            p4["overall"]["atom_exact"] == GATES["p4_atom_coverage_exact"]
        ),
        "all_frozen_states_unchanged": states_before == states_after,
    }
    decision = (
        "direct_atom_pair_fresh640_go"
        if all(checks.values())
        else "direct_atom_pair_fresh640_no_go"
    )
    route_rows = {
        route: values.pop("rows") for route, values in evaluated["routes"].items()
    }
    paired_rows = {
        route: values.pop("rows") for route, values in paired.items()
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "rows.json", route_rows)
    _atomic_json(OUTPUT / "paired-rows.json", paired_rows)
    report = {
        "schema_version": "g3-p19e2-direct-atom-pair-fresh640-0.1",
        "decision": decision,
        "checks": checks,
        "gates": GATES,
        "threshold": THRESHOLD,
        "routes": evaluated["routes"],
        "paired_vs_v0103": paired,
        "gt_atom_nonstress_skeleton_exact": nonstress_exact,
        "prediction_sha256": evaluated["prediction_sha256"],
        "sources": {
            "direct_checkpoint": str(DIRECT_CHECKPOINT.resolve()),
            "direct_checkpoint_sha256": DIRECT_CHECKPOINT_SHA256,
            "p19e1_result": str(P19E1_RESULT.resolve()),
            "p19e1_result_sha256": P19E1_RESULT_SHA256,
            "old_rows": str(OLD_ROWS.resolve()),
            "old_rows_sha256": OLD_ROWS_SHA256,
        },
        "training_or_calibration": False,
        "checkpoint_or_threshold_selection_on_fresh640": False,
        "gt_atom_is_oracle_audit_only": True,
        "p4_selection_uses_gt_after_target_free_candidate_forward": True,
        "dense_bond_forward": False,
        "incidence_forward": False,
        "sealed_test_read": False,
        "confirm_read": False,
        "mr_tadf_read": False,
        "uspto_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
