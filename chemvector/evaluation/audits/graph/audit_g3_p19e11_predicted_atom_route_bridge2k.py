#!/usr/bin/env python3
"""Bridge the frozen P19 relation router from GT to deployed predicted atoms."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.graph.correction.output_route_risk_router import (
    OutputRouteRiskRouter,
    OutputRouteRiskRouterConfig,
)
from chemvector.components.graph.joint.direct_atom_pair_bond import (
    enumerate_unordered_atom_pairs,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p18e0_proposal_oracle_ladder as p18e0,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p19e0_direct_atom_pair_oracle_ladder as p19e0,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p19e5_old_new_logit_fusion as p19e5,
)
from chemvector.evaluation.audits.graph import (
    audit_g3_p19e10_pairwise_route_fixed_confirm20k as p19e10,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.bonds.relations.train_bond_presence_calibration import (
    load_components,
)
from chemvector.training.graph.correction import (
    train_g3_p19e7_output_route_capacity640 as p19e7,
)
from chemvector.training.graph.correction import (
    train_g3_p19e9_pairwise_route_risk12k as p19e9,
)
from chemvector.training.graph.relations.train_direct_atom_pair_bond import (
    build_pair_targets,
    frozen_atom_objects,
    match_atom_targets,
)
from chemvector.training.graph.scenes.g3_structural_data import G3StructuralCollator
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.predicted_incidence import PredictedProposalConfig


AUDIT_NAME = "chemvector-g3-p19e11-predicted-atom-route-bridge2k"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
SCREEN_MANIFEST = OUTPUT / "screen2000.jsonl"
SCREEN_CACHE = OUTPUT / "predicted-route-cache.pt"
SOURCE_MANIFEST = p19e10.CONFIRM_MANIFEST
SOURCE_MANIFEST_SHA256 = (
    "f035a481689c5a03e106e02963fd01da44778696a0442d7d9653468706fd3a63"
)
SOURCE_RESULT = p19e10.OUTPUT / "result.json"
SOURCE_RESULT_SHA256 = (
    "1ac1dcf2a86005199fcc0ed99c3a589d2aeb9d9990da577451546cafb5e535d9"
)
ROUTER_CHECKPOINT = p19e9.OUTPUT / "best.pt"
ROUTER_CHECKPOINT_SHA256 = p19e10.SOURCE_CHECKPOINT_SHA256
SCREEN_MOLECULES = 500
SCREEN_IMAGES = SCREEN_MOLECULES * 4
BATCH_SIZE = 4
SEED = 20260810
EXPECTED_FEATURE_DIM = 78
EXPECTED_STEP = 250
EXPECTED_RULE = {"rescue_threshold": 0.5, "harmful_threshold": 0.02}
MIN_ATOM_EXACT_SUPPORT = 500
MIN_ROUTER_ELIGIBLE_RATE = 0.99
MIN_CONDITIONAL_FUSION_EXACT_RATE = 0.70
MIN_ORACLE_HEADROOM = 5
MIN_ACTIONS = 5
MAX_ACTION_RATE = 0.05
MIN_RESCUES = 3


def select_screen_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["molecule_id"])].append(row)
    candidates = []
    for molecule_id, current in grouped.items():
        variants = {str(row["variant"]) for row in current}
        scaffolds = {str(row.get("scaffold", "")) for row in current}
        if len(current) != 4 or variants != set(p19e9.VARIANTS):
            raise ValueError(f"P19-E11 incomplete source molecule: {molecule_id}")
        if len(scaffolds) != 1 or not next(iter(scaffolds)):
            raise ValueError(f"P19-E11 source scaffold differs: {molecule_id}")
        digest = hashlib.sha256(
            f"{SEED}\0p19e11-bridge\0{molecule_id}".encode()
        ).hexdigest()
        candidates.append((digest, molecule_id, next(iter(scaffolds))))
    candidates.sort()
    selected = candidates[:SCREEN_MOLECULES]
    if len(selected) != SCREEN_MOLECULES:
        raise ValueError("P19-E11 source has too few molecules")
    selected_molecules = {item[1] for item in selected}
    selected_scaffolds = {item[2] for item in selected}
    screen = [
        row for row in rows if str(row["molecule_id"]) in selected_molecules
    ]
    if len(screen) != SCREEN_IMAGES or len(selected_scaffolds) != SCREEN_MOLECULES:
        raise RuntimeError("P19-E11 screen count differs")
    return screen, {
        "source_images": len(rows),
        "images": len(screen),
        "molecules": len(selected_molecules),
        "scaffolds": len(selected_scaffolds),
        "sample_keys_sha256": hashlib.sha256(
            "".join(f"{row['sample_key']}\n" for row in screen).encode()
        ).hexdigest(),
        "source_was_already_consumed_by_p19e10": True,
        "new_scaffolds_consumed": 0,
    }


def route_exact_flags(
    selected: dict[str, torch.Tensor],
    presence: torch.Tensor,
    *,
    atom_exact: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_exact = torch.tensor(
        [
            torch.equal(selected[name].cpu(), presence.cpu())
            for name in ("fusion", "new", "old")
        ],
        dtype=torch.bool,
    )
    return pair_exact & bool(atom_exact), pair_exact


def structural_targets(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "atom_anchors": batch["atom_anchors"].to(device, non_blocking=True),
        "atom_mask": batch["atom_mask"].to(device, non_blocking=True),
        "bond_atom_refs": batch["bond_atom_refs"].to(device, non_blocking=True),
        "bond_chemical": batch["bond_types"].to(device, non_blocking=True),
        "bond_mask": batch["bond_mask"].to(device, non_blocking=True),
    }


def force_keep_for_ineligible(
    actions: torch.Tensor, router_eligible: torch.Tensor
) -> torch.Tensor:
    if actions.shape != router_eligible.shape:
        raise ValueError("P19-E11 action eligibility shape differs")
    return torch.where(
        router_eligible.to(actions.device), actions, torch.zeros_like(actions)
    )


def _records_to_cache(records: list[dict], feature_dim: int) -> dict:
    features = []
    for record in records:
        feature = record["features"]
        if feature is None:
            feature = torch.zeros(feature_dim, dtype=torch.float32)
        features.append(feature)
    route_exact = torch.stack([record["route_exact"] for record in records])
    return {
        "features": torch.stack(features),
        "route_exact": route_exact,
        "pair_exact": torch.stack([record["pair_exact"] for record in records]),
        "outcomes": p19e9.route_outcome_labels(route_exact),
        "router_eligible": torch.tensor(
            [record["router_eligible"] for record in records]
        ),
        "atom_exact": torch.tensor([record["atom_exact"] for record in records]),
        "bond_representable": torch.tensor(
            [record["bond_representable"] for record in records]
        ),
        "target_atoms": torch.tensor([record["target_atoms"] for record in records]),
        "selected_atoms": torch.tensor(
            [record["selected_atoms"] for record in records]
        ),
        "matched_atoms": torch.tensor(
            [record["matched_atoms"] for record in records]
        ),
        "target_bonds": torch.tensor([record["target_bonds"] for record in records]),
        "represented_bonds": torch.tensor(
            [record["represented_bonds"] for record in records]
        ),
        "observable_disagreement": torch.tensor(
            [record["routes_observably_disagree"] for record in records]
        ),
        "sample_keys": [record["sample_key"] for record in records],
        "variants": [record["variant"] for record in records],
        "stress_operations": [record["stress_operation"] for record in records],
    }


@torch.inference_mode()
def collect_predicted_route_cache(
    rows: list[dict],
    *,
    vision,
    atom,
    bond,
    set_head,
    new_model,
    old_model,
    proposal_config: PredictedProposalConfig,
    atom_threshold: float,
    expected_feature_names: list[str],
    device: torch.device,
) -> dict:
    samples = list(
        p1a.iter_manifest_samples(
            p18e0._manifest_records(rows), accepted_split="train"
        )
    )
    collator = G3StructuralCollator(
        maximum_atoms=proposal_config.maximum_atoms,
        maximum_bonds=proposal_config.maximum_bonds,
        image_size=proposal_config.image_size,
    )
    records = []
    for start in range(0, len(samples), BATCH_SIZE):
        current = samples[start : start + BATCH_SIZE]
        batch = collator(current)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = structural_targets(batch, device)
        feature_map, atoms = frozen_atom_objects(
            vision,
            atom,
            bond,
            set_head,
            pixels,
            proposal_config,
            atom_threshold,
        )
        mapping, _ = match_atom_targets(atoms, targets, proposal_config)
        pairs, sample_indices = enumerate_unordered_atom_pairs(atoms["atom_mask"])
        pair_targets = build_pair_targets(
            pairs, sample_indices, mapping, targets, atoms
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            new_output = new_model(
                feature_map,
                atoms["atom_features"],
                atoms["atom_geometry"][..., :2],
                atoms["atom_scores"],
                pairs,
                sample_indices,
            )
            old_output = old_model(
                feature_map,
                atoms["atom_features"],
                atoms["atom_geometry"][..., :2],
                atoms["atom_scores"],
                pairs,
                sample_indices,
            )
        new_probabilities = new_output["presence_logits"].sigmoid().float().cpu()
        old_probabilities = old_output["presence_logits"].sigmoid().float().cpu()
        presence = pair_targets["presence"].cpu()
        for batch_index, sample in enumerate(current):
            pair_rows = torch.where(sample_indices == batch_index)[0].cpu()
            target_atoms = int(pair_targets["target_atoms"][batch_index])
            selected_atoms = int(pair_targets["selected_atoms"][batch_index])
            matched_atoms = int(pair_targets["matched_atoms"][batch_index])
            target_bonds = int(pair_targets["target_bonds"][batch_index])
            current_presence = presence[pair_rows]
            represented_bonds = int(current_presence.sum())
            atom_exact = (
                target_atoms == selected_atoms == matched_atoms
            )
            router_eligible = len(pair_rows) > 0
            if router_eligible:
                names, features, selected = p19e7.build_observable_features(
                    new_probabilities[pair_rows], old_probabilities[pair_rows]
                )
                if names != expected_feature_names:
                    raise RuntimeError("P19-E11 feature schema differs")
            else:
                features = None
                selected = {
                    name: torch.zeros(0, dtype=torch.bool)
                    for name in ("fusion", "new", "old")
                }
            route_exact, pair_exact = route_exact_flags(
                selected, current_presence, atom_exact=atom_exact
            )
            variant = str(sample["render"]["variant"])
            operation = (
                str(
                    sample["render"].get("render_parameters", {}).get(
                        "stress_operation", "unknown"
                    )
                )
                if variant == "stress"
                else None
            )
            records.append(
                {
                    "sample_key": str(sample["key"]),
                    "variant": variant,
                    "stress_operation": operation,
                    "features": features,
                    "route_exact": route_exact,
                    "pair_exact": pair_exact,
                    "router_eligible": router_eligible,
                    "atom_exact": atom_exact,
                    "bond_representable": represented_bonds == target_bonds,
                    "target_atoms": target_atoms,
                    "selected_atoms": selected_atoms,
                    "matched_atoms": matched_atoms,
                    "target_bonds": target_bonds,
                    "represented_bonds": represented_bonds,
                    "routes_observably_disagree": bool(
                        (selected["new"] != selected["old"]).any()
                        or (selected["new"] != selected["fusion"]).any()
                    ),
                }
            )
    if len(records) != len(rows):
        raise RuntimeError("P19-E11 collected image count differs")
    return _records_to_cache(records, len(expected_feature_names))


def evaluation_records(cache: dict) -> list[dict]:
    records = []
    for index in range(len(cache["route_exact"])):
        route_exact = cache["route_exact"][index]
        records.append(
            {
                "sample_key": cache["sample_keys"][index],
                "variant": cache["variants"][index],
                "stress_operation": cache["stress_operations"][index],
                "route_exact": route_exact,
                "target_action": p19e7._route_label(route_exact),
                "routes_observably_disagree": bool(
                    cache["observable_disagreement"][index]
                ),
            }
        )
    return records


def route_summary(cache: dict) -> dict:
    atom_exact = cache["atom_exact"]
    route_exact = cache["route_exact"]
    pair_exact = cache["pair_exact"]
    target_atoms = int(cache["target_atoms"].sum())
    matched_atoms = int(cache["matched_atoms"].sum())
    names = ("fusion", "new", "old")
    return {
        "images": len(atom_exact),
        "atom_exact": int(atom_exact.sum()),
        "atom_exact_rate": float(atom_exact.float().mean()),
        "bond_representable": int(cache["bond_representable"].sum()),
        "router_eligible": int(cache["router_eligible"].sum()),
        "target_atoms": target_atoms,
        "selected_atoms": int(cache["selected_atoms"].sum()),
        "matched_atoms": matched_atoms,
        "matched_atom_recall": matched_atoms / max(target_atoms, 1),
        "pair_exact": {
            name: int(pair_exact[:, index].sum()) for index, name in enumerate(names)
        },
        "skeleton_exact": {
            name: int(route_exact[:, index].sum()) for index, name in enumerate(names)
        },
        "route_oracle": int(route_exact.any(1).sum()),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P19-E11 output is not empty: {OUTPUT}")
    inputs = (
        (SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256),
        (SOURCE_RESULT, SOURCE_RESULT_SHA256),
        (ROUTER_CHECKPOINT, ROUTER_CHECKPOINT_SHA256),
        (p19e9.NEW_CHECKPOINT, p19e9.NEW_CHECKPOINT_SHA256),
        (p19e9.OLD_CHECKPOINT, p19e9.OLD_CHECKPOINT_SHA256),
        (p19e0.SOURCE_CHECKPOINT, p19e0.SOURCE_CHECKPOINT_SHA256),
        (p19e0.VISION_CHECKPOINT, p19e0.VISION_CHECKPOINT_SHA256),
    )
    for path, expected in inputs:
        if file_sha256(path) != expected:
            raise ValueError(f"P19-E11 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P19-E11 requires CUDA")
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

    source_rows = p18e0._jsonl(SOURCE_MANIFEST)
    screen_rows, split_audit = select_screen_rows(source_rows)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    SCREEN_MANIFEST.write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n" for row in screen_rows
        ),
        encoding="utf-8",
    )
    split_audit["manifest_sha256"] = file_sha256(SCREEN_MANIFEST)
    _atomic_json(OUTPUT / "split-audit.json", split_audit)

    router_payload = torch.load(
        ROUTER_CHECKPOINT, map_location="cpu", weights_only=False
    )
    if router_payload["global_step"] != EXPECTED_STEP:
        raise ValueError("P19-E11 router step differs")
    if router_payload["risk_rule"] != EXPECTED_RULE:
        raise ValueError("P19-E11 router rule differs")
    if len(router_payload["feature_names"]) != EXPECTED_FEATURE_DIM:
        raise ValueError("P19-E11 router feature dimension differs")
    router = OutputRouteRiskRouter(
        OutputRouteRiskRouterConfig(**router_payload["config"])
    ).to(device)
    router.load_state_dict(router_payload["router"], strict=True)
    router.eval().requires_grad_(False)

    vision, atom, bond, set_head, source = load_components(
        str(p19e0.SOURCE_CHECKPOINT), str(p19e0.VISION_CHECKPOINT), device
    )
    for module in (vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    new_model = p19e5._load_model(p19e9.NEW_CHECKPOINT, device)
    old_model = p19e5._load_model(p19e9.OLD_CHECKPOINT, device)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    modules = {
        "vision": vision,
        "atom": atom,
        "bond": bond,
        "set_head": set_head,
        "new": new_model,
        "old": old_model,
        "router": router,
    }
    states_before = {
        name: module_state_sha256(module) for name, module in modules.items()
    }
    cache = collect_predicted_route_cache(
        screen_rows,
        vision=vision,
        atom=atom,
        bond=bond,
        set_head=set_head,
        new_model=new_model,
        old_model=old_model,
        proposal_config=proposal_config,
        atom_threshold=atom_threshold,
        expected_feature_names=router_payload["feature_names"],
        device=device,
    )
    _atomic_torch_save(
        SCREEN_CACHE,
        {
            "schema_version": "g3-p19e11-predicted-atom-route-cache-0.1",
            "split": cache,
            "feature_names": router_payload["feature_names"],
            "source_checkpoint_sha256": ROUTER_CHECKPOINT_SHA256,
            "fixed_risk_rule": EXPECTED_RULE,
            "atom_match_radius_px": proposal_config.atom_match_radius_px,
        },
    )
    cache_sha256 = file_sha256(SCREEN_CACHE)
    normalized = (
        (cache["features"] - router_payload["feature_mean"])
        / router_payload["feature_std"]
    ).to(device)
    with torch.inference_mode():
        logits = router(normalized)
        actions = p19e9.actions_from_risk_logits(
            logits,
            EXPECTED_RULE["rescue_threshold"],
            EXPECTED_RULE["harmful_threshold"],
        ).cpu()
    actions = force_keep_for_ineligible(actions, cache["router_eligible"])
    records = evaluation_records(cache)
    metrics = p19e9.evaluate_risk_actions(actions, records)
    atom_mask = cache["atom_exact"]
    atom_records = [record for record, keep in zip(records, atom_mask.tolist()) if keep]
    atom_actions = actions[atom_mask]
    atom_exact_metrics = p19e9.evaluate_risk_actions(atom_actions, atom_records)
    summary = route_summary(cache)
    outcome_counts = p19e10.selected_outcome_counts(actions, cache)
    normalized_cpu = normalized.float().cpu()
    feature_drift = {
        "all_finite": bool(torch.isfinite(normalized_cpu).all()),
        "absolute_mean": float(normalized_cpu.abs().mean()),
        "absolute_p95": float(torch.quantile(normalized_cpu.abs(), 0.95)),
        "absolute_max": float(normalized_cpu.abs().max()),
    }
    overall = metrics["overall"]
    action_count = SCREEN_IMAGES - int(overall["actions"].get("KEEP_FUSION", 0))
    action_rate = action_count / SCREEN_IMAGES
    fusion_exact = int(overall["fusion_exact"])
    route_oracle_headroom = summary["route_oracle"] - fusion_exact
    conditional_fusion_rate = (
        atom_exact_metrics["overall"]["fusion_exact"] / max(int(atom_mask.sum()), 1)
    )
    states_after = {
        name: module_state_sha256(module) for name, module in modules.items()
    }
    checks = {
        "screen_counts_exact": (
            split_audit["images"] == SCREEN_IMAGES
            and split_audit["molecules"] == SCREEN_MOLECULES
            and split_audit["scaffolds"] == SCREEN_MOLECULES
        ),
        "uses_only_consumed_p19e10_data": (
            split_audit["source_was_already_consumed_by_p19e10"]
            and split_audit["new_scaffolds_consumed"] == 0
        ),
        "feature_schema_and_values_valid": (
            cache["features"].shape == (SCREEN_IMAGES, EXPECTED_FEATURE_DIM)
            and feature_drift["all_finite"]
        ),
        "router_eligible_rate_min99pct": (
            float(cache["router_eligible"].float().mean()) >= MIN_ROUTER_ELIGIBLE_RATE
        ),
        "atom_exact_support_min500": int(atom_mask.sum()) >= MIN_ATOM_EXACT_SUPPORT,
        "conditional_fusion_exact_rate_min70pct": (
            conditional_fusion_rate >= MIN_CONDITIONAL_FUSION_EXACT_RATE
        ),
        "route_oracle_headroom_min5": route_oracle_headroom >= MIN_ORACLE_HEADROOM,
        "action_count_min5": action_count >= MIN_ACTIONS,
        "action_rate_max5pct": action_rate <= MAX_ACTION_RATE,
        "overall_rescue_min3": int(overall["rescue"]) >= MIN_RESCUES,
        "overall_zero_harmful": int(overall["harmful"]) == 0,
        "atom_exact_subset_zero_harmful": (
            atom_exact_metrics["overall"]["harmful"] == 0
        ),
        "selected_exact_improves_fusion": (
            overall["selected_exact"] > overall["fusion_exact"]
        ),
        "frozen_sources_unchanged": states_before == states_after,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    decision = (
        "predicted_atom_route_bridge2k_go"
        if all(checks.values())
        else "predicted_atom_route_bridge2k_no_go"
    )
    result = {
        "schema_version": "g3-p19e11-predicted-atom-route-bridge2k-result-0.1",
        "audit_name": AUDIT_NAME,
        "decision": decision,
        "checks": checks,
        "fixed_router_checkpoint": str(ROUTER_CHECKPOINT.resolve()),
        "fixed_router_checkpoint_sha256": ROUTER_CHECKPOINT_SHA256,
        "fixed_step": EXPECTED_STEP,
        "fixed_risk_rule": EXPECTED_RULE,
        "atom_source": {
            "source_checkpoint": str(p19e0.SOURCE_CHECKPOINT.resolve()),
            "vision_checkpoint": str(p19e0.VISION_CHECKPOINT.resolve()),
            "decision_threshold": atom_threshold,
            "match_radius_px": proposal_config.atom_match_radius_px,
        },
        "route_summary": summary,
        "router_metrics": metrics,
        "atom_exact_subset_router_metrics": atom_exact_metrics,
        "selected_outcomes": outcome_counts,
        "feature_drift": feature_drift,
        "action_count": action_count,
        "action_rate": action_rate,
        "route_oracle_headroom": route_oracle_headroom,
        "conditional_fusion_exact_rate": conditional_fusion_rate,
        "cache": str(SCREEN_CACHE.resolve()),
        "cache_sha256": cache_sha256,
        "split": split_audit,
        "gt_fields_forwarded_to_router": False,
        "gt_used_only_for_posthoc_matching_and_metrics": True,
        "model_or_rule_selection_on_screen": False,
        "fresh_scaffolds_read": False,
        "sealed_test_read": False,
        "mr_tadf_read": False,
        "uspto_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
