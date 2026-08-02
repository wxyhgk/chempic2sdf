#!/usr/bin/env python3
"""Audit the four fixed P0-K relation-alignment checkpoints without replay."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import textwrap

import torch

from chemvector.audit_g3_p0j_frozen_composition import (
    P0G_CHECKPOINT_SHA256,
    P0I_CHECKPOINT_SHA256,
)
from chemvector.direct_atom_pair_bond import (
    DirectAtomPairBond,
    DirectAtomPairBondConfig,
)
from chemvector.g3_structural_data import (
    OVERFIT32_MANIFEST_SHA256,
    file_sha256,
)
from chemvector.train_g3_hierarchical_graph import (
    DEFAULT_VISION_CHECKPOINT_SHA256,
    module_state_sha256,
)
from chemvector import train_g3_p0k_relation_alignment as p0k_trainer


DEFAULT_ARTIFACT_DIR = Path(
    "outputs/smoke/chemvector-g3-p0k-predicted-anchor-relation-alignment"
)
EXPECTED_STEPS = (100, 200, 300, 400)
EXPECTED_IMAGES = 32
EXPECTED_ATOMS = 1424
EXPECTED_BONDS = 1508
EXPECTED_PAIRS = 37380
EXPECTED_PAIR_PARAMETERS = 2_364_869
EXPECTED_PAIR_TENSORS = 32
P0J_AUDIT = Path(
    "outputs/smoke/chemvector-g3-p0j-frozen-relation-composition/"
    "composition_audit.json"
)
P0J_BASE_TYPED_EXACT = 26


def pair_config_from_g3(g3_config: dict) -> DirectAtomPairBondConfig:
    required = {"input_channels", "d_model"}
    if not isinstance(g3_config, dict) or not required <= g3_config.keys():
        raise ValueError("P0-K source P0-G checkpoint has no compatible G3 config")
    return DirectAtomPairBondConfig(
        feature_channels=int(g3_config["input_channels"]),
        atom_feature_dim=int(g3_config["d_model"]),
        longitudinal_samples=32,
        transverse_samples=9,
        ribbon_radius_pixels=12.0,
        image_size=768.0,
        hidden_dim=256,
        pair_chunk_size=512,
    )


def pair_head_from_state(
    state: dict[str, torch.Tensor], config: DirectAtomPairBondConfig
) -> DirectAtomPairBond:
    if not isinstance(state, dict) or not state:
        raise ValueError("P0-K checkpoint contains no pair-head state")
    model = DirectAtomPairBond(config)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def source_pair_head(
    p0g_payload: dict,
) -> tuple[DirectAtomPairBond, DirectAtomPairBondConfig]:
    config = pair_config_from_g3(p0g_payload.get("g3_config"))
    prefix = "pair_bond."
    state = {
        name.removeprefix(prefix): value
        for name, value in p0g_payload.get("model", {}).items()
        if name.startswith(prefix)
    }
    return pair_head_from_state(state, config), config


def _subscript_base_and_key(node: ast.AST) -> tuple[str | None, str | None]:
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return None, None
    key = node.slice.value if isinstance(node.slice, ast.Constant) else None
    return node.value.id, key if isinstance(key, str) else None


def validate_pair_forward_source(source: str) -> dict[str, int | bool]:
    """Prove the pair forward consumes predicted Atom anchors, not GT anchors."""

    tree = ast.parse(textwrap.dedent(source))
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "p0k_training_step"
    ]
    if len(functions) != 1:
        raise ValueError("P0-K trainer source has no unique training step")
    pair_calls = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pair_bond"
    ]
    if len(pair_calls) != 1:
        raise ValueError("P0-K training step has no unique pair forward")
    anchor_inputs = [
        _subscript_base_and_key(node)
        for node in ast.walk(pair_calls[0])
        if isinstance(node, ast.Subscript)
    ]
    predicted_anchor_inputs = sum(
        base == "atoms" and key == "atom_anchors" for base, key in anchor_inputs
    )
    gt_anchor_inputs = sum(
        base in {"target", "targets"} and key == "atom_anchors"
        for base, key in anchor_inputs
    )
    if predicted_anchor_inputs != 1 or gt_anchor_inputs:
        raise ValueError("P0-K pair forward does not exclusively use predicted anchors")
    return {
        "pair_forward_calls": len(pair_calls),
        "predicted_anchor_inputs": predicted_anchor_inputs,
        "gt_anchor_inputs": gt_anchor_inputs,
        "gt_anchor_in_pair_forward": False,
    }


def validate_optimizer_scope(
    payload: dict,
    pair_head: DirectAtomPairBond,
    expected_parameter_names: list[str],
) -> dict[str, int | bool]:
    state_names = list(pair_head.state_dict())
    parameter_names = [name for name, _ in pair_head.named_parameters()]
    if state_names != list(payload.get("pair_bond", {})):
        raise ValueError("P0-K checkpoint pair-head state order differs")
    if parameter_names != expected_parameter_names:
        raise ValueError("P0-K trainable names differ from DirectAtomPairBond")
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict) or not isinstance(
        optimizer.get("param_groups"), list
    ):
        raise ValueError("P0-K checkpoint has no optimizer state")
    identifiers = [
        identifier
        for group in optimizer["param_groups"]
        for identifier in group.get("params", [])
    ]
    if len(optimizer["param_groups"]) != 1:
        raise ValueError("P0-K optimizer must contain one pair-head parameter group")
    if len(identifiers) != len(parameter_names) or len(set(identifiers)) != len(
        parameter_names
    ):
        raise ValueError("P0-K optimizer does not contain exactly the pair head")
    if not set(optimizer.get("state", {})).issubset(identifiers):
        raise ValueError("P0-K optimizer contains state outside its parameter group")
    return {
        "parameter_tensors": len(parameter_names),
        "optimizer_parameter_tensors": len(identifiers),
        "optimizer_state_tensors": len(optimizer.get("state", {})),
        "pair_head_only": True,
    }


def coverage_from_metrics(metrics: dict) -> dict[str, int | bool]:
    predicted = metrics.get("predicted_atom_composition", {})
    coverage = {
        "images": int(predicted.get("images", -1)),
        "atom_targets": int(predicted.get("atom_targets", -1)),
        "atom_matched": int(predicted.get("atom_matched", -1)),
        "atom_set_exact_images": int(predicted.get("atom_set_exact_images", -1)),
        "target_bonds": int(predicted.get("target_bonds", -1)),
        "representable_bonds": int(predicted.get("representable_bonds", -1)),
        "pair_universe": int(predicted.get("pair_universe", -1)),
    }
    coverage["atom_mapping_complete"] = (
        coverage["images"] == EXPECTED_IMAGES
        and coverage["atom_targets"] == EXPECTED_ATOMS
        and coverage["atom_matched"] == EXPECTED_ATOMS
        and coverage["atom_set_exact_images"] == EXPECTED_IMAGES
    )
    coverage["bond_coverage_complete"] = (
        coverage["target_bonds"] == EXPECTED_BONDS
        and coverage["representable_bonds"] == EXPECTED_BONDS
    )
    coverage["pair_universe_complete"] = coverage["pair_universe"] == EXPECTED_PAIRS
    if not all(
        coverage[name]
        for name in (
            "atom_mapping_complete",
            "bond_coverage_complete",
            "pair_universe_complete",
        )
    ):
        raise ValueError("P0-K stored coverage differs from the fixed protocol")
    return coverage


def strict_final_capability(metrics: dict, p0j_audit: dict) -> dict[str, int | bool]:
    predicted = metrics.get("predicted_atom_composition", {})
    oracle = metrics.get("gt_atom_oracle_replay", {})
    base = p0j_audit.get("p0g_slot_replay_feature", {})
    values = {
        "predicted_gt_typed_graph_exact_images": int(
            predicted.get("gt_bond_count", {}).get("typed_graph_exact_images", -1)
        ),
        "predicted_deploy_typed_graph_exact_images": int(
            predicted.get("deploy_bond_count", {}).get(
                "typed_graph_exact_images", -1
            )
        ),
        "oracle_gt_structural_graph_exact_images": int(
            oracle.get("gt_count", {}).get("structural_graph_exact_images", -1)
        ),
        "oracle_deploy_structural_graph_exact_images": int(
            oracle.get("deploy_count", {}).get("structural_graph_exact_images", -1)
        ),
        "p0j_base_gt_typed_graph_exact_images": int(
            base.get("gt_bond_count", {}).get("typed_graph_exact_images", -1)
        ),
    }
    if values["p0j_base_gt_typed_graph_exact_images"] != P0J_BASE_TYPED_EXACT:
        raise ValueError("P0-J frozen-composition baseline differs from 26/32")
    strict_names = (
        "predicted_gt_typed_graph_exact_images",
        "predicted_deploy_typed_graph_exact_images",
        "oracle_gt_structural_graph_exact_images",
        "oracle_deploy_structural_graph_exact_images",
    )
    values["all_four_branches_32_of_32"] = all(
        values[name] == EXPECTED_IMAGES for name in strict_names
    )
    values["p0j_per_sample_break_impossible"] = (
        values["predicted_gt_typed_graph_exact_images"] == EXPECTED_IMAGES
    )
    values["p0j_exact_image_improvement"] = (
        values["predicted_gt_typed_graph_exact_images"]
        - values["p0j_base_gt_typed_graph_exact_images"]
    )
    if not values["all_four_branches_32_of_32"]:
        raise ValueError("P0-K strict final capability requires four 32/32 branches")
    return values


def validate_checkpoint(
    payload: dict,
    *,
    step: int,
    audit: dict,
    pair_config: DirectAtomPairBondConfig,
    expected_parameter_names: list[str],
) -> dict:
    if payload.get("checkpoint_schema_version") != (
        "g3-p0k-relation-alignment-checkpoint-0.1"
    ):
        raise ValueError("P0-K checkpoint schema differs")
    if payload.get("model_type") != p0k_trainer.MODEL_TYPE:
        raise ValueError("P0-K checkpoint model type differs")
    if payload.get("global_step") != step:
        raise ValueError("P0-K checkpoint filename and global step differ")
    if payload.get("samples_seen") != step * 4:
        raise ValueError("P0-K checkpoint sample count differs")
    scope = payload.get("training_scope", {})
    if scope != {
        "predicted_atom_anchors": True,
        "gt_anchor_in_pair_forward": False,
        "p0g_slot_replay_features": True,
        "train_pair_bond_only": True,
    }:
        raise ValueError("P0-K checkpoint training scope differs")
    args = payload.get("training_args", {})
    fixed_args = {
        "max_steps": 400,
        "batch_size": 4,
        "vision_batch_size": 4,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "log_every": 20,
        "evaluate_every": 100,
        "save_every": 100,
        "seed": 20260713,
        "device": "cuda",
        "precision": "bf16",
    }
    if any(args.get(name) != value for name, value in fixed_args.items()):
        raise ValueError("P0-K checkpoint training arguments differ")
    path_args = {
        "output_dir": DEFAULT_ARTIFACT_DIR,
        "sample_manifest": Path(audit["sample_manifest"]["path"]),
        "p0i_checkpoint": Path(audit["checkpoints"]["p0i"]["path"]),
        "p0g_checkpoint": Path(audit["checkpoints"]["p0g"]["path"]),
    }
    if any(
        Path(args.get(name, "")).resolve() != expected.resolve()
        for name, expected in path_args.items()
    ):
        raise ValueError("P0-K checkpoint path arguments differ")
    for name, audit_name in (
        ("source_p0i_checkpoint", "p0i"),
        ("source_p0g_checkpoint", "p0g"),
    ):
        if payload.get(name) != audit["checkpoints"][audit_name]:
            raise ValueError(f"P0-K checkpoint {name} differs from audit")
    if payload.get("source_vision_checkpoint") != audit["vision"]:
        raise ValueError("P0-K checkpoint vision source differs from audit")
    if payload.get("sample_manifest") != audit["sample_manifest"]:
        raise ValueError("P0-K checkpoint manifest differs from audit")
    metrics = payload.get("train_only_metrics", {})
    if (
        not isinstance(metrics.get("p0k_gate"), dict)
        or metrics.get("step") != step
    ):
        raise ValueError("P0-K checkpoint has no stored P0-K gate")
    pair_head = pair_head_from_state(payload.get("pair_bond", {}), pair_config)
    optimizer = validate_optimizer_scope(
        payload, pair_head, expected_parameter_names
    )
    return {
        "step": step,
        "pair_bond_state_sha256": module_state_sha256(pair_head),
        "optimizer": optimizer,
        "coverage": coverage_from_metrics(metrics),
        "metrics": metrics,
    }


def validate_training_audit(audit: dict, artifact_dir: Path) -> tuple[dict, dict]:
    if audit.get("schema_version") != "g3-p0k-training-audit-0.1":
        raise ValueError("P0-K training audit schema differs")
    manifest = audit.get("sample_manifest", {})
    if manifest.get("sha256") != OVERFIT32_MANIFEST_SHA256:
        raise ValueError("P0-K manifest SHA differs")
    if file_sha256(manifest["path"]) != OVERFIT32_MANIFEST_SHA256:
        raise ValueError("P0-K manifest bytes changed")
    expected_sources = {
        "p0i": P0I_CHECKPOINT_SHA256,
        "p0g": P0G_CHECKPOINT_SHA256,
    }
    for name, checksum in expected_sources.items():
        source = audit.get("checkpoints", {}).get(name, {})
        if source.get("sha256") != checksum or file_sha256(source["path"]) != checksum:
            raise ValueError(f"P0-K frozen {name.upper()} checkpoint changed")
    vision = audit.get("vision", {})
    if (
        vision.get("sha256") != DEFAULT_VISION_CHECKPOINT_SHA256
        or file_sha256(vision["path"]) != DEFAULT_VISION_CHECKPOINT_SHA256
    ):
        raise ValueError("P0-K frozen vision checkpoint changed")
    protocol = audit.get("protocol", {})
    if protocol != {
        "name": "g3-p0k-predicted-anchor-relation-alignment",
        "gt_anchor_in_pair_forward": False,
        "p0g_slot_replay_features": True,
        "train_pair_bond_only": True,
    }:
        raise ValueError("P0-K training audit protocol differs")
    implementation = audit.get("implementation", {})
    trainer_path = Path(implementation.get("trainer", ""))
    if (
        trainer_path.resolve() != Path(p0k_trainer.__file__).resolve()
        or implementation.get("trainer_sha256") != file_sha256(trainer_path)
    ):
        raise ValueError("P0-K trainer implementation SHA differs")
    forward = validate_pair_forward_source(
        inspect.getsource(p0k_trainer.p0k_training_step)
    )
    p0g_payload = torch.load(
        audit["checkpoints"]["p0g"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    source_pair, pair_config = source_pair_head(p0g_payload)
    parameter_names = [name for name, _ in source_pair.named_parameters()]
    parameter_count = sum(value.numel() for value in source_pair.parameters())
    if (
        len(parameter_names) != EXPECTED_PAIR_TENSORS
        or parameter_count != EXPECTED_PAIR_PARAMETERS
        or audit.get("parameter_count") != parameter_count
        or audit.get("trainable_parameter_names") != parameter_names
        or audit.get("initial_pair_bond_sha256") != module_state_sha256(source_pair)
    ):
        raise ValueError("P0-K pair-head initialization or optimizer scope differs")
    if artifact_dir.resolve() != DEFAULT_ARTIFACT_DIR.resolve():
        raise ValueError("P0-K audit requires the fixed artifact directory")
    return {
        "manifest_sha256": manifest["sha256"],
        "vision_sha256": vision["sha256"],
        "p0i_sha256": audit["checkpoints"]["p0i"]["sha256"],
        "p0g_sha256": audit["checkpoints"]["p0g"]["sha256"],
        "trainer_sha256": implementation["trainer_sha256"],
        "initial_pair_bond_sha256": audit["initial_pair_bond_sha256"],
        "pair_parameter_tensors": len(parameter_names),
        "pair_parameter_count": parameter_count,
        "forward_source": forward,
        "frozen_components_verified": True,
    }, {"config": pair_config, "parameter_names": parameter_names}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir)
    paths = [artifact_dir / f"step-{step:06d}.pt" for step in EXPECTED_STEPS]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("P0-K audit requires checkpoints 100/200/300/400")
    discovered = sorted(artifact_dir.glob("step-*.pt"))
    if [path.resolve() for path in discovered] != [path.resolve() for path in paths]:
        raise ValueError("P0-K artifact contains checkpoints outside 100/200/300/400")
    audit_path = artifact_dir / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError("P0-K artifact has no training audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    provenance, pair = validate_training_audit(audit, artifact_dir)
    reports = []
    for step, path in zip(EXPECTED_STEPS, paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        report = validate_checkpoint(
            payload,
            step=step,
            audit=audit,
            pair_config=pair["config"],
            expected_parameter_names=pair["parameter_names"],
        )
        report["checkpoint"] = str(path.resolve())
        report["checkpoint_sha256"] = file_sha256(path)
        reports.append(report)
    final = reports[-1]
    if not P0J_AUDIT.is_file():
        raise FileNotFoundError("P0-K audit requires the committed P0-J baseline")
    p0j_audit = json.loads(P0J_AUDIT.read_text(encoding="utf-8"))
    strict_capability = strict_final_capability(final["metrics"], p0j_audit)
    final_checks = {
        "expected_final_step": final["step"] == 400,
        "atom_mapping_1424": final["coverage"]["atom_mapping_complete"],
        "bond_coverage_1508": final["coverage"]["bond_coverage_complete"],
        "pair_universe_37380": final["coverage"]["pair_universe_complete"],
        "optimizer_pair_head_only": final["optimizer"]["pair_head_only"],
        "gt_anchor_not_in_forward": not provenance["forward_source"][
            "gt_anchor_in_pair_forward"
        ],
        "frozen_components_verified": provenance["frozen_components_verified"],
        "strict_four_branch_32_of_32": strict_capability[
            "all_four_branches_32_of_32"
        ],
        "p0j_per_sample_zero_break": strict_capability[
            "p0j_per_sample_break_impossible"
        ],
    }
    report = {
        "schema_version": "g3-p0k-checkpoint-final-audit-0.1",
        "scope": "train-only-overfit32-read-only-checkpoint-audit",
        "artifact_dir": str(artifact_dir.resolve()),
        "provenance": provenance,
        "expected_steps": list(EXPECTED_STEPS),
        "checkpoints": reports,
        "final": {
            "step": final["step"],
            "checkpoint_sha256": final["checkpoint_sha256"],
            "coverage": final["coverage"],
            "stored_gate": final["metrics"]["p0k_gate"],
            "strict_capability": strict_capability,
            "protocol_checks": final_checks,
            "protocol_status": "passed" if all(final_checks.values()) else "failed",
        },
        "implementation": {
            "audit_script": str(Path(__file__).resolve()),
            "audit_script_sha256": file_sha256(__file__),
        },
        "p0j_baseline": {
            "path": str(P0J_AUDIT.resolve()),
            "sha256": file_sha256(P0J_AUDIT),
        },
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
