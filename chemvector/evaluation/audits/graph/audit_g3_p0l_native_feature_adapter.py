#!/usr/bin/env python3
"""Audit P0-L adapter checkpoints, frozen lineage, and fixed-final gates."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
import textwrap

import torch

from chemvector.atom_feature_adapter import (
    ATOM_FEATURE_ADAPTER_PARAMETERS,
    ATOM_FEATURE_DIM,
    AtomFeatureAdapter,
)
from chemvector.audit_g3_p0j_frozen_composition import (
    P0G_CHECKPOINT_SHA256,
    P0I_CHECKPOINT_SHA256,
    load_frozen_p0_models,
)
from chemvector.g3_structural_data import OVERFIT32_MANIFEST_SHA256, file_sha256
from chemvector.train_g3_hierarchical_graph import (
    DEFAULT_VISION_CHECKPOINT_SHA256,
    module_state_sha256,
)
from chemvector import train_g3_p0l_native_feature_adapter as p0l_trainer


DEFAULT_ARTIFACT_DIR = Path(
    "outputs/smoke/chemvector-g3-p0l-native-feature-adapter-overfit32"
)
EXPECTED_STEPS = (100, 200, 300, 400)
EXPECTED_IMAGES = 32
EXPECTED_ATOMS = 1424
EXPECTED_FEATURE_ELEMENTS = EXPECTED_ATOMS * ATOM_FEATURE_DIM
EXPECTED_BONDS = 1508
EXPECTED_PAIRS = 37380


def adapter_from_state(state: dict[str, torch.Tensor]) -> AtomFeatureAdapter:
    if not isinstance(state, dict) or not state:
        raise ValueError("P0-L checkpoint contains no adapter state")
    adapter = AtomFeatureAdapter()
    adapter.load_state_dict(state, strict=True)
    adapter.eval()
    return adapter


def validate_adapter_optimizer(payload: dict, adapter: AtomFeatureAdapter) -> dict:
    parameter_names = [name for name, _ in adapter.named_parameters()]
    parameter_count = sum(value.numel() for value in adapter.parameters())
    if parameter_names != ["projection.weight", "projection.bias"]:
        raise ValueError("P0-L adapter architecture differs")
    if parameter_count != ATOM_FEATURE_ADAPTER_PARAMETERS:
        raise ValueError("P0-L adapter parameter count differs from 147840")
    optimizer = payload.get("optimizer", {})
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("P0-L optimizer must contain one adapter-only group")
    identifiers = groups[0].get("params", [])
    if len(identifiers) != len(parameter_names) or len(set(identifiers)) != len(
        parameter_names
    ):
        raise ValueError("P0-L optimizer does not contain exactly the adapter")
    if not set(optimizer.get("state", {})).issubset(identifiers):
        raise ValueError("P0-L optimizer state contains a frozen parameter")
    return {
        "parameter_names": parameter_names,
        "parameter_tensors": len(parameter_names),
        "parameter_count": parameter_count,
        "optimizer_parameter_tensors": len(identifiers),
        "optimizer_state_tensors": len(optimizer.get("state", {})),
        "adapter_only": True,
    }


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def validate_distillation_source(loss_source: str, step_source: str) -> dict:
    """Verify detached teacher use and absence of GT fields in the train loss."""

    loss_tree = ast.parse(textwrap.dedent(loss_source))
    functions = [node for node in loss_tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "masked_feature_distillation_loss":
        raise ValueError("P0-L source has no unique distillation loss")
    arguments = [argument.arg for argument in functions[0].args.args]
    if any(name in {"target", "targets", "labels"} for name in arguments):
        raise ValueError("P0-L distillation loss accepts a GT label argument")
    smooth_calls = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Call) and _call_name(node) == "smooth_l1_loss"
    ]
    if len(smooth_calls) != 1 or len(smooth_calls[0].args) < 2:
        raise ValueError("P0-L distillation source has no unique SmoothL1")
    teacher_detaches = [
        node
        for node in ast.walk(smooth_calls[0].args[1])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "detach"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "teacher_features"
    ]
    if len(teacher_detaches) != 1:
        raise ValueError("P0-L SmoothL1 teacher is not explicitly detached")
    forbidden_names = {
        node.id
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Name)
        and node.id in {"target", "targets", "labels", "bond_refs", "bond_types"}
    }
    if forbidden_names:
        raise ValueError("P0-L distillation loss reads a forbidden GT field")

    step_tree = ast.parse(textwrap.dedent(step_source))
    step_functions = [node for node in step_tree.body if isinstance(node, ast.FunctionDef)]
    if len(step_functions) != 1 or step_functions[0].name != "p0l_training_step":
        raise ValueError("P0-L source has no unique training step")
    step_arguments = [argument.arg for argument in step_functions[0].args.args]
    if any(name in {"target", "targets", "labels"} for name in step_arguments):
        raise ValueError("P0-L training step accepts GT labels")
    loss_calls = [
        node
        for node in ast.walk(step_functions[0])
        if isinstance(node, ast.Call)
        and _call_name(node) == "masked_feature_distillation_loss"
    ]
    if len(loss_calls) != 1:
        raise ValueError("P0-L training step does not use exactly one distillation loss")
    return {
        "smooth_l1_calls": 1,
        "teacher_detach_calls": 1,
        "gt_label_arguments": 0,
        "teacher_detached": True,
        "gt_labels_in_training_loss": False,
    }


def validate_lineage(audit: dict) -> tuple[dict, dict]:
    if audit.get("schema_version") != "g3-p0l-training-audit-0.1":
        raise ValueError("P0-L training audit schema differs")
    lineage = audit.get("lineage", {})
    expected = {
        "vision": DEFAULT_VISION_CHECKPOINT_SHA256,
        "p0i": P0I_CHECKPOINT_SHA256,
        "p0g": P0G_CHECKPOINT_SHA256,
        "p0k": p0l_trainer.P0K_CHECKPOINT_SHA256,
    }
    for name, checksum in expected.items():
        source = lineage.get(name, {})
        if source.get("sha256") != checksum or file_sha256(source["path"]) != checksum:
            raise ValueError(f"P0-L frozen {name.upper()} source changed")
    manifest = lineage.get("sample_manifest", {})
    if (
        manifest.get("sha256") != OVERFIT32_MANIFEST_SHA256
        or file_sha256(manifest["path"]) != OVERFIT32_MANIFEST_SHA256
    ):
        raise ValueError("P0-L manifest changed")
    if lineage["p0k"].get("global_step") != 400:
        raise ValueError("P0-L source P0-K is not fixed-final step 400")
    implementation = audit.get("implementation", {})
    trainer_path = Path(implementation.get("trainer", ""))
    if (
        trainer_path.resolve() != Path(p0l_trainer.__file__).resolve()
        or implementation.get("trainer_sha256") != file_sha256(trainer_path)
    ):
        raise ValueError("P0-L trainer implementation SHA differs")
    protocol = audit.get("protocol", {})
    if protocol != {
        "teacher": "p0g_same_slot_detached",
        "student": "p0i_native_atom_features",
        "loss": "valid_atom_smooth_l1_only",
        "gt_atom_or_bond_labels_in_training": False,
        "train_adapter_only": True,
    }:
        raise ValueError("P0-L training protocol differs")
    source = validate_distillation_source(
        inspect.getsource(p0l_trainer.masked_feature_distillation_loss),
        inspect.getsource(p0l_trainer.p0l_training_step),
    )
    initial_adapter = AtomFeatureAdapter()
    adapter_audit = audit.get("adapter", {})
    if (
        adapter_audit.get("parameter_count") != ATOM_FEATURE_ADAPTER_PARAMETERS
        or adapter_audit.get("trainable_parameter_names")
        != [name for name, _ in initial_adapter.named_parameters()]
        or adapter_audit.get("initial_state_sha256")
        != module_state_sha256(initial_adapter)
    ):
        raise ValueError("P0-L initial adapter architecture or state differs")
    expected_cache = {
        "native_features": [32, 128, 384],
        "teacher_features": [32, 128, 384],
        "atom_mask": [32, 128],
        "atom_slot_indices": [32, 128, 3],
    }
    if audit.get("feature_cache") != expected_cache:
        raise ValueError("P0-L feature cache shape differs")
    return {
        "manifest_sha256": manifest["sha256"],
        **{f"{name}_sha256": checksum for name, checksum in expected.items()},
        "trainer_sha256": implementation["trainer_sha256"],
        "initial_adapter_sha256": adapter_audit["initial_state_sha256"],
        "distillation_source": source,
        "lineage_verified": True,
    }, lineage


def recompute_frozen_hashes(audit: dict) -> dict[str, str | bool]:
    lineage = audit["lineage"]
    p0i, p0g, _, _, _ = load_frozen_p0_models(
        Path(lineage["p0i"]["path"]),
        Path(lineage["p0g"]["path"]),
        torch.device("cpu"),
    )
    p0l_trainer.load_p0k_pair_overlay(
        p0g, Path(lineage["p0k"]["path"]), torch.device("cpu")
    )
    current = {
        "p0i": module_state_sha256(p0i),
        "p0g_with_p0k_overlay": module_state_sha256(p0g),
        "p0k_pair_bond": module_state_sha256(p0g.pair_bond),
    }
    expected = {
        "p0i": audit["frozen_p0i_state_sha256"],
        "p0g_with_p0k_overlay": audit["frozen_p0g_state_sha256"],
        "p0k_pair_bond": audit["frozen_pair_bond_sha256"],
    }
    if current != expected:
        raise ValueError("P0-L frozen component state SHA changed")
    return {
        "p0i_state_sha256": current["p0i"],
        "p0g_with_p0k_overlay_state_sha256": current["p0g_with_p0k_overlay"],
        "p0k_pair_bond_state_sha256": current["p0k_pair_bond"],
        "initial_equals_reloaded_final_source": True,
        "checkpoint_save_guard_in_verified_trainer": True,
    }


def fixed_metrics(metrics: dict) -> dict[str, int | float | bool]:
    alignment = metrics.get("feature_alignment", {})
    adapted = metrics.get("adapted_native_feature", {})
    control = metrics.get("same_slot_p0k_control", {})
    oracle = metrics.get("gt_atom_oracle_control", {})
    baseline = float(alignment.get("step0_smooth_l1", -1.0))
    final_loss = float(alignment.get("smooth_l1", -1.0))
    ratio = final_loss / baseline if baseline > 0.0 else float("inf")
    stored_ratio = float(alignment.get("smooth_l1_ratio_to_step0", float("nan")))
    if abs(ratio - stored_ratio) > 1e-12:
        raise ValueError("P0-L stored SmoothL1 ratio is inconsistent")
    result = {
        "valid_atoms": int(alignment.get("valid_atoms", -1)),
        "feature_elements": int(alignment.get("feature_elements", -1)),
        "step0_smooth_l1": baseline,
        "smooth_l1": final_loss,
        "smooth_l1_ratio_to_step0": ratio,
        "smooth_l1_reduction_fraction": 1.0 - ratio,
        "adapted_atom_targets": int(adapted.get("atom_targets", -1)),
        "adapted_atom_matched": int(adapted.get("atom_matched", -1)),
        "adapted_target_bonds": int(adapted.get("target_bonds", -1)),
        "adapted_representable_bonds": int(adapted.get("representable_bonds", -1)),
        "adapted_pair_universe": int(adapted.get("pair_universe", -1)),
        "adapted_gt_skeleton_exact_images": int(
            adapted.get("gt_bond_count", {}).get("skeleton_exact_images", -1)
        ),
        "adapted_gt_typed_graph_exact_images": int(
            adapted.get("gt_bond_count", {}).get("typed_graph_exact_images", -1)
        ),
        "adapted_deploy_count_exact_images": int(
            adapted.get("deploy_bond_count", {}).get("count_exact_images", -1)
        ),
        "adapted_deploy_skeleton_exact_images": int(
            adapted.get("deploy_bond_count", {}).get("skeleton_exact_images", -1)
        ),
        "adapted_deploy_typed_graph_exact_images": int(
            adapted.get("deploy_bond_count", {}).get("typed_graph_exact_images", -1)
        ),
        "same_slot_gt_typed_graph_exact_images": int(
            control.get("gt_bond_count", {}).get("typed_graph_exact_images", -1)
        ),
        "same_slot_deploy_typed_graph_exact_images": int(
            control.get("deploy_bond_count", {}).get("typed_graph_exact_images", -1)
        ),
        "oracle_gt_structural_graph_exact_images": int(
            oracle.get("gt_count", {}).get("structural_graph_exact_images", -1)
        ),
        "oracle_deploy_structural_graph_exact_images": int(
            oracle.get("deploy_count", {}).get("structural_graph_exact_images", -1)
        ),
    }
    result["coverage_complete"] = (
        result["valid_atoms"] == EXPECTED_ATOMS
        and result["feature_elements"] == EXPECTED_FEATURE_ELEMENTS
        and result["adapted_atom_targets"] == result["adapted_atom_matched"] == EXPECTED_ATOMS
        and result["adapted_target_bonds"]
        == result["adapted_representable_bonds"]
        == EXPECTED_BONDS
        and result["adapted_pair_universe"] == EXPECTED_PAIRS
    )
    result["controls_32_of_32"] = all(
        result[name] == EXPECTED_IMAGES
        for name in (
            "same_slot_gt_typed_graph_exact_images",
            "same_slot_deploy_typed_graph_exact_images",
            "oracle_gt_structural_graph_exact_images",
            "oracle_deploy_structural_graph_exact_images",
        )
    )
    result["feature_error_reduced_90_percent"] = ratio <= 0.1
    result["adapted_graph_gate"] = all(
        result[name] >= 30
        for name in (
            "adapted_gt_skeleton_exact_images",
            "adapted_gt_typed_graph_exact_images",
            "adapted_deploy_count_exact_images",
            "adapted_deploy_skeleton_exact_images",
            "adapted_deploy_typed_graph_exact_images",
        )
    )
    if not result["coverage_complete"] or not result["controls_32_of_32"]:
        raise ValueError("P0-L checkpoint coverage or frozen controls differ")
    return result


def validate_checkpoint(
    payload: dict,
    *,
    step: int,
    lineage: dict,
    expected_args: dict,
    step0_alignment: dict,
) -> dict:
    if payload.get("checkpoint_schema_version") != (
        "g3-p0l-native-feature-adapter-checkpoint-0.1"
    ) or payload.get("model_type") != p0l_trainer.MODEL_TYPE:
        raise ValueError("P0-L checkpoint schema or model type differs")
    if payload.get("global_step") != step or payload.get("samples_seen") != step * 4:
        raise ValueError("P0-L checkpoint step or samples_seen differs")
    if payload.get("training_args") != expected_args:
        raise ValueError("P0-L checkpoint training arguments differ")
    if payload.get("lineage") != lineage:
        raise ValueError("P0-L checkpoint lineage differs")
    if payload.get("step0_feature_alignment") != step0_alignment:
        raise ValueError("P0-L checkpoint step-0 feature baseline differs")
    metrics = payload.get("train_only_metrics", {})
    if metrics.get("step") != step or not isinstance(metrics.get("p0l_gate"), dict):
        raise ValueError("P0-L checkpoint stored metrics differ")
    adapter = adapter_from_state(payload.get("adapter", {}))
    optimizer = validate_adapter_optimizer(payload, adapter)
    return {
        "step": step,
        "adapter_state_sha256": module_state_sha256(adapter),
        "optimizer": optimizer,
        "metrics": metrics,
        "fixed_metrics": fixed_metrics(metrics),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir)
    if artifact_dir.resolve() != DEFAULT_ARTIFACT_DIR.resolve():
        raise ValueError("P0-L audit requires the fixed artifact directory")
    paths = [artifact_dir / f"step-{step:06d}.pt" for step in EXPECTED_STEPS]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("P0-L audit requires checkpoints 100/200/300/400")
    if sorted(artifact_dir.glob("step-*.pt")) != paths:
        raise ValueError("P0-L artifact contains checkpoints outside 100/200/300/400")
    audit = json.loads((artifact_dir / "audit.json").read_text(encoding="utf-8"))
    expected_args = json.loads(
        (artifact_dir / "training_args.json").read_text(encoding="utf-8")
    )
    p0l_trainer.validate_args(argparse.Namespace(**expected_args))
    provenance, lineage = validate_lineage(audit)
    frozen = recompute_frozen_hashes(audit)
    initial = json.loads(
        (artifact_dir / "initial_metrics.json").read_text(encoding="utf-8")
    )
    step0_alignment = initial["feature_alignment"]
    reports = []
    for step, path in zip(EXPECTED_STEPS, paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        current = validate_checkpoint(
            payload,
            step=step,
            lineage=lineage,
            expected_args=expected_args,
            step0_alignment=step0_alignment,
        )
        current["checkpoint"] = str(path.resolve())
        current["checkpoint_sha256"] = file_sha256(path)
        reports.append(current)
    final = reports[-1]
    status = json.loads((artifact_dir / "status.json").read_text(encoding="utf-8"))
    stored_final_metrics = json.loads(
        (artifact_dir / "metrics.json").read_text(encoding="utf-8")
    )
    if (
        status.get("status") != "completed"
        or status.get("global_step") != 400
        or status.get("checkpoint") != "step-000400.pt"
        or stored_final_metrics != final["metrics"]
    ):
        raise ValueError("P0-L final status or metrics differ from step 400")
    last_payload = torch.load(
        artifact_dir / "last.pt", map_location="cpu", weights_only=False
    )
    last_adapter = adapter_from_state(last_payload.get("adapter", {}))
    if (
        last_payload.get("global_step") != 400
        or module_state_sha256(last_adapter) != final["adapter_state_sha256"]
    ):
        raise ValueError("P0-L last checkpoint differs from fixed-final step 400")
    protocol_checks = {
        "fixed_final_step": final["step"] == 400,
        "status_metrics_last_match_step400": True,
        "adapter_147840_optimizer_only": final["optimizer"]["adapter_only"],
        "frozen_lineage_sha": provenance["lineage_verified"],
        "frozen_state_sha": frozen["initial_equals_reloaded_final_source"],
        "teacher_detached": provenance["distillation_source"]["teacher_detached"],
        "gt_labels_not_in_loss": not provenance["distillation_source"][
            "gt_labels_in_training_loss"
        ],
        "coverage_1424_1508_37380": final["fixed_metrics"]["coverage_complete"],
        "controls_32_of_32": final["fixed_metrics"]["controls_32_of_32"],
    }
    capability = {
        "feature_interface_gate": final["fixed_metrics"][
            "feature_error_reduced_90_percent"
        ],
        "adapted_graph_gate": final["fixed_metrics"]["adapted_graph_gate"],
    }
    report = {
        "schema_version": "g3-p0l-checkpoint-final-audit-0.1",
        "scope": "train-only-overfit32-read-only-checkpoint-audit",
        "artifact_dir": str(artifact_dir.resolve()),
        "expected_steps": list(EXPECTED_STEPS),
        "provenance": provenance,
        "frozen_components": frozen,
        "checkpoints": reports,
        "final": {
            "step": final["step"],
            "checkpoint_sha256": final["checkpoint_sha256"],
            "fixed_metrics": final["fixed_metrics"],
            "stored_gate": final["metrics"]["p0l_gate"],
            "protocol_checks": protocol_checks,
            "protocol_status": "passed" if all(protocol_checks.values()) else "failed",
            "capability_checks": capability,
            "capability_status": "passed" if all(capability.values()) else "failed",
            "decision": (
                "GO"
                if all(protocol_checks.values()) and all(capability.values())
                else "NO-GO"
            ),
        },
        "implementation": {
            "audit_script": str(Path(__file__).resolve()),
            "audit_script_sha256": file_sha256(__file__),
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
