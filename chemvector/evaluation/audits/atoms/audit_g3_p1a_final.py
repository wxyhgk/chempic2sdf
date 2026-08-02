#!/usr/bin/env python3
"""Independently audit the fixed-final G3 P1-A training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from chemvector.components.graph.joint.g3_hierarchical_graph import G3AtomOnlyPredictor
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    load_frozen_vision_checkpoint,
    module_state_sha256,
)


EXPECTED_STEPS = (1_250, 2_500, 3_750, 5_000, 6_250)
EXPECTED_TRAINER_SHA256 = (
    "9fe59edef699220269614768f9654a5bdeefa5ba2644edf99ba55a944686f15b"
)
EXPECTED_ARGS = {
    "device": "cuda",
    "precision": "bf16",
    "epochs": 5,
    "max_steps": 6_250,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "save_every": 1_250,
    "num_workers": 0,
    "seed": 20260713,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_equal(left, right) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _nested_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _source_hashes(root: Path) -> dict[str, str]:
    paths = {
        "trainer": root / "chemvector/training/atoms/train_g3_p1a_canonical_atom_selector.py",
        "g3_model": root / "chemvector/components/graph/joint/g3_hierarchical_graph.py",
        "g3_data": root / "chemvector/training/graph/scenes/g3_structural_data.py",
        "core_data": root / "chemvector/core/data.py",
        "webdataset_adapter": root / "Data/sdf/chemmol_webdataset.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _forbidden_reference_count(values: list[str]) -> int:
    forbidden = (
        "dev_route_" + "confirm_400",
        "fresh_" + "eval_12k",
        "sealed_" + "test",
        "public_" + "ocsr_v1",
        "MR-" + "TADF",
    )
    return sum(token in value for value in values for token in forbidden)


def _optimizer_audit(payload: dict) -> dict:
    config = p1a.config_from_payload(payload)
    model = G3AtomOnlyPredictor(config)
    model.atom_head.load_state_dict(payload["atom_head"], strict=True)
    optimizer = torch.optim.AdamW(
        model.atom_head.parameters(),
        lr=float(payload["training_args"]["learning_rate"]),
        weight_decay=float(payload["training_args"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer"])
    p1a.validate_atom_optimizer(model, optimizer)
    groups = payload["optimizer"]["param_groups"]
    parameter_ids = [item for group in groups for item in group["params"]]
    return {
        "atom_head_only": True,
        "parameter_tensors": len(list(model.atom_head.parameters())),
        "optimizer_parameter_tensors": len(parameter_ids),
        "optimizer_state_entries": len(payload["optimizer"]["state"]),
        "single_parameter_group": len(groups) == 1,
        "learning_rate": groups[0]["lr"],
        "weight_decay": groups[0]["weight_decay"],
    }


def _checkpoint_audit(output_dir: Path, trainer_sha: str) -> tuple[list[dict], list[dict]]:
    expected_names = [f"step-{step:06d}.pt" for step in EXPECTED_STEPS]
    actual_names = sorted(path.name for path in output_dir.glob("step-*.pt"))
    checks = [{"name": "checkpoint_file_set", "passed": actual_names == expected_names}]
    if actual_names != expected_names:
        raise RuntimeError(f"P1-A checkpoint set differs: {actual_names}")

    epoch_audit = _json(output_dir / "epoch_audit.json")["epochs"]
    step0_metrics = _json(output_dir / "step0_route_metrics.json")
    final_metrics = _json(output_dir / "final_route_metrics.json")
    gate = _json(output_dir / "p1a_gate.json")
    checkpoints = []
    payload_strings = []
    for index, (step, name) in enumerate(zip(EXPECTED_STEPS, expected_names), 1):
        path = output_dir / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload_strings.append(json.dumps(payload.get("training_args", {}), sort_keys=True))
        payload_strings.append(json.dumps(payload.get("lineage", {}), sort_keys=True))
        expected_epoch_prefix = epoch_audit[:index]
        args_match = all(payload["training_args"].get(key) == value for key, value in EXPECTED_ARGS.items())
        checkpoint_checks = {
            "schema": payload.get("checkpoint_schema_version") == "g3-p1a-atom-selector-checkpoint-0.1",
            "model_type": payload.get("model_type") == p1a.MODEL_TYPE,
            "step": payload.get("global_step") == step,
            "epoch": payload.get("epoch") == index,
            "samples_seen": payload.get("samples_seen") == step * 4,
            "trainer_sha": payload.get("trainer_sha256") == trainer_sha,
            "fixed_args": args_match,
            "epoch_sequence_prefix": payload.get("completed_epoch_audits") == expected_epoch_prefix,
            "step0_metrics": _json_equal(payload.get("step0_route_metrics"), step0_metrics),
            "final_metrics_only_at_final": (
                _json_equal(payload.get("final_route_metrics"), final_metrics)
                if step == EXPECTED_STEPS[-1]
                else payload.get("final_route_metrics") is None
            ),
            "gate_only_at_final": (
                _json_equal(payload.get("p1a_gate"), gate)
                if step == EXPECTED_STEPS[-1]
                else payload.get("p1a_gate") is None
            ),
            "atom_head_only_payload": "model" not in payload and "vision" not in payload,
        }
        optimizer = _optimizer_audit(payload)
        checkpoint_checks["optimizer_atom_head_only"] = optimizer["atom_head_only"]
        checkpoints.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "step": step,
                "epoch": index,
                "samples_seen": step * 4,
                "atom_head_state_sha256": p1a.module_state_sha256(
                    _model_from_payload(payload).atom_head
                ),
                "optimizer": optimizer,
                "checks": checkpoint_checks,
                "passed": all(checkpoint_checks.values()),
            }
        )
    checks.extend(
        [
            {"name": "all_checkpoints", "passed": all(item["passed"] for item in checkpoints)},
            {
                "name": "last_equals_fixed_final",
                "passed": _nested_equal(
                    torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False),
                    torch.load(output_dir / expected_names[-1], map_location="cpu", weights_only=False),
                ),
            },
            {
                "name": "checkpoint_forbidden_data_references",
                "passed": _forbidden_reference_count(payload_strings) == 0,
            },
        ]
    )
    return checkpoints, checks


def _model_from_payload(payload: dict) -> G3AtomOnlyPredictor:
    model = G3AtomOnlyPredictor(p1a.config_from_payload(payload))
    model.atom_head.load_state_dict(payload["atom_head"], strict=True)
    return model


def _replay_train_sequence() -> dict:
    manifest, audit = p1a.load_train_manifest()
    digest = hashlib.sha256()
    count = 0
    unique = set()
    for sample in p1a.iter_manifest_samples(manifest):
        key = str(sample["key"])
        digest.update(f"{key}\n".encode("utf-8"))
        unique.add(key)
        count += 1
    return {
        "images": count,
        "unique_keys": len(unique),
        "sample_keys_sha256": digest.hexdigest(),
        "manifest_sample_keys_sha256": audit["sample_keys_sha256"],
        "passed": count == 5_000 and len(unique) == 5_000 and digest.hexdigest() == p1a.TRAIN_SEQUENCE_SHA256,
    }


def _strip_derived_metrics(metrics: dict) -> dict:
    result = dict(metrics)
    result.pop("step", None)
    result.pop("delta_from_step0", None)
    return result


@torch.no_grad()
def _evaluate_gt_count_topk(
    model: G3AtomOnlyPredictor,
    vision,
    manifest,
    collator,
    *,
    device: torch.device,
) -> dict:
    totals = p1a._empty_atom_totals()
    digest = hashlib.sha256()
    was_training = model.training
    model.eval()
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
        for key in batch["keys"]:
            digest.update(f"{key}\n".encode("utf-8"))
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        target_counts = batch["atom_mask"].sum(dim=1).to(device).long()
        with p1a._autocast_context(device, "bf16"):
            features = vision.forward_stroke_resolution(pixels)
            output = model(features.detach(), atom_counts=target_counts)
        for index in range(len(batch["keys"])):
            predicted = torch.where(output["atom_mask"][index])[0]
            target = torch.where(batch["atom_mask"][index])[0]
            current = p1a.atom_sample_statistics(
                output["atom_anchors"][index, predicted].detach().float().cpu(),
                batch["atom_anchors"][index, target].float(),
            )
            p1a._merge_atom_totals(totals, current)
        del pixels, features, output
    if was_training:
        model.train()
    return {
        "scope": "route-select100-oracle-gt-count-topk-diagnostic-only",
        **p1a._finalize_atom_totals(totals),
        "sample_keys_sha256": digest.hexdigest(),
        "changes_preregistered_gate": False,
    }


def _recompute_metrics(output_dir: Path, device: torch.device) -> dict:
    route_manifest, route_audit = p1a.load_route_manifest()
    stored_audit = _json(output_dir / "audit.json")
    occupancy = stored_audit["preflight"]["route_select100"]
    final_payload = torch.load(output_dir / "step-006250.pt", map_location="cpu", weights_only=False)
    warmstart, warmstart_payload, _ = p1a.load_warmstart_p0i(Path(p1a.P0I_CHECKPOINT), device)
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        warmstart_payload["source_vision_checkpoint"], device
    )
    vision.requires_grad_(False).eval()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    step0 = p1a.evaluate_route_streaming(
        warmstart.eval(), vision, route_manifest, collator, occupancy,
        device=device, precision="bf16", batch_size=4,
    )
    final = _model_from_payload(final_payload).to(device).eval()
    final_metrics = p1a.evaluate_route_streaming(
        final, vision, route_manifest, collator, occupancy,
        device=device, precision="bf16", batch_size=4,
    )
    step0_gt_count = _evaluate_gt_count_topk(
        warmstart, vision, route_manifest, collator, device=device
    )
    final_gt_count = _evaluate_gt_count_topk(
        final, vision, route_manifest, collator, device=device
    )
    stored_step0 = _strip_derived_metrics(_json(output_dir / "step0_route_metrics.json"))
    stored_final = _strip_derived_metrics(_json(output_dir / "final_route_metrics.json"))
    recomputed_gate = p1a.atom_capability_gate(final_metrics)
    return {
        "device": str(device),
        "vision_checkpoint_sha256": vision_sha,
        "route_manifest_sha256": route_audit["sha256"],
        "step0": step0,
        "final": final_metrics,
        "gt_count_topk_diagnostic": {
            "step0": step0_gt_count,
            "final": final_gt_count,
        },
        "step0_matches_saved": _json_equal(step0, stored_step0),
        "final_matches_saved": _json_equal(final_metrics, stored_final),
        "capability_gate": recomputed_gate,
        "capability_gate_matches_saved": _json_equal(
            recomputed_gate, _json(output_dir / "p1a_gate.json")["capability"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=p1a.DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fixed BF16 metric replay")

    root = Path(__file__).resolve().parents[4]
    source_hashes = _source_hashes(root)
    trainer_sha = source_hashes["trainer"]
    checkpoints, checkpoint_checks = _checkpoint_audit(output_dir, trainer_sha)
    audit = _json(output_dir / "audit.json")
    status = _json(output_dir / "status.json")
    epochs = _json(output_dir / "epoch_audit.json")["epochs"]
    saved_gate = _json(output_dir / "p1a_gate.json")
    sequence_replay = _replay_train_sequence()
    metric_replay = _recompute_metrics(output_dir, torch.device(args.device))
    source_contract = p1a.static_training_contract()
    route_metric_artifacts = sorted(
        path.name for path in output_dir.glob("*route_metrics*.json")
    )

    epoch_checks = [
        item.get("epoch") == index
        and item.get("images") == 5_000
        and item.get("steps") == 1_250
        and item.get("sample_keys_sha256") == p1a.TRAIN_SEQUENCE_SHA256
        for index, item in enumerate(epochs, 1)
    ]
    protocol_checks = {
        "five_fixed_checkpoints": all(item["passed"] for item in checkpoint_checks),
        "trainer_sha256": trainer_sha == EXPECTED_TRAINER_SHA256,
        "train_manifest_sha256": _sha256(p1a.TRAIN_MANIFEST) == p1a.TRAIN_MANIFEST_SHA256,
        "route_manifest_sha256": _sha256(p1a.ROUTE_SELECT_MANIFEST) == p1a.ROUTE_SELECT_MANIFEST_SHA256,
        "warmstart_sha256": _sha256(Path(p1a.P0I_CHECKPOINT)) == p1a.P0I_CHECKPOINT_SHA256,
        "vision_sha256": metric_replay["vision_checkpoint_sha256"] == p1a.DEFAULT_VISION_CHECKPOINT_SHA256,
        "five_epoch_sequences": len(epoch_checks) == 5 and all(epoch_checks) and sequence_replay["passed"],
        "fixed_final_step_and_exposure": status.get("global_step") == 6_250 and status.get("samples_seen") == 25_000,
        "route_evaluated_only_step0_final": status.get("route_evaluation_steps") == [0, 6_250],
        "route_metric_artifacts_only_step0_final": route_metric_artifacts
        == ["final_route_metrics.json", "step0_route_metrics.json"],
        "frozen_vision": (
            status.get("initial_vision_state_sha256") == status.get("final_vision_state_sha256")
            and audit["lineage"]["vision"]["state_sha256"] == status.get("initial_vision_state_sha256")
        ),
        "atom_only_optimizer": all(item["optimizer"]["atom_head_only"] for item in checkpoints),
        "forbidden_components_absent": all(
            value is False for value in audit["forbidden_components_loaded"].values()
        ),
        "trainer_static_atom_only_contract": all(
            (
                source_contract["forbidden_data_path_references"] == 0,
                source_contract["forbidden_model_identifiers"] == 0,
                source_contract["vision_calls_per_training_step"] == 1,
                source_contract["canonical_loss_calls_per_training_step"] == 1,
                source_contract["teacher_forced_count_references"] == 0,
                source_contract["whole_dataset_feature_cache_references"] == 0,
            )
        ),
        "saved_protocol_gate": saved_gate.get("protocol_status") == "passed",
        "step0_metrics_reproduced": metric_replay["step0_matches_saved"],
        "final_metrics_reproduced": metric_replay["final_matches_saved"],
        "capability_gate_reproduced": metric_replay["capability_gate_matches_saved"],
    }
    protocol_passed = all(protocol_checks.values())
    capability_passed = metric_replay["capability_gate"]["status"] == "passed"
    report = {
        "schema_version": "g3-p1a-checkpoint-final-audit-0.1",
        "scope": "train-distribution-read-only-fixed-final-checkpoint-audit",
        "artifact_dir": str(output_dir),
        "source_sha256": source_hashes,
        "source_contract": source_contract,
        "manifests": {
            "train5k": {"path": str(p1a.TRAIN_MANIFEST.resolve()), "sha256": _sha256(p1a.TRAIN_MANIFEST)},
            "route_select100": {"path": str(p1a.ROUTE_SELECT_MANIFEST.resolve()), "sha256": _sha256(p1a.ROUTE_SELECT_MANIFEST)},
        },
        "lineage": audit["lineage"],
        "checkpoints": checkpoints,
        "sequence_replay": sequence_replay,
        "epoch_audits": epochs,
        "metric_replay": metric_replay,
        "protocol_checks": protocol_checks,
        "protocol_status": "passed" if protocol_passed else "failed",
        "capability_status": "passed" if capability_passed else "failed",
        "decision": "GO" if protocol_passed and capability_passed else "NO-GO",
        "conclusion_boundary": "train-distribution Atom-head transfer; not external or unseen-image generalization",
    }
    destination = output_dir / "checkpoint_final_audit.json"
    p1a._atomic_json(destination, report)
    print(json.dumps({"output": str(destination), "protocol": report["protocol_status"], "capability": report["capability_status"], "decision": report["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
