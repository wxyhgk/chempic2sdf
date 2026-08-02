#!/usr/bin/env python3
"""Independently audit the fixed-final G3 P1-B training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch

from chemvector.components.graph.joint.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    G3HierarchicalGraphConfig,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1b_frozen_presence_ranker as p1b
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    load_frozen_vision_checkpoint,
    module_state_sha256,
)


EXPECTED_STEPS = (1_250, 2_500, 3_750, 5_000, 6_250)
EXPECTED_TRAINER_SHA256 = (
    "392b59e38d9f13a68b406e8770dca69f727076b58f60748c23e006a28bf5fa9e"
)
EXPECTED_RANKER_SHA256 = (
    "423b7414f8eb6406f361a916630879fade2fb2c462dffd513f566d64b96ea7ef"
)
EXPECTED_ARGS = {
    "device": "cuda",
    "precision": "bf16",
    "epochs": 5,
    "max_steps": 6_250,
    "batch_size": 4,
    "learning_rate": 3e-4,
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


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _json_equal(left, right) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _nested_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "trainer": root
        / "chemvector/training/atoms/train_g3_p1b_frozen_presence_ranker.py",
        "ranker": root / "chemvector/components/atoms/g3_presence_ranker.py",
        "g3_model": root
        / "chemvector/components/graph/joint/g3_hierarchical_graph.py",
        "core_data": root / "chemvector/core/data.py",
        "webdataset_adapter": root / "Data/sdf/chemmol_webdataset.py",
    }


def _source_hashes(root: Path) -> dict[str, str]:
    return {name: _sha256(path) for name, path in _source_paths(root).items()}


def _forbidden_reference_count(values: list[str]) -> int:
    forbidden = (
        "dev_route_" + "confirm_400",
        "fresh_" + "eval_12k",
        "sealed_" + "test",
        "public_" + "ocsr_v1",
        "MR-" + "TADF",
        "MR_" + "TADF",
    )
    return sum(token in value for value in values for token in forbidden)


def _fixed_lineage() -> dict:
    return {
        "p1a": {
            "path": str(p1b.P1A_CHECKPOINT.resolve()),
            "sha256": p1b.P1A_CHECKPOINT_SHA256,
            "global_step": 6_250,
            "atom_head_state_sha256": p1b.P1A_ATOM_STATE_SHA256,
        },
        "vision": {
            "path": str(
                Path("outputs/checkpoints/chemvector-v4-clean-50k/last.pt").resolve()
            ),
            "sha256": p1b.DEFAULT_VISION_CHECKPOINT_SHA256,
            "state_sha256": "43e3289fed751a25fb7ea06356457d23b615986b765fc785a4db202ce78ae795",
        },
    }


def _load_p1a_base(device: torch.device):
    if _sha256(p1b.P1A_CHECKPOINT) != p1b.P1A_CHECKPOINT_SHA256:
        raise RuntimeError("P1-B audit found a changed P1-A checkpoint")
    payload = torch.load(p1b.P1A_CHECKPOINT, map_location="cpu", weights_only=False)
    config = G3HierarchicalGraphConfig(**payload["g3_config"])
    base = G3AtomOnlyPredictor(config)
    base.atom_head.load_state_dict(payload["atom_head"], strict=True)
    if module_state_sha256(base.atom_head) != p1b.P1A_ATOM_STATE_SHA256:
        raise RuntimeError("P1-B audit found a changed P1-A Atom state")
    return base.to(device), payload, config


def _ranker_from_payload(payload: dict, device: torch.device):
    base, _, config = _load_p1a_base(device)
    ranker = p1b.G3FrozenAtomPresenceRanker(base.atom_head, config).to(device)
    ranker.presence_ranker.load_state_dict(payload["residual"], strict=True)
    ranker.eval()
    return ranker


def _optimizer_audit(payload: dict) -> dict:
    ranker = _ranker_from_payload(payload, torch.device("cpu"))
    optimizer = torch.optim.AdamW(
        ranker.presence_ranker.parameters(),
        lr=float(payload["training_args"]["learning_rate"]),
        weight_decay=float(payload["training_args"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer"])
    p1b.validate_optimizer(ranker, optimizer)
    groups = payload["optimizer"]["param_groups"]
    parameter_ids = [item for group in groups for item in group["params"]]
    residual_parameters = list(ranker.presence_ranker.parameters())
    return {
        "residual_only": True,
        "parameter_count": sum(value.numel() for value in residual_parameters),
        "parameter_tensors": len(residual_parameters),
        "optimizer_parameter_tensors": len(parameter_ids),
        "optimizer_state_entries": len(payload["optimizer"]["state"]),
        "single_parameter_group": len(groups) == 1,
        "learning_rate": groups[0]["lr"],
        "weight_decay": groups[0]["weight_decay"],
    }


def _checkpoint_audit(
    output_dir: Path, source_hashes: dict[str, str]
) -> tuple[list[dict], list[dict]]:
    expected_names = [f"step-{step:06d}.pt" for step in EXPECTED_STEPS]
    actual_names = sorted(path.name for path in output_dir.glob("step-*.pt"))
    checks = [{"name": "checkpoint_file_set", "passed": actual_names == expected_names}]
    if actual_names != expected_names:
        raise RuntimeError(f"P1-B checkpoint set differs: {actual_names}")

    epochs = _json(output_dir / "epoch_audit.json")
    step0_metrics = _json(output_dir / "step0_route_metrics.json")
    final_metrics = _json(output_dir / "final_route_metrics.json")
    gate = _json(output_dir / "p1b_gate.json")
    saved_audit = _json(output_dir / "audit.json")
    fixed_lineage = _fixed_lineage()
    checkpoints = []
    forbidden_payload_values = []
    for index, (step, name) in enumerate(zip(EXPECTED_STEPS, expected_names), 1):
        path = output_dir / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        forbidden_payload_values.extend(
            (
                json.dumps(payload.get("training_args", {}), sort_keys=True),
                json.dumps(payload.get("lineage", {}), sort_keys=True),
                json.dumps(payload.get("audit", {}), sort_keys=True),
            )
        )
        fixed_args = all(
            payload.get("training_args", {}).get(key) == value
            for key, value in EXPECTED_ARGS.items()
        )
        optimizer = _optimizer_audit(payload)
        ranker = _ranker_from_payload(payload, torch.device("cpu"))
        checkpoint_checks = {
            "schema": payload.get("checkpoint_schema_version")
            == "g3-p1b-presence-ranker-checkpoint-0.1",
            "model_type": payload.get("model_type") == p1b.MODEL_TYPE,
            "step": payload.get("global_step") == step,
            "epoch": payload.get("epoch") == index,
            "samples_seen": payload.get("samples_seen") == step * 4,
            "trainer_sha": payload.get("trainer_sha256")
            == source_hashes["trainer"]
            == EXPECTED_TRAINER_SHA256,
            "ranker_sha": payload.get("audit", {})
            .get("implementation", {})
            .get("ranker_sha256")
            == source_hashes["ranker"]
            == EXPECTED_RANKER_SHA256,
            "fixed_args": fixed_args,
            "fixed_lineage": _json_equal(payload.get("lineage"), fixed_lineage),
            "audit_artifact": _json_equal(payload.get("audit"), saved_audit),
            "epoch_sequence_prefix": payload.get("completed_epoch_audits")
            == epochs[:index],
            "step0_metrics": _json_equal(
                payload.get("step0_route_metrics"), step0_metrics
            ),
            "final_metrics_only_at_final": (
                _json_equal(payload.get("final_route_metrics"), final_metrics)
                if step == EXPECTED_STEPS[-1]
                else payload.get("final_route_metrics") is None
            ),
            "gate_only_at_final": (
                _json_equal(payload.get("p1b_gate"), gate)
                if step == EXPECTED_STEPS[-1]
                else payload.get("p1b_gate") is None
            ),
            "residual_only_payload": (
                "residual" in payload
                and "atom_head" not in payload
                and "vision" not in payload
                and "model" not in payload
            ),
            "residual_parameter_count": optimizer["parameter_count"]
            == p1b.EXPECTED_RESIDUAL_PARAMETERS,
            "optimizer_residual_only": (
                optimizer["residual_only"]
                and optimizer["parameter_tensors"]
                == optimizer["optimizer_parameter_tensors"]
                == optimizer["optimizer_state_entries"]
                == 4
                and optimizer["single_parameter_group"]
                and optimizer["learning_rate"] == EXPECTED_ARGS["learning_rate"]
                and optimizer["weight_decay"] == EXPECTED_ARGS["weight_decay"]
            ),
        }
        checkpoints.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "step": step,
                "epoch": index,
                "samples_seen": step * 4,
                "residual_state_sha256": module_state_sha256(
                    ranker.presence_ranker
                ),
                "optimizer": optimizer,
                "checks": checkpoint_checks,
                "passed": all(checkpoint_checks.values()),
            }
        )
    checks.extend(
        (
            {
                "name": "all_checkpoints",
                "passed": all(item["passed"] for item in checkpoints),
            },
            {
                "name": "last_equals_fixed_final",
                "passed": _nested_equal(
                    torch.load(
                        output_dir / "last.pt", map_location="cpu", weights_only=False
                    ),
                    torch.load(
                        output_dir / expected_names[-1],
                        map_location="cpu",
                        weights_only=False,
                    ),
                ),
            },
            {
                "name": "checkpoint_forbidden_data_references",
                "passed": _forbidden_reference_count(forbidden_payload_values) == 0,
            },
        )
    )
    return checkpoints, checks


def _replay_train_sequence() -> dict:
    manifest, audit = p1b.load_train_manifest()
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
        "passed": (
            count == 5_000
            and len(unique) == 5_000
            and digest.hexdigest() == p1b.TRAIN_SEQUENCE_SHA256
        ),
    }


@torch.no_grad()
def _recompute_metrics(output_dir: Path, device: torch.device) -> dict:
    route_manifest, route_audit = p1b.load_route_manifest()
    final_payload = torch.load(
        output_dir / "step-006250.pt", map_location="cpu", weights_only=False
    )
    step0_ranker, p1a_payload, config = p1b.load_p1a_ranker(device)
    vision_path = Path(p1a_payload["lineage"]["vision"]["path"])
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(vision_path, device)
    vision.requires_grad_(False).eval()
    collator = p1b.G3AtomCollator(maximum_atoms=128, image_size=768)
    step0 = p1b.evaluate_route(
        step0_ranker.eval(),
        vision,
        route_manifest,
        collator,
        device=device,
        precision="bf16",
        batch_size=4,
        require_zero_parity=True,
    )
    step0["step"] = 0

    final_ranker = p1b.G3FrozenAtomPresenceRanker(
        G3AtomOnlyPredictor(config).atom_head, config
    )
    final_ranker.atom_head.load_state_dict(p1a_payload["atom_head"], strict=True)
    final_ranker.presence_ranker.load_state_dict(
        final_payload["residual"], strict=True
    )
    final_ranker = final_ranker.to(device).eval()
    final = p1b.evaluate_route(
        final_ranker,
        vision,
        route_manifest,
        collator,
        device=device,
        precision="bf16",
        batch_size=4,
        require_zero_parity=False,
    )
    final["step"] = 6_250
    stored_step0 = _json(output_dir / "step0_route_metrics.json")
    stored_final = _json(output_dir / "final_route_metrics.json")
    return {
        "device": str(device),
        "vision_checkpoint_sha256": vision_sha,
        "vision_state_sha256": module_state_sha256(vision),
        "p1a_atom_state_sha256": module_state_sha256(step0_ranker.atom_head),
        "route_manifest_sha256": route_audit["sha256"],
        "step0": step0,
        "final": final,
        "step0_matches_saved": _json_equal(step0, stored_step0),
        "final_matches_saved": _json_equal(final, stored_final),
    }


def _epoch_checks(epochs: list[dict]) -> list[bool]:
    return [
        item.get("epoch") == index
        and item.get("images") == 5_000
        and item.get("steps") == 1_250
        and item.get("sample_keys_sha256") == p1b.TRAIN_SEQUENCE_SHA256
        for index, item in enumerate(epochs, 1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=p1b.DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fixed BF16 metric replay")

    root = Path(__file__).resolve().parents[4]
    source_paths = _source_paths(root)
    source_hashes = _source_hashes(root)
    checkpoints, checkpoint_checks = _checkpoint_audit(output_dir, source_hashes)
    saved_audit = _json(output_dir / "audit.json")
    saved_gate = _json(output_dir / "p1b_gate.json")
    status = _json(output_dir / "status.json")
    epochs = _json(output_dir / "epoch_audit.json")
    epoch_checks = _epoch_checks(epochs)
    sequence_replay = _replay_train_sequence()
    metric_replay = _recompute_metrics(output_dir, torch.device(args.device))
    route_metric_artifacts = sorted(
        path.name for path in output_dir.glob("*route_metrics*.json")
    )
    forbidden_source_references = _forbidden_reference_count(
        [
            source_paths["trainer"].read_text(encoding="utf-8"),
            source_paths["ranker"].read_text(encoding="utf-8"),
        ]
    )
    constructed_modules = saved_audit.get("constructed_modules", {})
    forbidden_modules = saved_audit.get("forbidden_components_loaded", {})
    fixed_lineage = _fixed_lineage()
    lineage_valid = _json_equal(saved_audit.get("lineage"), fixed_lineage)
    all_checkpoint_lineages = all(
        item["checks"]["fixed_lineage"] for item in checkpoints
    )
    all_optimizers_residual = all(
        item["optimizer"]["residual_only"]
        and item["optimizer"]["parameter_count"]
        == p1b.EXPECTED_RESIDUAL_PARAMETERS
        for item in checkpoints
    )

    independent_protocol_checks = {
        "train_manifest": _sha256(p1b.TRAIN_MANIFEST)
        == p1b.TRAIN_MANIFEST_SHA256,
        "train_sequence": (
            len(epoch_checks) == 5
            and all(epoch_checks)
            and sequence_replay["passed"]
        ),
        "route_sequence": (
            metric_replay["step0"]["sample_keys_sha256"]
            == metric_replay["final"]["sample_keys_sha256"]
            == p1b.EXPECTED_ROUTE_SEQUENCE_SHA256
        ),
        "identity_disjoint": saved_audit["manifests"]["identity_overlap"]
        ["all_three_identity_zero"],
        "p1a_checkpoint": _sha256(p1b.P1A_CHECKPOINT)
        == p1b.P1A_CHECKPOINT_SHA256,
        "atom_state_unchanged": (
            lineage_valid
            and all_checkpoint_lineages
            and metric_replay["p1a_atom_state_sha256"]
            == p1b.P1A_ATOM_STATE_SHA256
        ),
        "vision_state_unchanged": (
            lineage_valid
            and metric_replay["vision_checkpoint_sha256"]
            == p1b.DEFAULT_VISION_CHECKPOINT_SHA256
            and metric_replay["vision_state_sha256"]
            == fixed_lineage["vision"]["state_sha256"]
        ),
        "residual_only_optimizer": all_optimizers_residual,
        "fixed_exposure": (
            status.get("global_step") == 6_250
            and status.get("samples_seen") == 25_000
            and checkpoints[-1]["step"] == 6_250
            and checkpoints[-1]["samples_seen"] == 25_000
        ),
        "five_epochs": len(epoch_checks) == 5 and all(epoch_checks),
        "checkpoint_schedule": all(item["passed"] for item in checkpoint_checks),
        "route_steps": (
            route_metric_artifacts
            == ["final_route_metrics.json", "step0_route_metrics.json"]
            and all(
                item["checks"]["final_metrics_only_at_final"]
                for item in checkpoints
            )
        ),
        "deploy_count_vector_unchanged": (
            metric_replay["step0"]["deploy_count_vector"]
            == metric_replay["final"]["deploy_count_vector"]
        ),
        "count_logits_unchanged": (
            metric_replay["step0"]["count_logits_sha256"]
            == metric_replay["final"]["count_logits_sha256"]
        ),
        "deploy_count_metrics_unchanged": (
            metric_replay["final"]["deploy_count"]["count_exact_images"] == 13
            and metric_replay["final"]["deploy_count"]["count_mae"] == 3.83
        ),
        "forbidden_components_absent": (
            not any(forbidden_modules.values())
            and set(constructed_modules)
            == {"vision", "atom_head", "presence_ranker"}
            and forbidden_source_references == 0
        ),
        "count_free_dense_forward": tuple(
            inspect.signature(p1b.G3FrozenAtomPresenceRanker.forward_dense).parameters
        )
        == ("self", "features"),
    }
    recomputed_gate = p1b.selector_gate(
        independent_protocol_checks, metric_replay["final"]
    )
    protocol_checks = {
        "trainer_sha256": source_hashes["trainer"]
        == EXPECTED_TRAINER_SHA256,
        "ranker_sha256": source_hashes["ranker"] == EXPECTED_RANKER_SHA256,
        "five_fixed_checkpoints": all(
            item["passed"] for item in checkpoint_checks
        ),
        "last_checkpoint_status_sha256": status.get("checkpoint_sha256")
        == checkpoints[-1]["sha256"],
        "step0_metrics_reproduced": metric_replay["step0_matches_saved"],
        "final_metrics_reproduced": metric_replay["final_matches_saved"],
        "saved_gate_reproduced": _json_equal(recomputed_gate, saved_gate),
        "saved_protocol_passed": saved_gate.get("protocol_status") == "passed",
        **independent_protocol_checks,
    }
    protocol_passed = all(protocol_checks.values())
    capability_passed = recomputed_gate["capability_status"] == "passed"
    report = {
        "schema_version": "g3-p1b-checkpoint-final-audit-0.1",
        "scope": "train-distribution-read-only-fixed-final-checkpoint-audit",
        "artifact_dir": str(output_dir),
        "source_sha256": source_hashes,
        "manifests": {
            "train5k": {
                "path": str(p1b.TRAIN_MANIFEST.resolve()),
                "sha256": _sha256(p1b.TRAIN_MANIFEST),
            },
            "route_select100": {
                "path": str(p1a.ROUTE_SELECT_MANIFEST.resolve()),
                "sha256": _sha256(p1a.ROUTE_SELECT_MANIFEST),
            },
        },
        "lineage": saved_audit["lineage"],
        "checkpoints": checkpoints,
        "sequence_replay": sequence_replay,
        "epoch_audits": epochs,
        "metric_replay": metric_replay,
        "recomputed_gate": recomputed_gate,
        "protocol_checks": protocol_checks,
        "protocol_status": "passed" if protocol_passed else "failed",
        "capability_status": "passed" if capability_passed else "failed",
        "decision": "GO" if protocol_passed and capability_passed else "NO-GO",
        "conclusion_boundary": (
            "train-distribution frozen presence ranking; not count, external, "
            "unseen-image, or complete chemical-structure generalization"
        ),
    }
    destination = output_dir / "checkpoint_final_audit.json"
    p1b._atomic_json(destination, report)
    print(
        json.dumps(
            {
                "output": str(destination),
                "protocol": report["protocol_status"],
                "capability": report["capability_status"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
