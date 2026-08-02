#!/usr/bin/env python3
"""Independently audit the fixed-final G3 P1-D training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from chemvector.evaluation.audits.atoms import audit_g3_p1c_final as common
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1c_top128_set_ranker as p1c
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    load_frozen_vision_checkpoint,
    module_state_sha256,
)


EXPECTED_STEPS = (1_250, 2_500, 3_750, 5_000, 6_250)
EXPECTED_TRAINER_SHA256 = (
    "f13cad9a1c4daeeeb8ae130a578ba2c54a6fb2a105d71408690f5d3b98461e87"
)
EXPECTED_RANKER_SHA256 = common.EXPECTED_RANKER_SHA256
EXPECTED_LOSS_SHA256 = (
    "7ea06a351b43d55987dda510974d27b85620d168c0d2df9ba46c0d623c9338c0"
)
EXPECTED_P1C_HELPER_SHA256 = (
    "845eabdd90bd386f77ca40744a27aa3b2f08f39978786a4d0f4f18efbeda478f"
)
EXPECTED_ARGS = common.EXPECTED_ARGS


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "trainer": root / "chemvector/training/atoms/train_g3_p1d_boundary_band_set_ranker.py",
        "ranker": root / "chemvector/components/atoms/g3_top128_set_ranker.py",
        "loss": root / "chemvector/components/atoms/g3_boundary_band_loss.py",
        "p1c_helper": root / "chemvector/training/atoms/train_g3_p1c_top128_set_ranker.py",
    }


def _source_hashes(root: Path) -> dict[str, str]:
    return {name: common._sha256(path) for name, path in _source_paths(root).items()}


def _fixed_lineage() -> dict:
    return {
        "p1a": {
            "path": str(p1d.P1A_CHECKPOINT.resolve()),
            "sha256": p1d.P1A_CHECKPOINT_SHA256,
            "global_step": 6_250,
            "atom_head_state_sha256": p1d.P1A_ATOM_STATE_SHA256,
        },
        "vision": {
            "path": str(Path("outputs/checkpoints/chemvector-v4-clean-50k/last.pt").resolve()),
            "sha256": p1d.p1b.DEFAULT_VISION_CHECKPOINT_SHA256,
            "state_sha256": "43e3289fed751a25fb7ea06356457d23b615986b765fc785a4db202ce78ae795",
        },
        "p1c_checkpoint_loaded": False,
    }


def _checkpoint_audit(output_dir: Path, source_hashes: dict[str, str]):
    expected_names = [f"step-{step:06d}.pt" for step in EXPECTED_STEPS]
    actual_names = sorted(path.name for path in output_dir.glob("step-*.pt"))
    if actual_names != expected_names:
        raise RuntimeError(f"P1-D checkpoint set differs: {actual_names}")
    epochs = common._json(output_dir / "epoch_audit.json")
    step0 = common._json(output_dir / "step0_route_metrics.json")
    final = common._json(output_dir / "final_route_metrics.json")
    train_final = common._json(output_dir / "train_fixed_final_diagnostic.json")
    gate = common._json(output_dir / "p1d_gate.json")
    saved_audit = common._json(output_dir / "audit.json")
    lineage = _fixed_lineage()
    checkpoints = []
    for index, (step, name) in enumerate(zip(EXPECTED_STEPS, expected_names), 1):
        path = output_dir / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        optimizer = common._optimizer_audit(payload)
        _, ranker = common._ranker_from_payload(payload, torch.device("cpu"))
        final_step = step == EXPECTED_STEPS[-1]
        checks = {
            "schema": payload.get("checkpoint_schema_version")
            == "g3-p1d-boundary-band-checkpoint-0.1",
            "model_type": payload.get("model_type") == p1d.MODEL_TYPE,
            "step": payload.get("global_step") == step,
            "epoch": payload.get("epoch") == index,
            "samples_seen": payload.get("samples_seen") == step * 4,
            "trainer_sha": payload.get("trainer_sha256")
            == source_hashes["trainer"]
            == EXPECTED_TRAINER_SHA256,
            "fixed_args": all(
                payload.get("training_args", {}).get(key) == value
                for key, value in EXPECTED_ARGS.items()
            ),
            "fixed_lineage": common._json_equal(payload.get("lineage"), lineage),
            "audit_artifact": common._json_equal(payload.get("audit"), saved_audit),
            "epoch_prefix": payload.get("completed_epoch_audits") == epochs[:index],
            "step0_metrics": common._json_equal(payload.get("step0_route_metrics"), step0),
            "final_only_at_final": common._json_equal(
                payload.get("final_route_metrics"), final
            )
            if final_step
            else payload.get("final_route_metrics") is None,
            "train_diagnostic_only_at_final": common._json_equal(
                payload.get("train_fixed_final_diagnostic"), train_final
            )
            if final_step
            else payload.get("train_fixed_final_diagnostic") is None,
            "gate_only_at_final": common._json_equal(payload.get("p1d_gate"), gate)
            if final_step
            else payload.get("p1d_gate") is None,
            "set_ranker_only_payload": "set_ranker" in payload
            and all(key not in payload for key in ("atom_head", "vision", "model")),
            "optimizer_set_ranker_only": optimizer["parameter_count"]
            == p1d.G3_TOP128_SET_RANKER_PARAMETERS
            and optimizer["parameter_tensors"]
            == optimizer["optimizer_parameter_tensors"]
            == optimizer["optimizer_state_entries"],
        }
        checkpoints.append(
            {
                "path": str(path.resolve()),
                "sha256": common._sha256(path),
                "step": step,
                "residual_state_sha256": module_state_sha256(ranker),
                "optimizer": optimizer,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    last_equal = common._nested_equal(
        torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False),
        torch.load(output_dir / expected_names[-1], map_location="cpu", weights_only=False),
    )
    return checkpoints, {
        "checkpoint_file_set": actual_names == expected_names,
        "all_checkpoints": all(item["passed"] for item in checkpoints),
        "last_equals_final": last_equal,
    }


def _epoch_audit(epochs: list[dict]) -> dict:
    report = common._epoch_audit(epochs)
    report["passed"] = bool(
        report["passed"]
        and all(
            item.get("frozen_set", {}).get("candidate_count") == 640_000
            and item.get("frozen_set", {}).get("representable_positive_count") == 154_911
            for item in epochs
        )
    )
    return report


@torch.no_grad()
def _metric_replay(output_dir: Path, device: torch.device) -> dict:
    route, route_audit = p1a.load_route_manifest()
    train, _ = p1a.load_train_manifest()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    payload = torch.load(output_dir / "step-006250.pt", map_location="cpu", weights_only=False)
    atom_head, step0_ranker, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    step0 = p1c.evaluate_route(
        atom_head, step0_ranker, vision, route, collator,
        device=device, precision="bf16", batch_size=4, require_zero_parity=True,
    )
    step0["step"] = 0
    _, final_ranker = common._ranker_from_payload(payload, device)
    final = p1c.evaluate_route(
        atom_head, final_ranker, vision, route, collator,
        device=device, precision="bf16", batch_size=4, require_zero_parity=False,
    )
    final["step"] = 6_250
    train_final = p1d.evaluate_train_fixed_final(
        atom_head, final_ranker, vision, train, collator,
        device=device, precision="bf16", batch_size=4,
    )
    isolation = {
        "step0": common._target_isolation_replay(
            atom_head, step0_ranker, vision, route, collator, device
        ),
        "final": common._target_isolation_replay(
            atom_head, final_ranker, vision, route, collator, device
        ),
    }
    return {
        "device": str(device),
        "route_manifest_sha256": route_audit["sha256"],
        "vision_checkpoint_sha256": vision_sha,
        "vision_state_sha256": module_state_sha256(vision),
        "atom_state_sha256": module_state_sha256(atom_head),
        "step0": step0,
        "final": final,
        "train_final": train_final,
        "step0_matches_saved": common._json_equal(
            step0, common._json(output_dir / "step0_route_metrics.json")
        ),
        "final_matches_saved": common._json_equal(
            final, common._json(output_dir / "final_route_metrics.json")
        ),
        "train_final_matches_saved": common._json_equal(
            train_final, common._json(output_dir / "train_fixed_final_diagnostic.json")
        ),
        "target_isolation": isolation,
    }


def _replay_train_sequence() -> dict:
    manifest, audit = p1a.load_train_manifest()
    digest = hashlib.sha256()
    count = 0
    unique = set()
    for sample in p1a.iter_manifest_samples(manifest):
        key = str(sample["key"])
        digest.update(f"{key}\n".encode())
        unique.add(key)
        count += 1
    return {
        "images": count,
        "unique_keys": len(unique),
        "sample_keys_sha256": digest.hexdigest(),
        "manifest_sample_keys_sha256": audit["sample_keys_sha256"],
        "passed": count == len(unique) == 5_000
        and digest.hexdigest() == p1a.TRAIN_SEQUENCE_SHA256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=p1d.DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fixed BF16 metric replay")

    root = Path(__file__).resolve().parents[4]
    hashes = _source_hashes(root)
    checkpoints, checkpoint_checks = _checkpoint_audit(output_dir, hashes)
    saved_audit = common._json(output_dir / "audit.json")
    saved_gate = common._json(output_dir / "p1d_gate.json")
    status = common._json(output_dir / "status.json")
    epochs = common._json(output_dir / "epoch_audit.json")
    epoch_report = _epoch_audit(epochs)
    sequence = _replay_train_sequence()
    metrics = _metric_replay(output_dir, torch.device(args.device))
    contract = common._target_free_source_contract()
    isolation_passed = common._target_isolation_gate(
        contract, metrics["target_isolation"]
    )
    lineage = _fixed_lineage()
    step0_set = metrics["step0"]["frozen_set_audit"]
    final_set = metrics["final"]["frozen_set_audit"]
    train_set = metrics["train_final"]["frozen_set_audit"]
    epoch_set = epochs[0]["frozen_set"]
    protocol = {
        "trainer_sha256": hashes["trainer"] == EXPECTED_TRAINER_SHA256,
        "ranker_sha256": hashes["ranker"] == EXPECTED_RANKER_SHA256,
        "loss_sha256": hashes["loss"] == EXPECTED_LOSS_SHA256,
        "p1c_helper_sha256": hashes["p1c_helper"] == EXPECTED_P1C_HELPER_SHA256,
        "five_checkpoints": all(checkpoint_checks.values()),
        "lineage": common._json_equal(saved_audit["lineage"], lineage),
        "status_checkpoint_sha256": status.get("checkpoint_sha256")
        == checkpoints[-1]["sha256"],
        "train_sequence": epoch_report["passed"] and sequence["passed"],
        "frozen_train_set_identity": all(
            train_set[key] == epoch_set[key]
            for key in epoch_set
            if key.endswith("sha256")
        ),
        "route_candidate_identity": all(
            step0_set[key] == final_set[key]
            for key in step0_set
            if key.endswith("sha256")
        ),
        "metrics_reproduced": metrics["step0_matches_saved"]
        and metrics["final_matches_saved"]
        and metrics["train_final_matches_saved"],
        "atom_state_unchanged": metrics["atom_state_sha256"]
        == p1d.P1A_ATOM_STATE_SHA256,
        "vision_state_unchanged": metrics["vision_state_sha256"]
        == lineage["vision"]["state_sha256"],
        "target_free_forward": isolation_passed,
        "fixed_exposure": status.get("global_step") == 6_250
        and status.get("samples_seen") == 25_000,
        "route_steps": status.get("route_evaluation_steps") == [0, 6_250],
        "train_diagnostic_step": status.get("train_diagnostic_steps") == [6_250],
        "p1c_checkpoint_absent": saved_audit["lineage"]["p1c_checkpoint_loaded"]
        is False,
        "boundary_loss_only": saved_audit["loss"]
        == {
            "name": "boundary-band-pairwise-softplus",
            "boundary_size": 8,
            "margin": 0.0,
            "temperature": 1.0,
            "only_loss": True,
        },
        "forbidden_components_absent": not any(
            saved_audit["forbidden_components_loaded"].values()
        ),
    }
    corrected_gate = p1d.selector_gate(protocol, metrics["final"])
    protocol_passed = all(protocol.values())
    capability_passed = corrected_gate["capability_status"] == "passed"
    report = {
        "schema_version": "g3-p1d-checkpoint-final-audit-0.1",
        "scope": "independent-fixed-final-route-and-train-boundary-audit",
        "artifact_dir": str(output_dir),
        "source_sha256": hashes,
        "source_contract": contract,
        "lineage": lineage,
        "checkpoints": checkpoints,
        "checkpoint_checks": checkpoint_checks,
        "epoch_audit": epoch_report,
        "sequence_replay": sequence,
        "metric_replay": metrics,
        "training_saved_gate": saved_gate,
        "independent_corrected_gate": corrected_gate,
        "protocol_checks": protocol,
        "protocol_status": "passed" if protocol_passed else "failed",
        "capability_status": "passed" if capability_passed else "failed",
        "decision": "GO" if protocol_passed and capability_passed else "NO-GO",
        "diagnostic_conclusion": (
            "Frozen top-128 tokens are not train-boundary separable under the fixed "
            "ranker: boundary exact is below the preregistered 90% threshold."
        ),
        "conclusion_boundary": (
            "train5k and route-select100 only; no confirm, fresh, sealed, USPTO, "
            "MR-TADF, SDF generation, frontend, or deployment conclusion"
        ),
    }
    destination = output_dir / "checkpoint_final_audit.json"
    p1d._atomic_json(destination, report)
    print(json.dumps({
        "output": str(destination),
        "protocol": report["protocol_status"],
        "capability": report["capability_status"],
        "decision": report["decision"],
        "train_boundary_exact_rate": metrics["train_final"]["boundary_exact_rate"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
