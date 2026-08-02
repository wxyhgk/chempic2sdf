#!/usr/bin/env python3
"""Independently audit fixed-final G3 P1-E three-arm artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chemvector.components.atoms.g3_strong_top128_set_ranker import (
    G3_STRONG_SET_RANKER_PARAMETERS,
    G3StrongTop128SetRanker,
)
from chemvector.evaluation.audits.atoms import audit_g3_p1c_final as common
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import (
    train_g3_p1e_aligned_stroke_feature_sufficiency as p1e,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    load_frozen_vision_checkpoint,
    module_state_sha256,
)


EXPECTED_TRAINER_SHA256 = (
    "3513a25b4c6cafed4d3b13f93194cb4ea31bfb16e99db5e61e8436fc78c8dd32"
)
EXPECTED_STEPS = (1_250, 2_500, 3_750, 5_000, 6_250)
EXPECTED_ARGS = common.EXPECTED_ARGS


def _source_hashes(root: Path) -> dict[str, str]:
    paths = {
        "trainer": root
        / "chemvector/training/atoms/train_g3_p1e_aligned_stroke_feature_sufficiency.py",
        "ranker": root / "chemvector/components/atoms/g3_strong_top128_set_ranker.py",
        "loss": root / "chemvector/components/atoms/g3_boundary_band_loss.py",
        "p1c_helper": root
        / "chemvector/training/atoms/train_g3_p1c_top128_set_ranker.py",
        "p1d_helper": root
        / "chemvector/training/atoms/train_g3_p1d_boundary_band_set_ranker.py",
    }
    return {name: common._sha256(path) for name, path in paths.items()}


def _load_arm(payload: dict, device: torch.device) -> G3StrongTop128SetRanker:
    config = p1e.G3HierarchicalGraphConfig(**payload["g3_config"])
    model = G3StrongTop128SetRanker(config).to(device)
    model.load_state_dict(payload["strong_ranker"], strict=True)
    return model.eval()


def _optimizer_audit(payload: dict) -> dict:
    model = _load_arm(payload, torch.device("cpu"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(payload["training_args"]["learning_rate"]),
        weight_decay=float(payload["training_args"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer"])
    parameters = list(model.parameters())
    groups = payload["optimizer"]["param_groups"]
    parameter_ids = [value for group in groups for value in group["params"]]
    return {
        "parameter_count": sum(value.numel() for value in parameters),
        "parameter_tensors": len(parameters),
        "optimizer_parameter_tensors": len(parameter_ids),
        "optimizer_state_entries": len(payload["optimizer"]["state"]),
        "single_parameter_group": len(groups) == 1,
        "learning_rate": groups[0]["lr"],
        "weight_decay": groups[0]["weight_decay"],
    }


def _checkpoint_audit(output_dir: Path, source_hashes: dict[str, str]):
    epoch_audit = common._json(output_dir / "epoch_audit.json")
    step0 = common._json(output_dir / "step0_route_metrics.json")
    train_final = common._json(output_dir / "train_fixed_final.json")
    route_final = common._json(output_dir / "route_fixed_final.json")
    gate = common._json(output_dir / "p1e_gate.json")
    saved_audit = common._json(output_dir / "audit.json")
    checkpoints = {}
    global_checks = {}
    expected_names = [f"step-{step:06d}.pt" for step in EXPECTED_STEPS]
    for arm in p1e.ARMS:
        arm_dir = output_dir / arm
        actual_names = sorted(path.name for path in arm_dir.glob("step-*.pt"))
        global_checks[f"{arm}_checkpoint_file_set"] = actual_names == expected_names
        rows = []
        for index, (step, name) in enumerate(zip(EXPECTED_STEPS, expected_names), 1):
            path = arm_dir / name
            payload = torch.load(path, map_location="cpu", weights_only=False)
            optimizer = _optimizer_audit(payload)
            final_step = step == EXPECTED_STEPS[-1]
            checks = {
                "schema": payload.get("checkpoint_schema_version")
                == "g3-p1e-three-arm-checkpoint-0.1",
                "model_type": payload.get("model_type") == p1e.MODEL_TYPE,
                "arm": payload.get("arm") == arm,
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
                "audit_artifact": common._json_equal(
                    payload.get("audit"), saved_audit
                ),
                "epoch_prefix": payload.get("completed_epoch_audits")
                == epoch_audit[arm][:index],
                "step0_metrics": common._json_equal(
                    payload.get("step0_route_metrics"), step0
                ),
                "final_train_only_at_final": common._json_equal(
                    payload.get("train_fixed_final"), train_final["arms"][arm]
                )
                if final_step
                else payload.get("train_fixed_final") is None,
                "final_route_only_at_final": common._json_equal(
                    payload.get("route_fixed_final"), route_final["arms"][arm]
                )
                if final_step
                else payload.get("route_fixed_final") is None,
                "gate_only_at_final": common._json_equal(
                    payload.get("p1e_gate"), gate
                )
                if final_step
                else payload.get("p1e_gate") is None,
                "arm_local_payload": "strong_ranker" in payload
                and all(
                    key not in payload
                    for key in (
                        "atom_head",
                        "vision",
                        "p1c_checkpoint",
                        "p1d_checkpoint",
                    )
                ),
                "optimizer_arm_only": optimizer["parameter_count"]
                == G3_STRONG_SET_RANKER_PARAMETERS
                and optimizer["parameter_tensors"]
                == optimizer["optimizer_parameter_tensors"]
                == optimizer["optimizer_state_entries"],
            }
            model = _load_arm(payload, torch.device("cpu"))
            rows.append(
                {
                    "path": str(path.resolve()),
                    "sha256": common._sha256(path),
                    "step": step,
                    "state_sha256": module_state_sha256(model),
                    "optimizer": optimizer,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
        checkpoints[arm] = rows
        global_checks[f"{arm}_all_checkpoints"] = all(
            value["passed"] for value in rows
        )
        global_checks[f"{arm}_last_equals_final"] = common._nested_equal(
            torch.load(arm_dir / "last.pt", map_location="cpu", weights_only=False),
            torch.load(
                arm_dir / "step-006250.pt", map_location="cpu", weights_only=False
            ),
        )
    return checkpoints, global_checks


def _epoch_audit(epoch_audit: dict) -> dict:
    fixed_hashes = {}
    checks = {}
    for arm in p1e.ARMS:
        rows = epoch_audit.get(arm, [])
        checks[f"{arm}_five_epochs"] = len(rows) == 5
        checks[f"{arm}_fixed_stream"] = all(
            row.get("epoch") == index
            and row.get("images") == 5_000
            and row.get("steps") == 1_250
            and row.get("sample_keys_sha256") == p1a.TRAIN_SEQUENCE_SHA256
            for index, row in enumerate(rows, 1)
        )
        fixed_hashes[arm] = [
            {
                key: row["input_audit"][key]
                for key in p1e.FROZEN_INPUT_SHA_KEYS
            }
            for row in rows
        ]
        checks[f"{arm}_frozen_inputs"] = bool(fixed_hashes[arm]) and all(
            value == fixed_hashes[arm][0] for value in fixed_hashes[arm]
        )
    checks["cross_arm_frozen_inputs"] = all(
        fixed_hashes[arm][0] == fixed_hashes["cap"][0] for arm in p1e.ARMS
    )
    checks["sham_epoch_seeds"] = [
        row["sham_seed"] for row in epoch_audit["sham"]
    ] == [20_260_714, 20_260_715, 20_260_716, 20_260_717, 20_260_718]
    checks["sham_zero_fixed_points"] = all(
        row["input_audit"]["permutation_fixed_points"] == 0
        for row in epoch_audit["sham"]
    )
    checks["sham_permutations_unique"] = len(
        {
            row["input_audit"]["permutation_sha256"]
            for row in epoch_audit["sham"]
        }
    ) == 5
    return {"checks": checks, "passed": all(checks.values())}


@torch.no_grad()
def _metric_replay(output_dir: Path, device: torch.device) -> dict:
    payloads = {
        arm: torch.load(
            output_dir / arm / "step-006250.pt",
            map_location="cpu",
            weights_only=False,
        )
        for arm in p1e.ARMS
    }
    arms = {arm: _load_arm(payloads[arm], device) for arm in p1e.ARMS}
    atom_head, p1a_payload, _ = p1e.load_p1a_frozen_atom_head(device)
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    train_manifest, _ = p1a.load_train_manifest()
    route_manifest, _ = p1a.load_route_manifest()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    train = p1e.evaluate_train_all_arms(
        atom_head,
        arms,
        vision,
        train_manifest,
        collator,
        device=device,
        precision="bf16",
        batch_size=4,
        sham_seeds=p1e.SHAM_EVAL_SEEDS,
    )
    route = p1e.evaluate_route_all_arms(
        atom_head,
        arms,
        vision,
        route_manifest,
        collator,
        device=device,
        precision="bf16",
        batch_size=4,
        sham_seeds=p1e.SHAM_EVAL_SEEDS,
        include_counterfactual=True,
        selected_sham_seed=train["sham_eval_bank"]["best_seed"],
        selected_counterfactual_seed=train["align_on_sham_eval_bank"]["best_seed"],
    )
    route["step"] = 6_250
    return {
        "device": str(device),
        "vision_checkpoint_sha256": vision_sha,
        "vision_state_sha256": module_state_sha256(vision),
        "atom_state_sha256": module_state_sha256(atom_head),
        "arm_state_sha256": {
            arm: module_state_sha256(model) for arm, model in arms.items()
        },
        "train": train,
        "route": route,
        "train_matches_saved": common._json_equal(
            train, common._json(output_dir / "train_fixed_final.json")
        ),
        "route_matches_saved": common._json_equal(
            route, common._json(output_dir / "route_fixed_final.json")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=p1e.DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P1-E fixed BF16 metric replay requires CUDA")

    root = Path(__file__).resolve().parents[4]
    sources = _source_hashes(root)
    checkpoints, checkpoint_checks = _checkpoint_audit(output_dir, sources)
    epochs = common._json(output_dir / "epoch_audit.json")
    epoch_report = _epoch_audit(epochs)
    saved_gate = common._json(output_dir / "p1e_gate.json")
    status = common._json(output_dir / "status.json")
    metrics = _metric_replay(output_dir, torch.device(args.device))
    replay_gates = p1e.arm_gates(metrics["train"], metrics["route"])
    replay_capability = replay_gates["causal_decision"] != "no_go_indeterminate"
    implementation = common._json(output_dir / "audit.json")["implementation"]
    protocol_checks = {
        "trainer_sha256": sources["trainer"]
        == implementation["trainer_sha256"]
        == EXPECTED_TRAINER_SHA256,
        "dependency_sha256": {
            key: sources[key] for key in p1e.EXPECTED_DEPENDENCY_SHA256
        }
        == p1e.EXPECTED_DEPENDENCY_SHA256,
        "checkpoint_chain": all(checkpoint_checks.values()),
        "epoch_protocol": epoch_report["passed"],
        "metrics_reproduced": metrics["train_matches_saved"]
        and metrics["route_matches_saved"],
        "atom_state_unchanged": metrics["atom_state_sha256"]
        == p1e.P1A_ATOM_STATE_SHA256,
        "vision_checkpoint": metrics["vision_checkpoint_sha256"]
        == p1e.p1b.DEFAULT_VISION_CHECKPOINT_SHA256,
        "status": status["global_step_per_arm"] == 6_250
        and status["samples_seen_per_arm"] == 25_000
        and status["route_evaluation_steps"] == [0, 6_250]
        and status["train_diagnostic_steps"] == [6_250],
        "gate_reproduced": replay_gates["causal_decision"]
        == saved_gate["causal_decision"]
        and replay_capability
        == (saved_gate["capability_status"] == "passed"),
    }
    protocol_passed = all(protocol_checks.values())
    report = {
        "schema_version": "g3-p1e-checkpoint-final-audit-0.1",
        "scope": "independent-three-arm-checkpoint-train-route-bank-replay",
        "artifact_dir": str(output_dir),
        "source_sha256": sources,
        "checkpoints": checkpoints,
        "checkpoint_checks": checkpoint_checks,
        "epoch_audit": epoch_report,
        "metric_replay": metrics,
        "training_saved_gate": saved_gate,
        "independent_causal_gate": replay_gates,
        "protocol_checks": protocol_checks,
        "protocol_status": "passed" if protocol_passed else "failed",
        "capability_status": "passed" if replay_capability else "failed",
        "decision": "GO" if protocol_passed and replay_capability else "NO-GO",
        "causal_decision": replay_gates["causal_decision"],
        "conclusion_boundary": (
            "fixed train5k/route100 ranker and 9x9 stroke-patch sufficiency only; "
            "no count training, Bond, OCR, SDF, confirm, fresh, sealed, USPTO, "
            "MR-TADF, frontend, or deployment conclusion"
        ),
    }
    destination = output_dir / "checkpoint_final_audit.json"
    p1e._atomic_json(destination, report)
    print(
        json.dumps(
            {
                "output": str(destination),
                "protocol": report["protocol_status"],
                "capability": report["capability_status"],
                "decision": report["decision"],
                "causal_decision": report["causal_decision"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
