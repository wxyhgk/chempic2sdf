#!/usr/bin/env python3
"""Independently audit the fixed-final G3 P1-C training artifacts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import torch

from chemvector.components.atoms.g3_top128_set_ranker import (
    G3Top128SetRanker,
    extract_frozen_top128,
    select_ranked_top128,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1c_top128_set_ranker as p1c
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _autocast_context,
    load_frozen_vision_checkpoint,
    module_state_sha256,
)


EXPECTED_STEPS = (1_250, 2_500, 3_750, 5_000, 6_250)
EXPECTED_TRAINER_SHA256 = (
    "845eabdd90bd386f77ca40744a27aa3b2f08f39978786a4d0f4f18efbeda478f"
)
EXPECTED_RANKER_SHA256 = (
    "ff8b8fc01fb3b5dffd6c09dec45b247bb1b74700fee8fe8fed990955ba52c860"
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


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "trainer": root / "chemvector/training/atoms/train_g3_p1c_top128_set_ranker.py",
        "ranker": root / "chemvector/components/atoms/g3_top128_set_ranker.py",
        "g3_model": root / "chemvector/components/graph/joint/g3_hierarchical_graph.py",
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


def _target_free_source_contract() -> dict[str, bool | list[str]]:
    signature = tuple(inspect.signature(G3Top128SetRanker.forward).parameters)
    tree = ast.parse(textwrap.dedent(inspect.getsource(G3Top128SetRanker.forward)))
    identifiers = sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and any(token in node.id.lower() for token in ("target", "count"))
        }
    )
    return {
        "signature_parameters": list(signature),
        "signature_candidates_only": signature == ("self", "candidates"),
        "no_target_or_count_identifiers": not identifiers,
        "forbidden_identifiers": identifiers,
    }


def _fixed_lineage() -> dict:
    return {
        "p1a": {
            "path": str(p1c.P1A_CHECKPOINT.resolve()),
            "sha256": p1c.P1A_CHECKPOINT_SHA256,
            "global_step": 6_250,
            "atom_head_state_sha256": p1c.P1A_ATOM_STATE_SHA256,
        },
        "vision": {
            "path": str(Path("outputs/checkpoints/chemvector-v4-clean-50k/last.pt").resolve()),
            "sha256": p1c.p1b.DEFAULT_VISION_CHECKPOINT_SHA256,
            "state_sha256": "43e3289fed751a25fb7ea06356457d23b615986b765fc785a4db202ce78ae795",
        },
    }


def _ranker_from_payload(payload: dict, device: torch.device):
    atom_head, ranker, _, _ = p1c.load_p1a_set_ranker(device)
    ranker.load_state_dict(payload["set_ranker"], strict=True)
    return atom_head.eval(), ranker.eval()


def _optimizer_audit(payload: dict) -> dict:
    atom_head, ranker = _ranker_from_payload(payload, torch.device("cpu"))
    optimizer = torch.optim.AdamW(
        ranker.parameters(),
        lr=float(payload["training_args"]["learning_rate"]),
        weight_decay=float(payload["training_args"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer"])
    p1c.validate_optimizer(atom_head, ranker, optimizer)
    groups = payload["optimizer"]["param_groups"]
    parameter_ids = [item for group in groups for item in group["params"]]
    parameters = list(ranker.parameters())
    return {
        "set_ranker_only": True,
        "parameter_count": sum(value.numel() for value in parameters),
        "parameter_tensors": len(parameters),
        "optimizer_parameter_tensors": len(parameter_ids),
        "optimizer_state_entries": len(payload["optimizer"]["state"]),
        "single_parameter_group": len(groups) == 1,
        "learning_rate": groups[0]["lr"],
        "weight_decay": groups[0]["weight_decay"],
    }


def _checkpoint_audit(output_dir: Path, source_hashes: dict[str, str]):
    expected_names = [f"step-{step:06d}.pt" for step in EXPECTED_STEPS]
    actual_names = sorted(path.name for path in output_dir.glob("step-*.pt"))
    checks = [{"name": "checkpoint_file_set", "passed": actual_names == expected_names}]
    if actual_names != expected_names:
        raise RuntimeError(f"P1-C checkpoint set differs: {actual_names}")
    epochs = _json(output_dir / "epoch_audit.json")
    step0 = _json(output_dir / "step0_route_metrics.json")
    final = _json(output_dir / "final_route_metrics.json")
    saved_gate = _json(output_dir / "p1c_gate.json")
    saved_audit = _json(output_dir / "audit.json")
    fixed_lineage = _fixed_lineage()
    checkpoints = []
    forbidden_values = []
    for index, (step, name) in enumerate(zip(EXPECTED_STEPS, expected_names), 1):
        path = output_dir / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        forbidden_values.extend(
            json.dumps(payload.get(key, {}), sort_keys=True)
            for key in ("training_args", "lineage", "audit")
        )
        optimizer = _optimizer_audit(payload)
        _, ranker = _ranker_from_payload(payload, torch.device("cpu"))
        current = {
            "schema": payload.get("checkpoint_schema_version")
            == "g3-p1c-top128-set-ranker-checkpoint-0.1",
            "model_type": payload.get("model_type") == p1c.MODEL_TYPE,
            "step": payload.get("global_step") == step,
            "epoch": payload.get("epoch") == index,
            "samples_seen": payload.get("samples_seen") == step * 4,
            "trainer_sha": payload.get("trainer_sha256")
            == source_hashes["trainer"]
            == EXPECTED_TRAINER_SHA256,
            "ranker_sha": payload.get("audit", {}).get("implementation", {}).get(
                "ranker_sha256"
            )
            == source_hashes["ranker"]
            == EXPECTED_RANKER_SHA256,
            "fixed_args": all(
                payload.get("training_args", {}).get(key) == value
                for key, value in EXPECTED_ARGS.items()
            ),
            "fixed_lineage": _json_equal(payload.get("lineage"), fixed_lineage),
            "audit_artifact": _json_equal(payload.get("audit"), saved_audit),
            "epoch_prefix": payload.get("completed_epoch_audits") == epochs[:index],
            "step0_metrics": _json_equal(payload.get("step0_route_metrics"), step0),
            "final_only_at_final": _json_equal(payload.get("final_route_metrics"), final)
            if step == 6_250
            else payload.get("final_route_metrics") is None,
            "gate_only_at_final": _json_equal(payload.get("p1c_gate"), saved_gate)
            if step == 6_250
            else payload.get("p1c_gate") is None,
            "set_ranker_only_payload": "set_ranker" in payload
            and all(key not in payload for key in ("atom_head", "vision", "model")),
            "optimizer_set_ranker_only": optimizer["set_ranker_only"]
            and optimizer["parameter_count"] == p1c.G3_TOP128_SET_RANKER_PARAMETERS
            and optimizer["parameter_tensors"]
            == optimizer["optimizer_parameter_tensors"]
            == optimizer["optimizer_state_entries"],
        }
        checkpoints.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "step": step,
                "residual_state_sha256": module_state_sha256(ranker),
                "optimizer": optimizer,
                "checks": current,
                "passed": all(current.values()),
            }
        )
    checks.extend(
        (
            {"name": "all_checkpoints", "passed": all(x["passed"] for x in checkpoints)},
            {
                "name": "last_equals_final",
                "passed": _nested_equal(
                    torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False),
                    torch.load(
                        output_dir / "step-006250.pt", map_location="cpu", weights_only=False
                    ),
                ),
            },
            {
                "name": "forbidden_data_references",
                "passed": _forbidden_reference_count(forbidden_values) == 0,
            },
        )
    )
    return checkpoints, checks


def _epoch_audit(epochs: list[dict]) -> dict:
    fixed_rows = [
        item.get("epoch") == index
        and item.get("images") == 5_000
        and item.get("steps") == 1_250
        and item.get("sample_keys_sha256") == p1c.p1a.TRAIN_SEQUENCE_SHA256
        for index, item in enumerate(epochs, 1)
    ]
    hash_rows = [
        {key: value for key, value in item.get("frozen_set", {}).items() if key.endswith("sha256")}
        for item in epochs
    ]
    return {
        "rows": len(epochs),
        "fixed_rows": fixed_rows,
        "frozen_set_hashes_equal": bool(hash_rows)
        and all(value == hash_rows[0] for value in hash_rows),
        "passed": len(epochs) == 5
        and all(fixed_rows)
        and bool(hash_rows)
        and all(value == hash_rows[0] for value in hash_rows),
    }


def _replay_train_sequence() -> dict:
    manifest, audit = p1c.p1a.load_train_manifest()
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
        and digest.hexdigest() == p1c.p1a.TRAIN_SEQUENCE_SHA256,
    }


@torch.no_grad()
def _target_isolation_replay(atom_head, ranker, vision, manifest, collator, device):
    batches = bitwise = allclose = stable = score_immutable = 0
    maximum_abs_difference = 0.0
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
        pixels = batch["pixel_values"].to(device)
        with _autocast_context(device, "bf16"):
            features = vision.forward_stroke_resolution(pixels)
            dense = atom_head(features.detach())
            candidates = extract_frozen_top128(dense)
            first = ranker(candidates)
            second = ranker(candidates)
        left = first["ranking_logits"]
        right = second["ranking_logits"]
        batches += 1
        bitwise += int(torch.equal(left, right))
        close = torch.allclose(left, right, rtol=1e-5, atol=1e-5)
        allclose += int(close)
        maximum_abs_difference = max(
            maximum_abs_difference, float((left - right).abs().max().cpu())
        )
        gt_counts = batch["atom_mask"].sum(1).to(device).long()
        selections_equal = True
        snapshot = left.detach().clone()
        for counts in (
            torch.ones_like(gt_counts),
            gt_counts,
            torch.full_like(gt_counts, 128),
        ):
            first_selected = select_ranked_top128(first, counts)
            second_selected = select_ranked_top128(second, counts)
            selections_equal &= torch.equal(
                first_selected["atom_flat_ids"], second_selected["atom_flat_ids"]
            ) and torch.equal(first_selected["atom_mask"], second_selected["atom_mask"])
        stable += int(selections_equal)
        score_immutable += int(torch.equal(snapshot, first["ranking_logits"]))
    report = {
        "batches": batches,
        "bitwise_equal_batches": bitwise,
        "allclose_batches": allclose,
        "stable_selection_batches": stable,
        "score_immutable_batches": score_immutable,
        "maximum_absolute_difference": maximum_abs_difference,
    }
    report["passed"] = (
        batches == 25 and allclose == stable == score_immutable == batches
    )
    return report


@torch.no_grad()
def _metric_replay(output_dir: Path, device: torch.device) -> dict:
    route, route_audit = p1c.p1a.load_route_manifest()
    collator = p1c.p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    final_payload = torch.load(
        output_dir / "step-006250.pt", map_location="cpu", weights_only=False
    )
    atom_head, step0_ranker, p1a_payload, _ = p1c.load_p1a_set_ranker(device)
    vision_path = Path(p1a_payload["lineage"]["vision"]["path"])
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(vision_path, device)
    vision.requires_grad_(False).eval()
    step0 = p1c.evaluate_route(
        atom_head,
        step0_ranker,
        vision,
        route,
        collator,
        device=device,
        precision="bf16",
        batch_size=4,
        require_zero_parity=True,
    )
    step0["step"] = 0
    _, final_ranker = _ranker_from_payload(final_payload, device)
    final = p1c.evaluate_route(
        atom_head,
        final_ranker,
        vision,
        route,
        collator,
        device=device,
        precision="bf16",
        batch_size=4,
        require_zero_parity=False,
    )
    final["step"] = 6_250
    isolation = {
        "step0": _target_isolation_replay(
            atom_head, step0_ranker.eval(), vision, route, collator, device
        ),
        "final": _target_isolation_replay(
            atom_head, final_ranker.eval(), vision, route, collator, device
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
        "step0_matches_saved": _json_equal(
            step0, _json(output_dir / "step0_route_metrics.json")
        ),
        "final_matches_saved": _json_equal(
            final, _json(output_dir / "final_route_metrics.json")
        ),
        "target_isolation": isolation,
    }


def _target_isolation_gate(source_contract: dict, replay: dict) -> bool:
    return bool(
        source_contract["signature_candidates_only"]
        and source_contract["no_target_or_count_identifiers"]
        and replay["step0"]["passed"]
        and replay["final"]["passed"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=p1c.DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fixed BF16 metric replay")

    root = Path(__file__).resolve().parents[4]
    sources = _source_paths(root)
    source_hashes = _source_hashes(root)
    checkpoints, checkpoint_checks = _checkpoint_audit(output_dir, source_hashes)
    saved_audit = _json(output_dir / "audit.json")
    saved_gate = _json(output_dir / "p1c_gate.json")
    status = _json(output_dir / "status.json")
    epochs = _json(output_dir / "epoch_audit.json")
    epoch_report = _epoch_audit(epochs)
    sequence_replay = _replay_train_sequence()
    metrics = _metric_replay(output_dir, torch.device(args.device))
    source_contract = _target_free_source_contract()
    isolation_passed = _target_isolation_gate(
        source_contract, metrics["target_isolation"]
    )
    fixed_lineage = _fixed_lineage()
    route_files = sorted(path.name for path in output_dir.glob("*route_metrics*.json"))
    frozen_set_equal = all(
        metrics["step0"]["frozen_set_audit"][key]
        == metrics["final"]["frozen_set_audit"][key]
        for key in metrics["step0"]["frozen_set_audit"]
        if key.endswith("sha256")
    )
    independent_checks = {
        "train_manifest": _sha256(p1c.p1a.TRAIN_MANIFEST)
        == p1c.p1a.TRAIN_MANIFEST_SHA256,
        "train_sequence": epoch_report["passed"] and sequence_replay["passed"],
        "frozen_train_set_identity": epoch_report["frozen_set_hashes_equal"],
        "route_sequence": metrics["step0"]["sample_keys_sha256"]
        == metrics["final"]["sample_keys_sha256"]
        == p1c.EXPECTED_ROUTE_SEQUENCE_SHA256,
        "route_candidate_identity": frozen_set_equal,
        "identity_disjoint": saved_audit["manifests"]["identity_overlap"]
        ["all_three_identity_zero"],
        "p1a_checkpoint": _sha256(p1c.P1A_CHECKPOINT)
        == p1c.P1A_CHECKPOINT_SHA256,
        "atom_state_unchanged": metrics["atom_state_sha256"]
        == p1c.P1A_ATOM_STATE_SHA256,
        "vision_state_unchanged": metrics["vision_state_sha256"]
        == fixed_lineage["vision"]["state_sha256"],
        "set_ranker_only_optimizer": all(
            item["optimizer"]["set_ranker_only"] for item in checkpoints
        ),
        "fixed_exposure": status.get("global_step") == 6_250
        and status.get("samples_seen") == 25_000,
        "checkpoint_schedule": all(item["passed"] for item in checkpoint_checks),
        "route_steps": status.get("route_evaluation_steps") == [0, 6_250]
        and route_files == ["final_route_metrics.json", "step0_route_metrics.json"],
        "count_vector_unchanged": metrics["step0"]["deploy_count_vector"]
        == metrics["final"]["deploy_count_vector"],
        "count_logits_unchanged": metrics["step0"]["count_logits_sha256"]
        == metrics["final"]["count_logits_sha256"],
        "deploy_count_metrics_unchanged": metrics["final"]["deploy_count"]
        ["count_exact_images"]
        == 13
        and metrics["final"]["deploy_count"]["count_mae"] == 3.83,
        "count_invariant_scores": metrics["step0"]["count_invariant_ranked_scores"]
        and metrics["final"]["count_invariant_ranked_scores"],
        "target_free_forward": isolation_passed,
        "forbidden_components_absent": not any(
            saved_audit["forbidden_components_loaded"].values()
        )
        and _forbidden_reference_count(
            [
                sources["trainer"].read_text(encoding="utf-8"),
                sources["ranker"].read_text(encoding="utf-8"),
            ]
        )
        == 0,
    }
    corrected_gate = p1c.selector_gate(independent_checks, metrics["final"])
    protocol_checks = {
        "trainer_sha256": source_hashes["trainer"] == EXPECTED_TRAINER_SHA256,
        "ranker_sha256": source_hashes["ranker"] == EXPECTED_RANKER_SHA256,
        "five_checkpoints": all(item["passed"] for item in checkpoint_checks),
        "lineage": _json_equal(saved_audit["lineage"], fixed_lineage)
        and all(item["checks"]["fixed_lineage"] for item in checkpoints),
        "status_checkpoint_sha256": status.get("checkpoint_sha256")
        == checkpoints[-1]["sha256"],
        "step0_metrics_reproduced": metrics["step0_matches_saved"],
        "final_metrics_reproduced": metrics["final_matches_saved"],
        **independent_checks,
    }
    protocol_passed = all(protocol_checks.values())
    capability_passed = corrected_gate["capability_status"] == "passed"
    report = {
        "schema_version": "g3-p1c-checkpoint-final-audit-0.1",
        "scope": "train-distribution-read-only-fixed-final-checkpoint-audit",
        "artifact_dir": str(output_dir),
        "source_sha256": source_hashes,
        "source_contract": source_contract,
        "lineage": saved_audit["lineage"],
        "checkpoints": checkpoints,
        "epoch_audit": epoch_report,
        "sequence_replay": sequence_replay,
        "metric_replay": metrics,
        "training_saved_gate": saved_gate,
        "independent_corrected_gate": corrected_gate,
        "bitwise_interpretation": {
            "training_final_bitwise": metrics["final"]["target_free_forward_bytes"],
            "hard_gate_uses_bitwise": False,
            "reason": (
                "Target isolation is established by target/count-free source and API, "
                "allclose repeated forward, immutable scores, and stable selections."
            ),
        },
        "protocol_checks": protocol_checks,
        "protocol_status": "passed" if protocol_passed else "failed",
        "capability_status": "passed" if capability_passed else "failed",
        "decision": "GO" if protocol_passed and capability_passed else "NO-GO",
        "conclusion_boundary": (
            "train-distribution fixed-top128 set ranking; not count, external, "
            "unseen-image, or complete chemical-structure generalization"
        ),
    }
    destination = output_dir / "checkpoint_final_audit.json"
    p1c._atomic_json(destination, report)
    print(
        json.dumps(
            {
                "output": str(destination),
                "protocol": report["protocol_status"],
                "capability": report["capability_status"],
                "decision": report["decision"],
                "training_bitwise": report["bitwise_interpretation"][
                    "training_final_bitwise"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
