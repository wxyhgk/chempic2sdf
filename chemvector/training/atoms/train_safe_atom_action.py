#!/usr/bin/env python3
"""Train the v0.10.13 directional safe AtomObject action head."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.audit_safe_atom_suppression_oracle import (
    DATA_COMMITMENT,
    collect_samples,
    verify_data_gate,
)
from chemvector.direct_scene_inference import load_direct_model
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.safe_atom_action_model import (
    SafeAtomActionConfig,
    SafeAtomActionModel,
    delete_action_features,
    safe_atom_action_loss,
)
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_bond_set_multidomain import (
    mixed_batches,
    verify_data_gate as verify_training_data_gate,
)


DEFAULT_DIRECT = "outputs/checkpoints/chemvector-v0103-direct-atom-pair-style10k/selected.pt"
DEFAULT_CACHE = "outputs/caches/chemvector-v01013-route-select-actions/base-actions.pt"
DEFAULT_ORACLE = "outputs/audits/chemvector-v01013-safe-single-delete-oracle/report.json"
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v01013-safe-atom-action-50k"
ACTION_CACHE_SHA256 = "d480cd9fa26cd52e1658b3a7460653e06abc9cfb23d2f58c6e92eab618663923"
ACTION_CACHE_CODE_REVISION = "a769502fda1d34ec9bb6967cc05efc12b8df93b5"
THRESHOLDS = (0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-checkpoint", default=DEFAULT_DIRECT)
    parser.add_argument("--action-cache", default=DEFAULT_CACHE)
    parser.add_argument("--oracle-report", default=DEFAULT_ORACLE)
    parser.add_argument("--style-train-samples", type=int, default=33_334)
    parser.add_argument("--chemvision-replay-samples", type=int, default=8_333)
    parser.add_argument("--font-v2-replay-samples", type=int, default=8_333)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validate-every-total-samples", type=int, default=10_000)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_action_cache(path: str, oracle_report: str, validation_limit: int | None):
    observed_sha = _sha256_file(path)
    if observed_sha != ACTION_CACHE_SHA256:
        raise ValueError(
            f"Action cache SHA differs: {observed_sha} != {ACTION_CACHE_SHA256}"
        )
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("schema_version") != "safe-atom-action-cache-0.1":
        raise ValueError("Action cache schema differs")
    if cache.get("data_commitment_sha256") != DATA_COMMITMENT:
        raise ValueError("Action cache data commitment differs")
    if cache.get("code_git_revision") != ACTION_CACHE_CODE_REVISION:
        raise ValueError("Action cache generator revision differs")
    if cache.get("input_samples") != 8_000 or len(cache.get("samples", [])) != 7_976:
        raise ValueError("Action cache coverage differs")
    report = json.loads(Path(oracle_report).read_text(encoding="utf-8"))
    if not report.get("oracle_gate", {}).get("passed"):
        raise ValueError("Single-delete oracle did not pass")
    if report.get("cache", {}).get("sha256") != observed_sha:
        raise ValueError("Oracle report references another action cache")
    samples = cache["samples"]
    forced_failures = len(cache["coordinate_collisions"])
    if validation_limit is not None:
        if not 1 <= validation_limit <= len(samples):
            raise ValueError("Validation limit lies outside cached samples")
        samples = samples[:validation_limit]
        forced_failures = 0
    commitment = {
        "schema_version": "safe-atom-action-cache-commitment-0.1",
        "data_commitment_sha256": DATA_COMMITMENT,
        "cache_path": path,
        "cache_sha256": observed_sha,
        "generator_code_revision": ACTION_CACHE_CODE_REVISION,
        "direct_checkpoint_sha256": cache["direct_checkpoint_sha256"],
        "input_samples": cache["input_samples"],
        "cached_samples": len(cache["samples"]),
        "coordinate_collision_failures": len(cache["coordinate_collisions"]),
        "maximum_action_distance_px": cache["maximum_action_distance_px"],
        "primary_match_radius_px": cache["primary_match_radius_px"],
        "secondary_match_radius_px": cache["secondary_match_radius_px"],
    }
    commitment["commitment_sha256"] = hashlib.sha256(
        _canonical_json_bytes(commitment)
    ).hexdigest()
    return cache, samples, forced_failures, commitment


def utility_class(baseline: tuple[int, ...], action: dict) -> int:
    if not action["safe"]:
        return 0
    utility = tuple(action["utility"])
    if utility[0] > baseline[0]:
        return 3
    if utility[1] > baseline[1]:
        return 2
    if utility > baseline:
        return 1
    return 0


def action_training_batch(samples: list[dict], presence_threshold: float):
    features = []
    safe_targets = []
    utility_targets = []
    winner_targets = []
    sample_indices = []
    removed_nodes = []
    for sample_index, sample in enumerate(samples):
        values, removed = delete_action_features(
            sample, presence_threshold=presence_threshold
        )
        if not len(removed):
            continue
        label_by_removed = {
            int(action["removed"]): action
            for action in sample["action_labels"]["actions"]
        }
        if set(removed.tolist()) != set(label_by_removed):
            raise ValueError(f"Action feature/label nodes differ in {sample['key']}")
        winners = set(sample["action_labels"]["winner_removed_nodes"])
        baseline = tuple(sample["action_labels"]["baseline_utility"])
        features.append(values)
        safe_targets.extend(label_by_removed[int(node)]["safe"] for node in removed)
        utility_targets.extend(
            utility_class(baseline, label_by_removed[int(node)]) for node in removed
        )
        winner_targets.extend(int(node) in winners for node in removed)
        sample_indices.extend([sample_index] * len(removed))
        removed_nodes.extend(int(node) for node in removed)
    if not features:
        return None
    return {
        "features": torch.cat(features),
        "safe_targets": torch.tensor(safe_targets, dtype=torch.bool),
        "utility_targets": torch.tensor(utility_targets, dtype=torch.long),
        "winner_targets": torch.tensor(winner_targets, dtype=torch.bool),
        "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "removed_nodes": torch.tensor(removed_nodes, dtype=torch.long),
    }


@torch.no_grad()
def model_action_outputs(model, samples, presence_threshold, device):
    prepared = action_training_batch(samples, presence_threshold)
    if prepared is None:
        return None
    model.eval()
    outputs = defaultdict(list)
    for start in range(0, len(prepared["features"]), 2048):
        values = prepared["features"][start : start + 2048].to(device)
        current = model(values)
        for name, tensor in current.items():
            outputs[name].append(tensor.float().cpu())
    return {
        **prepared,
        **{name: torch.cat(values) for name, values in outputs.items()},
    }


def _summarize_threshold(
    samples,
    output,
    *,
    threshold,
    presence_threshold,
    forced_failures,
):
    totals = defaultdict(int)
    totals["samples"] = forced_failures
    totals["coordinate_collision_failures"] = forced_failures
    for name in (
        "executed_actions",
        "safe_executed_actions",
        "harmful_actions",
        "winner_actions",
        "atom_rescue",
        "atom_break",
        "skeleton_rescue",
        "skeleton_break",
    ):
        totals[name] = 0
    safety = output["safety_logits"].sigmoid()
    predicted_utility = output["utility_logits"].argmax(-1)
    action_score = output["rank_logits"] + torch.nn.functional.logsigmoid(
        output["safety_logits"]
    )
    cases = []
    for sample_index, sample in enumerate(samples):
        baseline = tuple(sample["action_labels"]["baseline_utility"])
        label_by_removed = {
            int(action["removed"]): action
            for action in sample["action_labels"]["actions"]
        }
        rows = torch.where(output["sample_indices"] == sample_index)[0]
        qualified = rows[
            (safety[rows] >= threshold)
            & (predicted_utility[rows] > 0)
            & (action_score[rows] > 0.0)
        ]
        selected_row = None
        if len(qualified):
            selected_row = int(qualified[action_score[qualified].argmax()])
        removed = None
        candidate = baseline
        if selected_row is not None:
            removed = int(output["removed_nodes"][selected_row])
            candidate = tuple(label_by_removed[removed]["utility"])
        totals["samples"] += 1
        totals["oracle_available"] += int(
            bool(sample["action_labels"]["winner_removed_nodes"])
        )
        totals["executed_actions"] += int(selected_row is not None)
        if selected_row is not None:
            safe = bool(output["safe_targets"][selected_row])
            winner = bool(output["winner_targets"][selected_row])
            totals["safe_executed_actions"] += int(safe)
            totals["harmful_actions"] += int(not safe)
            totals["winner_actions"] += int(winner)
        totals["baseline_atom_exact"] += baseline[1]
        totals["baseline_skeleton_exact"] += baseline[0]
        totals["baseline_atom_errors"] += -baseline[2]
        totals["baseline_bond_errors"] += -baseline[3]
        totals["candidate_atom_exact"] += candidate[1]
        totals["candidate_skeleton_exact"] += candidate[0]
        totals["candidate_atom_errors"] += -candidate[2]
        totals["candidate_bond_errors"] += -candidate[3]
        totals["atom_rescue"] += int(candidate[1] and not baseline[1])
        totals["atom_break"] += int(baseline[1] and not candidate[1])
        totals["skeleton_rescue"] += int(candidate[0] and not baseline[0])
        totals["skeleton_break"] += int(baseline[0] and not candidate[0])
        if selected_row is not None:
            cases.append(
                {
                    "key": sample["key"],
                    "removed": removed,
                    "safety_probability": float(safety[selected_row]),
                    "action_score": float(action_score[selected_row]),
                    "safe": bool(output["safe_targets"][selected_row]),
                    "winner": bool(output["winner_targets"][selected_row]),
                    "baseline_atom_exact": bool(baseline[1]),
                    "candidate_atom_exact": bool(candidate[1]),
                    "baseline_skeleton_exact": bool(baseline[0]),
                    "candidate_skeleton_exact": bool(candidate[0]),
                }
            )
    count = max(totals["samples"], 1)
    baseline_atom = totals["baseline_atom_exact"] / count
    candidate_atom = totals["candidate_atom_exact"] / count
    baseline_skeleton = totals["baseline_skeleton_exact"] / count
    candidate_skeleton = totals["candidate_skeleton_exact"] / count
    precision = totals["safe_executed_actions"] / max(totals["executed_actions"], 1)
    recall = totals["winner_actions"] / max(totals["oracle_available"], 1)
    return {
        **dict(totals),
        "threshold": threshold,
        "safe_action_precision": precision,
        "winner_recall": recall,
        "baseline_atom_exact_rate": baseline_atom,
        "candidate_atom_exact_rate": candidate_atom,
        "atom_exact_delta": candidate_atom - baseline_atom,
        "baseline_skeleton_exact_rate": baseline_skeleton,
        "candidate_skeleton_exact_rate": candidate_skeleton,
        "skeleton_exact_delta": candidate_skeleton - baseline_skeleton,
        "cases": cases,
    }


def select_threshold(model, samples, presence_threshold, forced_failures, device):
    output = model_action_outputs(model, samples, presence_threshold, device)
    if output is None:
        raise ValueError("Route-select cache contains no delete actions")
    grid = [
        _summarize_threshold(
            samples,
            output,
            threshold=threshold,
            presence_threshold=presence_threshold,
            forced_failures=forced_failures,
        )
        for threshold in THRESHOLDS
    ]
    for row in grid:
        row["safety_eligible"] = (
            row["harmful_actions"] == 0
            and row["atom_break"] == 0
            and row["skeleton_break"] == 0
            and row["skeleton_exact_delta"] >= 0.025
            and row["winner_recall"] >= 0.25
        )
    eligible = [row for row in grid if row["safety_eligible"]]
    selected = max(
        eligible or grid,
        key=lambda row: (
            int(row["safety_eligible"]),
            row["candidate_skeleton_exact_rate"],
            row["candidate_atom_exact_rate"],
            row["winner_recall"],
            row["threshold"],
        ),
    )
    return selected, grid


def _compact(value: dict) -> dict:
    return {name: item for name, item in value.items() if name != "cases"}


def save_checkpoint(path, model, optimizer, args, seen, validation, audit):
    torch.save(
        {
            "checkpoint_schema_version": "safe-atom-action-0.1",
            "model": model.state_dict(),
            "config": asdict(model.config),
            "optimizer": optimizer.state_dict(),
            "training_args": vars(args),
            "samples_seen": seen,
            "validation": validation,
            "audit": audit,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Safe Atom action training requires CUDA")
    domain_counts = {
        "style_train": args.style_train_samples,
        "chemvision_replay": args.chemvision_replay_samples,
        "font_v2_replay": args.font_v2_replay_samples,
    }
    if min(*domain_counts.values(), args.batch_size, args.validate_every_total_samples) < 1:
        raise ValueError("Training counts must be positive")
    if args.batch_size != 6:
        raise ValueError("The committed 4:1:1 schedule requires batch size 6")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data_gate = verify_data_gate()
    training_data_gate = verify_training_data_gate()
    cache, validation_samples, forced_failures, cache_commitment = load_action_cache(
        args.action_cache, args.oracle_report, args.validation_limit
    )
    _write_json(output_dir / "action_cache_commitment.json", cache_commitment)
    direct, direct_payload = load_direct_model(args.direct_checkpoint, device)
    direct_args = direct_payload["training_args"]
    if _sha256_file(args.direct_checkpoint) != cache["direct_checkpoint_sha256"]:
        raise ValueError("Training Direct checkpoint differs from action cache")
    vision, atom, bond, set_head, source = load_components(
        direct_args["source_checkpoint"], direct_args["grounded_graph_checkpoint"], device
    )
    for module in (direct, vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    presence_threshold = float(direct_payload["validation"]["threshold"])
    model = SafeAtomActionModel(SafeAtomActionConfig()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    audit = {
        "data_gate": data_gate,
        "training_data_gate": training_data_gate,
        "action_cache_commitment": cache_commitment,
        "direct_checkpoint": args.direct_checkpoint,
        "direct_checkpoint_sha256": _sha256_file(args.direct_checkpoint),
        "code_git_revision": code_revision,
        "parameters": model.parameter_count(),
        "route_confirm_read": False,
        "sealed_test_read": False,
    }
    _write_json(output_dir / "input_audit.json", audit)
    total_target = sum(domain_counts.values())
    seen = steps = actions_seen = safe_seen = winner_seen = 0
    next_validation = args.validate_every_total_samples
    best_score = None
    best_validation = None
    latest_losses = None
    for records, domain_seen in mixed_batches(
        domain_sample_counts=domain_counts,
        domain_batch_counts={"style_train": 4, "chemvision_replay": 1, "font_v2_replay": 1},
        batch_size=args.batch_size,
        config=scene_config,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    ):
        base_samples = collect_samples(
            direct,
            vision,
            atom,
            bond,
            set_head,
            records,
            collator,
            proposal_config,
            atom_threshold,
            presence_threshold,
            device,
        )
        prepared = action_training_batch(base_samples, presence_threshold)
        if prepared is not None:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            features = prepared["features"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(features)
            losses = safe_atom_action_loss(
                output,
                prepared["safe_targets"].to(device),
                prepared["utility_targets"].to(device),
                prepared["winner_targets"].to(device),
                prepared["sample_indices"].to(device),
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
            actions_seen += len(features)
            safe_seen += int(prepared["safe_targets"].sum())
            winner_seen += int(prepared["winner_targets"].sum())
            latest_losses = {name: float(value.detach()) for name, value in losses.items()}
        seen += len(records)
        if seen % 1_000 < len(records):
            print(
                json.dumps(
                    {
                        "samples_seen": seen,
                        "actions_seen": actions_seen,
                        "safe_seen": safe_seen,
                        "winner_seen": winner_seen,
                        "steps": steps,
                        "losses": latest_losses,
                        "domain_seen": domain_seen,
                    }
                ),
                flush=True,
            )
        if seen >= next_validation or seen == total_target:
            validation, grid = select_threshold(
                model,
                validation_samples,
                presence_threshold,
                forced_failures,
                device,
            )
            validation = {
                **validation,
                "samples_seen": seen,
                "grid": [_compact(row) for row in grid],
            }
            score = (
                int(validation["safety_eligible"]),
                validation["candidate_skeleton_exact_rate"],
                validation["candidate_atom_exact_rate"],
                validation["winner_recall"],
                validation["threshold"],
            )
            save_checkpoint(
                output_dir / "last.pt", model, optimizer, args, seen, validation, audit
            )
            save_checkpoint(
                output_dir / f"samples-{seen:06d}.pt",
                model,
                optimizer,
                args,
                seen,
                validation,
                audit,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_validation = validation
                save_checkpoint(
                    output_dir / "best.pt", model, optimizer, args, seen, validation, audit
                )
            _write_json(
                output_dir / f"validation-{seen:06d}.json",
                {"selected": _compact(validation), "grid": validation["grid"]},
            )
            print(
                json.dumps(
                    {
                        "validation": _compact(validation),
                        "best_samples_seen": best_validation["samples_seen"],
                    }
                ),
                flush=True,
            )
            while next_validation <= seen:
                next_validation += args.validate_every_total_samples
    payload = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    gate = {
        "minimum_skeleton_exact_delta": 0.025,
        "minimum_winner_recall": 0.25,
        "passed": bool(payload["validation"]["safety_eligible"]),
    }
    payload["route_select_gate"] = gate
    payload["training_elapsed_seconds"] = time.perf_counter() - started
    torch.save(payload, output_dir / "selected.pt")
    result = {
        "selected_step": payload["samples_seen"],
        "selected_validation": _compact(payload["validation"]),
        "route_select_gate": gate,
        "parameters": model.parameter_count(),
        "checkpoint": str(output_dir / "selected.pt"),
        "checkpoint_sha256": _sha256_file(output_dir / "selected.pt"),
        "route_confirm_read": False,
        "sealed_test_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "training_result.json", result)
    print(json.dumps({"result": result}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
