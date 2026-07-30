#!/usr/bin/env python3
"""Audit the v0.10.15 structural-twin certificate on route-select."""

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

from chemvector.safe_atom_action_model import SafeAtomActionConfig, SafeAtomActionModel
from chemvector.safe_atom_suppression import label_single_delete_actions
from chemvector.structural_twin_certificate import (
    StructuralTwinCertificateConfig,
    structural_twin_witnesses,
)
from chemvector.train_safe_atom_action import (
    ACTION_CACHE_SHA256,
    DEFAULT_CACHE,
    action_training_batch,
    load_action_cache,
)


DEFAULT_RANKING = (
    "outputs/checkpoints/chemvector-v01013-safe-atom-action-50k/samples-030000.pt"
)
RANKING_SHA256 = "43272dfc84e57e4f06ba344eb33a5850e8b62e4bf426c706f7975ae686bcdd18"
DEFAULT_OUTPUT = (
    "outputs/audits/chemvector-v01015-structural-twin-route-select/report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-cache", default=DEFAULT_CACHE)
    parser.add_argument("--ranking-checkpoint", default=DEFAULT_RANKING)
    parser.add_argument("--maximum-distance-px", type=float, default=2.0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _coverage_losses(baseline: dict, candidate: dict) -> dict[str, int]:
    return {
        name: int(not baseline[name].issubset(candidate[name]))
        for name in (
            "matched_atom_ids",
            "expressible_bond_ids",
            "covered_bond_ids",
            "typed_bond_ids",
        )
    }


def _selected_decisions(
    samples: list[dict],
    prepared: dict,
    certified: torch.Tensor,
    ranking_score: torch.Tensor,
    ranking_utility: torch.Tensor,
    presence_threshold: float,
    config: StructuralTwinCertificateConfig,
) -> list[dict]:
    decisions = []
    for sample_index, sample in enumerate(samples):
        baseline = tuple(sample["action_labels"]["baseline_utility"])
        label_by_removed = {
            int(action["removed"]): action
            for action in sample["action_labels"]["actions"]
        }
        rows = torch.where(prepared["sample_indices"] == sample_index)[0]
        qualified = rows[
            certified[rows]
            & (ranking_utility[rows] > 0)
            & (ranking_score[rows] > 0.0)
        ]
        selected = None
        if len(qualified):
            selected = int(qualified[ranking_score[qualified].argmax()])
        candidate = baseline
        removed = None
        safe = winner = False
        witnesses: tuple[int, ...] = ()
        losses = {
            "matched_atom_ids": 0,
            "expressible_bond_ids": 0,
            "covered_bond_ids": 0,
            "typed_bond_ids": 0,
        }
        if selected is not None:
            removed = int(prepared["removed_nodes"][selected])
            compact_action = label_by_removed[removed]
            candidate = tuple(compact_action["utility"])
            safe = bool(prepared["safe_targets"][selected])
            winner = bool(prepared["winner_targets"][selected])
            witnesses = structural_twin_witnesses(
                sample,
                removed,
                presence_threshold=presence_threshold,
                config=config,
            )
            full_labels = label_single_delete_actions(
                sample, presence_threshold=presence_threshold
            )
            full_action = next(
                action
                for action in full_labels["actions"]
                if int(action["removed"]) == removed
            )
            for mode in ("primary", "secondary"):
                current = _coverage_losses(
                    full_labels[f"baseline_{mode}"], full_action[mode]
                )
                losses = {
                    name: max(losses[name], current[name]) for name in losses
                }
        decisions.append(
            {
                "sample_index": sample_index,
                "key": sample["key"],
                "molecule_id": sample["molecule_id"],
                "variant": sample["variant"],
                "baseline": baseline,
                "candidate": candidate,
                "oracle_available": bool(
                    sample["action_labels"]["winner_removed_nodes"]
                ),
                "selected": selected is not None,
                "removed": removed,
                "witnesses": list(witnesses),
                "safe": safe,
                "winner": winner,
                "coverage_losses": losses,
                "ranking_score": (
                    None if selected is None else float(ranking_score[selected])
                ),
            }
        )
    return decisions


def _summarize(
    decisions: list[dict],
    *,
    molecule_ids: set[str],
    forced_failures: int,
) -> dict:
    totals = defaultdict(int)
    totals["samples"] = forced_failures
    totals["coordinate_collision_failures"] = forced_failures
    cases = []
    for row in decisions:
        if row["molecule_id"] not in molecule_ids:
            continue
        baseline = row["baseline"]
        candidate = row["candidate"]
        totals["samples"] += 1
        totals["oracle_available"] += int(row["oracle_available"])
        totals["executed_actions"] += int(row["selected"])
        totals["safe_executed_actions"] += int(row["selected"] and row["safe"])
        totals["harmful_actions"] += int(row["selected"] and not row["safe"])
        totals["winner_actions"] += int(row["winner"])
        totals["baseline_atom_exact"] += baseline[1]
        totals["baseline_skeleton_exact"] += baseline[0]
        totals["candidate_atom_exact"] += candidate[1]
        totals["candidate_skeleton_exact"] += candidate[0]
        totals["atom_rescue"] += int(candidate[1] and not baseline[1])
        totals["atom_break"] += int(baseline[1] and not candidate[1])
        totals["skeleton_rescue"] += int(candidate[0] and not baseline[0])
        totals["skeleton_break"] += int(baseline[0] and not candidate[0])
        for name, value in row["coverage_losses"].items():
            totals[f"lost_{name}"] += value
        if row["selected"]:
            cases.append({name: value for name, value in row.items() if name != "sample_index"})
    count = max(totals["samples"], 1)
    baseline_atom = totals["baseline_atom_exact"] / count
    candidate_atom = totals["candidate_atom_exact"] / count
    baseline_skeleton = totals["baseline_skeleton_exact"] / count
    candidate_skeleton = totals["candidate_skeleton_exact"] / count
    return {
        **dict(totals),
        "safe_action_precision": totals["safe_executed_actions"]
        / max(totals["executed_actions"], 1),
        "winner_recall": totals["winner_actions"]
        / max(totals["oracle_available"], 1),
        "baseline_atom_exact_rate": baseline_atom,
        "candidate_atom_exact_rate": candidate_atom,
        "atom_exact_delta": candidate_atom - baseline_atom,
        "baseline_skeleton_exact_rate": baseline_skeleton,
        "candidate_skeleton_exact_rate": candidate_skeleton,
        "skeleton_exact_delta": candidate_skeleton - baseline_skeleton,
        "cases": cases,
    }


def evaluate_structural_twin_samples(
    samples: list[dict],
    ranking: SafeAtomActionModel,
    *,
    presence_threshold: float,
    config: StructuralTwinCertificateConfig,
) -> tuple[dict, torch.Tensor, list[dict]]:
    prepared = action_training_batch(samples, presence_threshold)
    if prepared is None:
        raise ValueError("Structural-twin evaluation contains no delete actions")
    with torch.no_grad():
        ranking_output = ranking(prepared["features"])
        ranking_utility = ranking_output["utility_logits"].argmax(-1)
        ranking_score = (
            ranking_output["rank_logits"]
            + torch.nn.functional.logsigmoid(ranking_output["safety_logits"])
        ).float()
    certified = torch.tensor(
        [
            bool(
                structural_twin_witnesses(
                    samples[int(sample_index)],
                    int(removed),
                    presence_threshold=presence_threshold,
                    config=config,
                )
            )
            for sample_index, removed in zip(
                prepared["sample_indices"].tolist(),
                prepared["removed_nodes"].tolist(),
            )
        ],
        dtype=torch.bool,
    )
    decisions = _selected_decisions(
        samples,
        prepared,
        certified,
        ranking_score,
        ranking_utility,
        presence_threshold,
        config,
    )
    return prepared, certified, decisions


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = StructuralTwinCertificateConfig(
        maximum_distance_px=args.maximum_distance_px
    )
    cache, samples, forced_failures, cache_commitment = load_action_cache(
        args.action_cache,
        "outputs/audits/chemvector-v01013-safe-single-delete-oracle/report.json",
        None,
    )
    if _sha256_file(args.action_cache) != ACTION_CACHE_SHA256:
        raise ValueError("Structural-twin action cache differs")
    if _sha256_file(args.ranking_checkpoint) != RANKING_SHA256:
        raise ValueError("Structural-twin ranking checkpoint differs")
    presence_threshold = float(cache["presence_threshold"])
    ranking_payload = torch.load(
        args.ranking_checkpoint, map_location="cpu", weights_only=False
    )
    ranking = SafeAtomActionModel(
        SafeAtomActionConfig(**ranking_payload["config"])
    )
    ranking.load_state_dict(ranking_payload["model"])
    ranking.eval().requires_grad_(False)
    prepared, certified, decisions = evaluate_structural_twin_samples(
        samples,
        ranking,
        presence_threshold=presence_threshold,
        config=config,
    )
    close_molecules = {
        sample["molecule_id"]
        for sample in [*samples, *cache["coordinate_collisions"]]
        if sample["minimum_target_distance_px"] <= 16.0
    }
    all_molecules = {sample["molecule_id"] for sample in samples}
    all_molecules.update(
        row["molecule_id"] for row in cache["coordinate_collisions"]
    )
    general_molecules = all_molecules - close_molecules
    forced = {
        "all": forced_failures,
        "close_distinct": sum(
            row["molecule_id"] in close_molecules
            for row in cache["coordinate_collisions"]
        ),
        "general": sum(
            row["molecule_id"] in general_molecules
            for row in cache["coordinate_collisions"]
        ),
    }
    metrics = {
        name: _summarize(
            decisions, molecule_ids=molecules, forced_failures=forced[name]
        )
        for name, molecules in (
            ("all", all_molecules),
            ("close_distinct", close_molecules),
            ("general", general_molecules),
        )
    }
    action_audit = {
        "certified_actions": int(certified.sum()),
        "certified_safe_actions": int(
            (certified & prepared["safe_targets"]).sum()
        ),
        "certified_unsafe_actions": int(
            (certified & ~prepared["safe_targets"]).sum()
        ),
    }
    overall = metrics["all"]
    gate = {
        "passed": overall["harmful_actions"] == 0
        and overall["atom_break"] == 0
        and overall["skeleton_break"] == 0
        and overall["lost_matched_atom_ids"] == 0
        and overall["lost_covered_bond_ids"] == 0
        and overall["lost_typed_bond_ids"] == 0
        and overall["skeleton_exact_delta"] >= 0.025
        and overall["winner_actions"] / max(overall["oracle_available"], 1) >= 0.25,
        "minimum_skeleton_exact_delta": 0.025,
        "minimum_winner_recall": 0.25,
    }
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": "structural-twin-route-select-report-0.1",
        "status": "ROUTE-SELECT-GO" if gate["passed"] else "ROUTE-SELECT-NO-GO",
        "certificate": asdict(config),
        "inputs": {
            "action_cache": args.action_cache,
            "action_cache_sha256": _sha256_file(args.action_cache),
            "action_cache_commitment": cache_commitment["commitment_sha256"],
            "ranking_checkpoint": args.ranking_checkpoint,
            "ranking_checkpoint_sha256": _sha256_file(args.ranking_checkpoint),
            "route_confirm_read": False,
            "sealed_test_read": False,
        },
        "action_audit": action_audit,
        "metrics": metrics,
        "gate": gate,
        "code_git_revision": code_revision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(Path(args.output), report)
    compact = {
        name: {key: value for key, value in value.items() if key != "cases"}
        for name, value in metrics.items()
    }
    print(
        json.dumps(
            {"status": report["status"], "action_audit": action_audit, "metrics": compact, "gate": gate},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
