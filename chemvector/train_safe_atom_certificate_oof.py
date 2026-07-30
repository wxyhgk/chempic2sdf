#!/usr/bin/env python3
"""Nested molecule-OOF capacity audit for an independent unsafe certificate."""

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
from chemvector.safe_atom_certificate import (
    UnsafeActionCertificate,
    UnsafeActionCertificateConfig,
    molecule_fold,
    strict_unsafe_threshold,
    unsafe_certificate_loss,
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
DEFAULT_OUTPUT = "outputs/checkpoints/chemvector-v01014-unsafe-certificate-oof"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-cache", default=DEFAULT_CACHE)
    parser.add_argument("--ranking-checkpoint", default=DEFAULT_RANKING)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ensemble-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--unsafe-weight", type=float, default=8.0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260730)
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


@torch.no_grad()
def _predict(model, features, indices, batch_size):
    model.eval()
    rows = []
    for start in range(0, len(indices), batch_size):
        current = indices[start : start + batch_size]
        rows.append(model(features[current]).float().sigmoid().cpu())
    return torch.cat(rows) if rows else torch.empty(0)


def _train_certificate(
    features,
    unsafe,
    train_indices,
    *,
    seed,
    epochs,
    batch_size,
    learning_rate,
    unsafe_weight,
    device,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = UnsafeActionCertificate(UnsafeActionCertificateConfig()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latest = None
    for _ in range(epochs):
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        model.train()
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(features[rows])
            loss = unsafe_certificate_loss(
                logits, unsafe[rows], unsafe_weight=unsafe_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            latest = float(loss.detach())
    return model, latest


def _summarize(samples, prepared, accepted, ranking_score, ranking_utility, forced):
    totals = defaultdict(int)
    totals["samples"] = forced
    totals["coordinate_collision_failures"] = forced
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
    cases = []
    for sample_index, sample in enumerate(samples):
        baseline = tuple(sample["action_labels"]["baseline_utility"])
        label_by_removed = {
            int(action["removed"]): action
            for action in sample["action_labels"]["actions"]
        }
        rows = torch.where(prepared["sample_indices"] == sample_index)[0]
        qualified = rows[
            accepted[rows]
            & (ranking_utility[rows] > 0)
            & (ranking_score[rows] > 0.0)
        ]
        selected = None
        if len(qualified):
            selected = int(qualified[ranking_score[qualified].argmax()])
        candidate = baseline
        removed = None
        if selected is not None:
            removed = int(prepared["removed_nodes"][selected])
            candidate = tuple(label_by_removed[removed]["utility"])
        totals["samples"] += 1
        totals["oracle_available"] += int(
            bool(sample["action_labels"]["winner_removed_nodes"])
        )
        totals["executed_actions"] += int(selected is not None)
        if selected is not None:
            is_safe = bool(prepared["safe_targets"][selected])
            is_winner = bool(prepared["winner_targets"][selected])
            totals["safe_executed_actions"] += int(is_safe)
            totals["harmful_actions"] += int(not is_safe)
            totals["winner_actions"] += int(is_winner)
        totals["baseline_atom_exact"] += baseline[1]
        totals["baseline_skeleton_exact"] += baseline[0]
        totals["candidate_atom_exact"] += candidate[1]
        totals["candidate_skeleton_exact"] += candidate[0]
        totals["atom_rescue"] += int(candidate[1] and not baseline[1])
        totals["atom_break"] += int(baseline[1] and not candidate[1])
        totals["skeleton_rescue"] += int(candidate[0] and not baseline[0])
        totals["skeleton_break"] += int(baseline[0] and not candidate[0])
        if selected is not None:
            cases.append(
                {
                    "key": sample["key"],
                    "removed": removed,
                    "safe": bool(prepared["safe_targets"][selected]),
                    "winner": bool(prepared["winner_targets"][selected]),
                }
            )
    count = max(totals["samples"], 1)
    baseline_atom = totals["baseline_atom_exact"] / count
    candidate_atom = totals["candidate_atom_exact"] / count
    baseline_skeleton = totals["baseline_skeleton_exact"] / count
    candidate_skeleton = totals["candidate_skeleton_exact"] / count
    return {
        **dict(totals),
        "safe_action_precision": totals["safe_executed_actions"]
        / max(totals["executed_actions"], 1),
        "winner_recall": totals["winner_actions"] / max(totals["oracle_available"], 1),
        "baseline_atom_exact_rate": baseline_atom,
        "candidate_atom_exact_rate": candidate_atom,
        "atom_exact_delta": candidate_atom - baseline_atom,
        "baseline_skeleton_exact_rate": baseline_skeleton,
        "candidate_skeleton_exact_rate": candidate_skeleton,
        "skeleton_exact_delta": candidate_skeleton - baseline_skeleton,
        "cases": cases,
    }


def _compact(value):
    return {name: item for name, item in value.items() if name != "cases"}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Unsafe certificate OOF requires CUDA")
    if args.folds != 5:
        raise ValueError("The committed nested OOF protocol requires five folds")
    if min(
        args.ensemble_seeds,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.unsafe_weight,
    ) <= 0:
        raise ValueError("OOF training arguments must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    cache, samples, forced_failures, cache_commitment = load_action_cache(
        args.action_cache,
        "outputs/audits/chemvector-v01013-safe-single-delete-oracle/report.json",
        None,
    )
    if _sha256_file(args.action_cache) != ACTION_CACHE_SHA256:
        raise ValueError("OOF action cache differs")
    if _sha256_file(args.ranking_checkpoint) != RANKING_SHA256:
        raise ValueError("OOF ranking checkpoint differs")
    prepared = action_training_batch(samples, float(cache["presence_threshold"]))
    features_cpu = prepared["features"]
    features = features_cpu.to(device)
    unsafe = (~prepared["safe_targets"]).to(device)
    sample_folds = torch.tensor(
        [molecule_fold(sample["molecule_id"], folds=args.folds) for sample in samples]
    )
    action_folds = sample_folds[prepared["sample_indices"]]
    ranking_payload = torch.load(
        args.ranking_checkpoint, map_location="cpu", weights_only=False
    )
    ranking = SafeAtomActionModel(
        SafeAtomActionConfig(**ranking_payload["config"])
    ).to(device)
    ranking.load_state_dict(ranking_payload["model"])
    ranking.eval().requires_grad_(False)
    with torch.no_grad():
        ranking_output = ranking(features)
        ranking_utility = ranking_output["utility_logits"].argmax(-1).cpu()
        ranking_score = (
            ranking_output["rank_logits"]
            + torch.nn.functional.logsigmoid(ranking_output["safety_logits"])
        ).float().cpu()

    accepted_by_variant = {
        "single_seed": torch.zeros(len(features_cpu), dtype=torch.bool),
        "conservative_ensemble": torch.zeros(len(features_cpu), dtype=torch.bool),
    }
    fold_reports = []
    for outer_fold in range(args.folds):
        calibration_fold = (outer_fold + 1) % args.folds
        train_sample_indices = torch.where(
            (sample_folds != outer_fold) & (sample_folds != calibration_fold)
        )[0]
        calibration_sample_indices = torch.where(sample_folds == calibration_fold)[0]
        evaluation_sample_indices = torch.where(sample_folds == outer_fold)[0]
        train_molecules = {
            samples[index]["molecule_id"] for index in train_sample_indices.tolist()
        }
        calibration_molecules = {
            samples[index]["molecule_id"]
            for index in calibration_sample_indices.tolist()
        }
        evaluation_molecules = {
            samples[index]["molecule_id"]
            for index in evaluation_sample_indices.tolist()
        }
        if (
            train_molecules & calibration_molecules
            or train_molecules & evaluation_molecules
            or calibration_molecules & evaluation_molecules
        ):
            raise RuntimeError("Molecule leakage detected between nested OOF splits")
        train_indices = torch.where(
            (action_folds != outer_fold) & (action_folds != calibration_fold)
        )[0]
        calibration_indices = torch.where(action_folds == calibration_fold)[0]
        evaluation_indices = torch.where(action_folds == outer_fold)[0]
        seed_calibration = []
        seed_evaluation = []
        seed_reports = []
        for seed_offset in range(args.ensemble_seeds):
            seed = args.seed + outer_fold * 100 + seed_offset
            model, latest_loss = _train_certificate(
                features,
                unsafe,
                train_indices,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                unsafe_weight=args.unsafe_weight,
                device=device,
            )
            calibration_probabilities = _predict(
                model, features, calibration_indices, args.batch_size
            )
            evaluation_probabilities = _predict(
                model, features, evaluation_indices, args.batch_size
            )
            seed_calibration.append(calibration_probabilities)
            seed_evaluation.append(evaluation_probabilities)
            checkpoint = output_dir / f"fold-{outer_fold}-seed-{seed_offset}.pt"
            torch.save(
                {
                    "checkpoint_schema_version": "unsafe-action-certificate-0.1",
                    "model": model.state_dict(),
                    "config": asdict(model.config),
                    "outer_fold": outer_fold,
                    "calibration_fold": calibration_fold,
                    "seed": seed,
                    "epochs": args.epochs,
                    "latest_loss": latest_loss,
                    "cache_commitment": cache_commitment,
                },
                checkpoint,
            )
            seed_reports.append(
                {
                    "seed": seed,
                    "latest_loss": latest_loss,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256_file(checkpoint),
                }
            )
        calibration_stack = torch.stack(seed_calibration)
        evaluation_stack = torch.stack(seed_evaluation)
        calibration_unsafe = unsafe[calibration_indices.to(device)].cpu()
        thresholds = {}
        for name, calibration_scores, evaluation_scores in (
            ("single_seed", calibration_stack[0], evaluation_stack[0]),
            (
                "conservative_ensemble",
                calibration_stack.max(dim=0).values,
                evaluation_stack.max(dim=0).values,
            ),
        ):
            threshold = strict_unsafe_threshold(
                calibration_scores[calibration_unsafe]
            )
            accepted_by_variant[name][evaluation_indices] = (
                evaluation_scores < threshold
            )
            thresholds[name] = {
                "threshold": threshold,
                "calibration_unsafe": int(calibration_unsafe.sum()),
                "calibration_unsafe_accepted": int(
                    (calibration_scores[calibration_unsafe] < threshold).sum()
                ),
                "evaluation_actions_accepted": int(
                    (evaluation_scores < threshold).sum()
                ),
            }
        fold_reports.append(
            {
                "outer_fold": outer_fold,
                "calibration_fold": calibration_fold,
                "train_samples": len(train_sample_indices),
                "calibration_samples": len(calibration_sample_indices),
                "evaluation_samples": len(evaluation_sample_indices),
                "train_molecules": len(train_molecules),
                "calibration_molecules": len(calibration_molecules),
                "evaluation_molecules": len(evaluation_molecules),
                "molecule_overlap": 0,
                "train_actions": len(train_indices),
                "calibration_actions": len(calibration_indices),
                "evaluation_actions": len(evaluation_indices),
                "thresholds": thresholds,
                "seeds": seed_reports,
            }
        )
        print(json.dumps({"fold": fold_reports[-1]}), flush=True)

    variants = {}
    for name, accepted in accepted_by_variant.items():
        metrics = _summarize(
            samples,
            prepared,
            accepted,
            ranking_score,
            ranking_utility,
            forced_failures,
        )
        gate = {
            "passed": metrics["harmful_actions"] == 0
            and metrics["atom_break"] == 0
            and metrics["skeleton_break"] == 0
            and metrics["skeleton_exact_delta"] >= 0.025
            and metrics["winner_recall"] >= 0.25,
            "minimum_skeleton_exact_delta": 0.025,
            "minimum_winner_recall": 0.25,
        }
        variants[name] = {"metrics": metrics, "gate": gate}
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": "unsafe-action-certificate-nested-oof-report-0.1",
        "status": (
            "OOF-GO"
            if variants["conservative_ensemble"]["gate"]["passed"]
            else "OOF-NO-GO"
        ),
        "protocol": {
            "folds": args.folds,
            "train_folds": 3,
            "calibration_folds": 1,
            "evaluation_folds": 1,
            "molecule_grouped": True,
            "ensemble_seeds": args.ensemble_seeds,
            "epochs": args.epochs,
            "unsafe_weight": args.unsafe_weight,
            "threshold": "next-float-below-minimum-calibration-unsafe",
        },
        "inputs": {
            "action_cache": args.action_cache,
            "action_cache_sha256": _sha256_file(args.action_cache),
            "action_cache_commitment": cache_commitment["commitment_sha256"],
            "ranking_checkpoint": args.ranking_checkpoint,
            "ranking_checkpoint_sha256": _sha256_file(args.ranking_checkpoint),
            "samples": len(samples),
            "actions": len(features_cpu),
            "unsafe_actions": int(unsafe.sum()),
            "route_confirm_read": False,
            "sealed_test_read": False,
        },
        "folds": fold_reports,
        "variants": {
            name: {"metrics": _compact(value["metrics"]), "gate": value["gate"]}
            for name, value in variants.items()
        },
        "code_git_revision": code_revision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "oof_report.json", report)
    print(json.dumps({"report": report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
