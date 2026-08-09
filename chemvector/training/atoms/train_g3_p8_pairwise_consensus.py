#!/usr/bin/env python3
"""Train and certify a three-seed P8 pairwise visual consensus."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from chemvector.components.recovery.pairwise_structural_counterfactual import (
    IMAGE_SIZE,
    PairwiseStructuralCounterfactual,
    PairwiseStructuralCounterfactualConfig,
)
from chemvector.components.recovery.structural_peak_residual import normalized_peak_prior
from chemvector.evaluation.audits.atoms import build_g3_p7e0_pairwise_counterfactual_data as p7e0
from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9
from chemvector.training.atoms import train_g3_p7_pairwise_counterfactual as p7
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json, _atomic_torch_save


CYCLE_NAME = "chemvector-g3-p8-pairwise-consensus"
OUTPUT = Path("outputs/checkpoints") / CYCLE_NAME
BASE_MODEL_TYPE = "chemvector-g3-p7e2-pairwise-counterfactual-patch-only-train11762"
BASE_CHECKPOINT = Path("outputs/checkpoints") / BASE_MODEL_TYPE / "step-008000.pt"
BASE_SHA256 = "a5fb6356a442957f67177288512c1af3f5560b6e00a5d80625fa46ca65956687"
MEMBER_SEEDS = (2026089201, 2026089202)
SCHEDULE_SEEDS = (2026089211, 2026089212)
FULL_STEPS = 8_000
FOLDS = 5
MINIMUM_FULL_RESCUES = 80
MINIMUM_CROSS_CERTIFICATE_RESCUES = 60
BATCH_SIZE = 64


def certificate_fold(scaffold: str) -> int:
    digest = hashlib.sha256(f"p8-cert:{scaffold}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % FOLDS


def fixed_threshold_metrics(records: list[dict], threshold: float) -> dict:
    executed = [row for row in records if float(row["score"]) >= threshold]
    harmful = sum(row["prior_good"] and not row["alternative_good"] for row in executed)
    rescue = sum(not row["prior_good"] and row["alternative_good"] for row in executed)
    return {
        "threshold": threshold,
        "executed": len(executed),
        "rescues": rescue,
        "harmful": harmful,
        "useless": len(executed) - rescue - harmful,
    }


def select_unanimous_candidate(member_scores: torch.Tensor) -> tuple[int | None, float]:
    """Return the common winner and its worst-member score, or no action."""

    if member_scores.ndim != 2 or member_scores.size(0) < 2:
        raise ValueError("member_scores must have shape [members,candidates]")
    if not bool(torch.isfinite(member_scores).all()):
        raise ValueError("member_scores must be finite")
    winners = member_scores.argmax(dim=1)
    if not bool((winners == winners[0]).all()):
        return None, -math.inf
    winner = int(winners[0])
    return winner, float(member_scores[:, winner].min())


def _load_model(path: Path, expected_sha256: str, device: torch.device) -> PairwiseStructuralCounterfactual:
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"P8 member checkpoint SHA256 differs: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = PairwiseStructuralCounterfactual(
        PairwiseStructuralCounterfactualConfig(arm="patch-only")
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_consensus_cache(
    models: list[PairwiseStructuralCounterfactual],
    cache: dict,
    source_kind: str,
    device: torch.device,
) -> list[dict]:
    records = []
    for row in range(len(cache["offset_targets"])):
        mask = cache["candidate_mask"][row]
        alternatives = torch.where(mask)[0].tolist()
        normalized = normalized_peak_prior(
            cache["candidate_peak_logits"][row : row + 1].float(), mask[None]
        )[0]
        prior_index = int(normalized.argmax())
        alternatives.remove(prior_index)
        count = len(alternatives)
        image = cache["image_crops"][row : row + 1].to(device).float().expand(count, -1, -1, -1)
        score = cache["score_crops"][row : row + 1].to(device).float().expand(count, -1, -1, -1)
        offsets = cache["candidate_offsets"][row].to(device).float()
        prior_offsets = offsets[prior_index : prior_index + 1].expand(count, -1)
        alternative_offsets = offsets[torch.tensor(alternatives, device=device)]
        mask_tensor = torch.ones(count, dtype=torch.bool, device=device)
        base_margin = normalized[alternatives].to(device) - normalized[prior_index].to(device)
        scores = []
        for model in models:
            delta = model(
                image,
                score,
                torch.zeros_like(prior_offsets),
                prior_offsets,
                alternative_offsets,
                mask_tensor,
            )
            scores.append((base_margin + delta.float()).cpu())
        winner, certificate = select_unanimous_candidate(torch.stack(scores))
        target = cache["offset_targets"][row].float()
        candidate_offsets = cache["candidate_offsets"][row].float()
        prior_distance = float(
            torch.linalg.vector_norm((candidate_offsets[prior_index] - target) * IMAGE_SIZE)
        )
        if winner is None:
            alternative_index = prior_index
            alternative_distance = prior_distance
        else:
            alternative_index = alternatives[winner]
            alternative_distance = float(
                torch.linalg.vector_norm(
                    (candidate_offsets[alternative_index] - target) * IMAGE_SIZE
                )
            )
        records.append(
            {
                "source": source_kind,
                "row": row,
                "scaffold": str(cache["scaffold"][row]),
                "score": certificate,
                "consensus": winner is not None,
                "prior_good": prior_distance <= 6.0,
                "alternative_good": alternative_distance <= 6.0,
                "prior_index": prior_index,
                "alternative_index": alternative_index,
            }
        )
    return records


def calibrate(records: list[dict]) -> dict:
    finite = [row for row in records if math.isfinite(float(row["score"]))]
    return p7.calibrate_zero_harmful(finite)


def cross_certificate(records: list[dict]) -> dict:
    folds = []
    totals = {"executed": 0, "rescues": 0, "harmful": 0, "useless": 0}
    for fold in range(FOLDS):
        selection = [row for row in records if certificate_fold(row["scaffold"]) != fold]
        certificate = [row for row in records if certificate_fold(row["scaffold"]) == fold]
        threshold = calibrate(selection)
        metrics = fixed_threshold_metrics(certificate, threshold["threshold"])
        folds.append({"fold": fold, "selection": threshold, "certificate": metrics, "rows": len(certificate)})
        for name in totals:
            totals[name] += int(metrics[name])
    return {"folds": folds, "totals": totals}


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P8 output already exists: {OUTPUT}")
    if file_sha256(BASE_CHECKPOINT) != BASE_SHA256:
        raise ValueError("P8 base checkpoint SHA256 differs")
    if any((path.stat().st_mode & 0o777) != 0 for path in p6e9.HOLDOUT_FILES):
        raise RuntimeError("P8 historical holdout must remain sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("P8 requires CUDA")
    device = torch.device("cuda")
    pairs = torch.load(p7e0.TRAIN_PAIRS, map_location="cpu", weights_only=False)
    sources = p7._load_sources()
    pools = p7.select_training_pairs(pairs, sources["train"])
    models = [_load_model(BASE_CHECKPOINT, BASE_SHA256, device)]
    members = [
        {
            "model_type": BASE_MODEL_TYPE,
            "checkpoint": str(BASE_CHECKPOINT.resolve()),
            "checkpoint_sha256": BASE_SHA256,
            "reused": True,
        }
    ]
    for member_seed, schedule_seed in zip(MEMBER_SEEDS, SCHEDULE_SEEDS, strict=True):
        model, training = p7._train_model(
            "patch-only",
            pairs,
            sources["train"],
            pools,
            steps=FULL_STEPS,
            seed=schedule_seed,
            model_seed=member_seed,
            device=device,
        )
        model_type = f"chemvector-g3-p8e1-pairwise-consensus-seed{member_seed}-train11762"
        checkpoint = Path("outputs/checkpoints") / model_type / f"step-{FULL_STEPS:06d}.pt"
        checkpoint.parent.mkdir(parents=True)
        _atomic_torch_save(
            checkpoint,
            {
                "checkpoint_schema_version": "g3-p8-pairwise-consensus-member-0.1",
                "model_type": model_type,
                "model_config": model.config_dict(),
                "model": model.state_dict(),
                "global_step": FULL_STEPS,
                "model_seed": member_seed,
                "schedule_seed": schedule_seed,
            },
        )
        members.append(
            {
                "model_type": model_type,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
                "training": training,
                "reused": False,
            }
        )
        _atomic_json(checkpoint.parent / "audit.json", members[-1])
        models.append(model)
    records = evaluate_consensus_cache(models, sources["calibration"][0], "hard", device)
    records += evaluate_consensus_cache(models, sources["calibration"][1], "easy", device)
    full = calibrate(records)
    cross = cross_certificate(records)
    checks = {
        "three_members": len(models) == 3,
        "full_zero_harmful": full["harmful"] == 0,
        "full_rescues_min80": full["rescues"] >= MINIMUM_FULL_RESCUES,
        "cross_certificate_zero_harmful": cross["totals"]["harmful"] == 0,
        "cross_certificate_rescues_min60": cross["totals"]["rescues"] >= MINIMUM_CROSS_CERTIFICATE_RESCUES,
        "historical_holdout_mode_000": all((path.stat().st_mode & 0o777) == 0 for path in p6e9.HOLDOUT_FILES),
    }
    decision = "pairwise_consensus_certificate_go" if all(checks.values()) else "pairwise_consensus_certificate_no_go"
    manifest = {
        "schema_version": "g3-p8-pairwise-consensus-audit-0.1",
        "decision": decision,
        "checks": checks,
        "members": members,
        "full_calibration": full,
        "cross_certificate": cross,
        "training_pools": {name: len(value) for name, value in pools.items()},
        "protocol": {
            "consensus": "all three members choose the same alternative",
            "certificate_score": "minimum corrected margin across members",
            "threshold": "zero harmful, maximum rescue",
            "historical_p7_holdout_read": False,
            "dev_confirm_external_sealed_read": False,
        },
        "historical_p7_holdout_read": False,
        "dev_confirm_external_sealed_read": False,
    }
    OUTPUT.mkdir(parents=True)
    _atomic_json(OUTPUT / "audit.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
