#!/usr/bin/env python3
"""Train identity-invariant soft candidate coverage on Micro160."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from chemvector.components.atoms.objects.g3_latent_matching import (
    g3_gt_candidate_compatibility,
)
from chemvector.components.atoms.objects.g3_poisson_binomial_count import (
    G3PoissonBinomialCount,
    G3PoissonBinomialCountConfig,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e23_top192_cycle_recovery as p3e23,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e24_top192_input_protocol_ab as p3e24,
)
from chemvector.training.atoms import audit_g3_p3e7_base_expert_router as p3e7
from chemvector.training.atoms import (
    train_g3_p3e25_hard_latent_occupancy_micro160 as p3e25,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p3e26-soft-coverage-occupancy-micro160"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
P3E25_AUDIT = p3e25.OUTPUT / "audit.json"
P3E25_AUDIT_SHA256 = "2bea5768cd3ac9e83725032f1737bbe3b68c00098cc2b85a4efe02ff678642b4"
SAMPLE_KEYS_SHA256 = "06f44d30a88013643be6041fe55741fd75f6d58bcf1d875a22224972a7944501"
INPUT_TENSORS_SHA256 = "274222ec409a6f90999976cbb7898af6ca4d8dca9c5597fcea28c515353d8a9a"


def compatibility_components(compatibility: torch.Tensor) -> list[dict]:
    if compatibility.ndim != 2 or compatibility.dtype != torch.bool:
        raise ValueError("P3-E26 compatibility must be GT-by-candidate bool")
    gt_count, candidate_count = compatibility.shape
    if gt_count < 1 or candidate_count < 1 or not bool(compatibility.any(dim=1).all()):
        raise ValueError("P3-E26 every GT requires a compatible candidate")
    adjacency = (
        compatibility.float() @ compatibility.float().T
    ) > 0
    remaining = set(range(gt_count))
    output = []
    while remaining:
        stack = [min(remaining)]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(
                index for index in torch.where(adjacency[current])[0].tolist()
                if index not in component
            )
        remaining -= component
        gt_indices = torch.tensor(sorted(component), dtype=torch.long)
        candidate_mask = compatibility[gt_indices].any(dim=0)
        output.append({"gt_indices": gt_indices, "candidate_mask": candidate_mask})
    return output


def soft_coverage_loss(output: dict[str, torch.Tensor], records: list[dict]) -> dict:
    logits = output["presence_logits"].float()
    probabilities = logits.sigmoid()
    coverage_losses = []
    mass_losses = []
    outside_losses = []
    for index, record in enumerate(records):
        expected = record["atom_anchors"][record["full_atom_mask"]].float()
        compatibility = g3_gt_candidate_compatibility(
            expected, record["candidate_anchors"].float()
        ).to(logits.device)
        components = compatibility_components(compatibility)
        per_gt = []
        for mask in compatibility:
            uncovered = (1.0 - probabilities[index, mask]).prod()
            per_gt.append(-torch.log1p(-uncovered.clamp(max=1.0 - 1e-6)))
        coverage_losses.append(torch.stack(per_gt).mean())
        per_component = []
        for component in components:
            mask = component["candidate_mask"].to(logits.device)
            target = float(len(component["gt_indices"]))
            per_component.append(
                (probabilities[index, mask].sum() - target).abs() / target
            )
        mass_losses.append(torch.stack(per_component).mean())
        outside = ~compatibility.any(dim=0)
        outside_losses.append(F.softplus(logits[index, outside]).mean())
    coverage_nll = torch.stack(coverage_losses).mean()
    component_mass = torch.stack(mass_losses).mean()
    outside_negative = torch.stack(outside_losses).mean()
    labels = torch.tensor(
        [int(record["full_atom_mask"].sum()) for record in records],
        device=logits.device,
    )
    count_nll = F.nll_loss(p3e25.bounded_count_log_probs(output), labels)
    loss = coverage_nll + 0.5 * component_mass + 0.5 * outside_negative + 0.1 * count_nll
    return {
        "loss": loss,
        "coverage_nll": coverage_nll,
        "component_relative_mass_l1": component_mass,
        "outside_negative_bce": outside_negative,
        "count_nll": count_nll,
    }


@torch.no_grad()
def soft_diagnostics(model, records: list[dict], device: torch.device) -> dict:
    model.eval()
    coverage_minima = []
    component_errors = []
    digest_values = []
    for start in range(0, len(records), p3e25.BATCH_SIZE):
        indices = list(range(start, min(start + p3e25.BATCH_SIZE, len(records))))
        current, evidence = p3e25.batch_evidence(records, indices, device)
        with _autocast_context(device, "bf16"):
            output = model(*evidence)
        probabilities = output["presence_logits"].float().sigmoid().cpu()
        digest_values.append(probabilities)
        for index, record in enumerate(current):
            expected = record["atom_anchors"][record["full_atom_mask"]].float()
            compatibility = g3_gt_candidate_compatibility(
                expected, record["candidate_anchors"].float()
            )
            coverages = []
            for mask in compatibility:
                coverages.append(
                    1.0 - float((1.0 - probabilities[index, mask]).prod())
                )
            errors = []
            for component in compatibility_components(compatibility):
                target = len(component["gt_indices"])
                mass = float(probabilities[index, component["candidate_mask"]].sum())
                errors.append(abs(mass - target) / target)
            coverage_minima.append(min(coverages))
            component_errors.append(max(errors))
    digest = p3e23.tensor_digest(digest_values)
    return {
        "images": len(records),
        "minimum_gt_coverage_min": min(coverage_minima),
        "minimum_gt_coverage_mean": sum(coverage_minima) / len(coverage_minima),
        "maximum_component_relative_mass_error_max": max(component_errors),
        "maximum_component_relative_mass_error_mean": (
            sum(component_errors) / len(component_errors)
        ),
        "coverage95_mass10_images": sum(
            coverage >= 0.95 and error <= 0.10
            for coverage, error in zip(coverage_minima, component_errors)
        ),
        "prediction_digest": digest,
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E26 output is not empty: {OUTPUT}")
    for path, expected in (
        (P3E25_AUDIT, P3E25_AUDIT_SHA256),
        (p3e23.TOP256_CACHE, p3e23.TOP256_CACHE_SHA256),
        (p3e23.OLD_CACHE_ROOT / "train.pt", p3e23.OLD_CACHE_SHA256["train"]),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E26 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P3-E26 requires CUDA BF16")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(p3e25.SEED)
    np.random.seed(p3e25.SEED)
    torch.manual_seed(p3e25.SEED)
    torch.cuda.manual_seed_all(p3e25.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    old = torch.load(
        p3e23.OLD_CACHE_ROOT / "train.pt", map_location="cpu", weights_only=False
    )["records"]
    old_by_key = {str(record["sample_key"]): record for record in old}
    top256 = torch.load(p3e23.TOP256_CACHE, map_location="cpu", weights_only=False)
    records, input_audit = p3e25.true_global192_records(top256, old)
    micro, selection_audit = p3e25.select_micro160(records, old)
    if selection_audit["sample_keys_sha256"] != SAMPLE_KEYS_SHA256:
        raise ValueError("P3-E26 Micro160 selection differs from P3-E25")
    if input_audit["candidate_tensors_sha256"] != INPUT_TENSORS_SHA256:
        raise ValueError("P3-E26 true-global192 input differs from P3-E25")

    base_model = p3e23.load_dynamic_model(p3e7.BASE_CHECKPOINT, p3e7.BASE_SHA256, device)
    expert_model = p3e23.load_dynamic_model(p3e7.EXPERT_CHECKPOINT, p3e7.EXPERT_SHA256, device)
    evidence, compatibility = p3e24.build_query_evidence(
        {"micro160": micro}, base_model, expert_model, device
    )
    micro = evidence["micro160"]
    frozen_base_sha = module_state_sha256(base_model)
    frozen_expert_sha = module_state_sha256(expert_model)
    del base_model, expert_model

    config = G3PoissonBinomialCountConfig(hidden_dim=64)
    model = G3PoissonBinomialCount(config).to(device)
    with torch.no_grad():
        model.occupancy[-1].weight.zero_()
        model.occupancy[-1].bias.fill_(-1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=p3e25.LEARNING_RATE, weight_decay=p3e25.WEIGHT_DECAY
    )
    batches, schedule = p3e25.training_batches()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    audit = {
        "schema_version": "g3-p3e26-soft-coverage-occupancy-micro160-audit-0.1",
        "model_type": MODEL_TYPE,
        "status": "training",
        "input": input_audit,
        "selection": selection_audit,
        "schedule": schedule,
        "model_config": asdict(config),
        "trainable_parameters": sum(value.numel() for value in model.parameters()),
        "loss": {
            "coverage_nll": 1.0,
            "component_relative_mass_l1": 0.5,
            "outside_negative_bce": 0.5,
            "count_nll": 0.1,
            "hard_identity": False,
        },
        "lineage": {
            "p3e25_audit_sha256": P3E25_AUDIT_SHA256,
            "sample_keys_sha256": SAMPLE_KEYS_SHA256,
            "input_tensors_sha256": INPUT_TENSORS_SHA256,
            "base_checkpoint_sha256": p3e7.BASE_SHA256,
            "expert_checkpoint_sha256": p3e7.EXPERT_SHA256,
            "frozen_base_state_sha256": frozen_base_sha,
            "frozen_expert_state_sha256": frozen_expert_sha,
            "compatibility": compatibility,
            "dev_read": False,
            "old_confirm_read": False,
            "fresh_dev_read": False,
            "sealed_test_read": False,
        },
    }
    _atomic_json(OUTPUT / "audit.json", audit)

    started = time.perf_counter()
    sums = defaultdict(float)
    log_path = OUTPUT / "train.jsonl"
    for step, indices in enumerate(batches, start=1):
        current, inputs = p3e25.batch_evidence(micro, indices, device)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        with _autocast_context(device, "bf16"):
            output = model(*inputs)
            losses = soft_coverage_loss(output, current)
        losses["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), p3e25.MAX_GRAD_NORM
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("P3-E26 gradient norm is not finite")
        optimizer.step()
        for name, value in losses.items():
            sums[name] += float(value.detach())
        sums["gradient_norm"] += float(gradient_norm)
        if step % 50 == 0 or step == p3e25.STEPS:
            record = {
                "step": step,
                "epoch": (step + p3e25.STEPS_PER_EPOCH - 1) // p3e25.STEPS_PER_EPOCH,
                "exposure": min(step * p3e25.BATCH_SIZE, p3e25.EXPOSURE),
                **{f"mean_{name}": value / step for name, value in sums.items()},
                "elapsed_seconds": time.perf_counter() - started,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            print(json.dumps(record, separators=(",", ":")), flush=True)

    metrics = p3e25.evaluate(model, micro, old_by_key, device)
    soft = soft_diagnostics(model, micro, device)
    repeated_metrics = p3e25.evaluate(model, micro, old_by_key, device)
    repeated_soft = soft_diagnostics(model, micro, device)
    predicted = metrics["predicted"]
    checks = {
        "count_exact_min152": metrics["count_exact"] >= 152,
        "count_mae_max0_10": metrics["count_mae"] <= 0.10,
        "predicted_atom_exact_min152": predicted["whole_set_exact_images"] >= 152,
        "known_atom_exact_min152": metrics["known_count"]["whole_set_exact_images"] >= 152,
        "predicted_atom_recall_min999": (
            predicted["matched_atoms"] / predicted["target_atoms"] >= 0.999
        ),
        "newly_reachable_exact_min50pct": metrics["newly_reachable_exact_rate"] >= 0.5,
        "all_count_buckets_min90pct": all(
            value["count_exact"] / value["images"] >= 0.9
            and value["atom_exact"] / value["images"] >= 0.9
            for value in metrics["by_count"].values()
        ),
        "all_variants_min90pct": all(
            value["count_exact"] / value["images"] >= 0.9
            and value["atom_exact"] / value["images"] >= 0.9
            for value in metrics["by_variant"].values()
        ),
        "coverage95_mass10_min152": soft["coverage95_mass10_images"] >= 152,
        "count_distribution_finite_normalized": (
            metrics["count_distribution_finite_normalized"]
        ),
        "repeat_metrics": repeated_metrics == metrics and repeated_soft == soft,
        "fixed_steps": len(batches) == p3e25.STEPS,
        "fixed_exposure": schedule["exposure"] == p3e25.EXPOSURE,
    }
    decision = (
        "soft_coverage_occupancy_micro160_go"
        if all(checks.values())
        else "soft_coverage_occupancy_micro160_no_go"
    )
    final = {
        "decision": decision,
        "checks": checks,
        "metrics": metrics,
        "soft_diagnostics": soft,
        "training_seconds": time.perf_counter() - started,
    }
    payload = {
        "checkpoint_schema_version": "g3-p3e26-soft-coverage-occupancy-micro160-0.1",
        "model_type": MODEL_TYPE,
        "model_config": asdict(config),
        "soft_coverage_occupancy": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": p3e25.STEPS,
        "samples_seen": p3e25.EXPOSURE,
        "training_images": p3e25.TRAIN_IMAGES,
        "trainable_parameter_count": audit["trainable_parameters"],
        "metrics": final,
        "lineage": audit["lineage"],
        "input": input_audit,
        "selection": selection_audit,
    }
    checkpoint = OUTPUT / f"step-{p3e25.STEPS:06d}.pt"
    _atomic_torch_save(checkpoint, payload)
    final["checkpoint"] = str(checkpoint.resolve())
    final["checkpoint_sha256"] = file_sha256(checkpoint)
    audit["status"] = "completed"
    audit["decision"] = decision
    audit["checkpoint_sha256"] = final["checkpoint_sha256"]
    _atomic_json(OUTPUT / "audit.json", audit)
    _atomic_json(OUTPUT / "final_metrics.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
