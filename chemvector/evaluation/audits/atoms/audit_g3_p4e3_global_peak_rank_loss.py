#!/usr/bin/env python3
"""Zero-step gradient and cost audit for the P4-E3 global peak-rank loss."""

from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    atom_center_global_peak_rank_loss,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


OUTPUT = Path("outputs/audits/chemvector-g3-p4e3-global-peak-rank-loss")
BATCH_SIZE = 4
IMAGE_SIZE = 768
CANDIDATE_BUDGET = 128
BOUNDARY_WINDOW = 16
MARGIN = 1.0


def _logical_checks(device: torch.device) -> dict[str, bool]:
    logits = torch.full((1, 1, 5, 7), -10.0, device=device, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, 2, 1] = 0.0
        logits[0, 0, 1, 3] = 5.0
        logits[0, 0, 3, 3] = 4.0
        logits[0, 0, 2, 5] = 3.0
    positive = torch.zeros_like(logits, dtype=torch.bool)
    positive[0, 0, 2, 1] = True
    loss = atom_center_global_peak_rank_loss(
        logits,
        positive,
        candidate_budget=2,
        boundary_window=1,
        margin=MARGIN,
    )
    loss.backward()

    adjacent = torch.full((1, 1, 5, 7), -10.0, device=device)
    adjacent[0, 0, 2, 1] = 2.0
    adjacent[0, 0, 2, 2] = 20.0
    adjacent[0, 0, 2, 5] = 1.0
    adjacent_positive = torch.zeros_like(adjacent, dtype=torch.bool)
    adjacent_positive[0, 0, 2, 1:3] = True
    adjacent_loss = atom_center_global_peak_rank_loss(
        adjacent,
        adjacent_positive,
        candidate_budget=2,
        boundary_window=1,
        margin=MARGIN,
    )
    adjacent_expected = (
        torch.nn.functional.softplus(adjacent.new_tensor(0.0))
        + torch.nn.functional.softplus(adjacent.new_tensor(-18.0))
    ) / 2.0

    satisfied = torch.full((1, 1, 5, 7), -10.0, device=device)
    satisfied[0, 0, 2, 1] = 20.0
    satisfied[0, 0, 1, 3] = 2.0
    satisfied[0, 0, 3, 3] = 1.0
    satisfied_positive = torch.zeros_like(satisfied, dtype=torch.bool)
    satisfied_positive[0, 0, 2, 1] = True
    satisfied_loss = atom_center_global_peak_rank_loss(
        satisfied,
        satisfied_positive,
        candidate_budget=2,
        boundary_window=1,
        margin=MARGIN,
    )
    return {
        "below_boundary_loss_nonzero": float(loss.detach()) > 1.0,
        "positive_gradient_negative": float(logits.grad[0, 0, 2, 1]) < 0.0,
        "hard_negative_gradient_positive": float(logits.grad[0, 0, 3, 3]) > 0.0,
        "unselected_higher_peak_has_no_gradient": float(logits.grad[0, 0, 1, 3]) == 0.0,
        "margin_satisfied_loss_near_zero": float(satisfied_loss) < 1e-7,
        "adjacent_gt_not_used_as_negative": bool(
            torch.isclose(adjacent_loss, adjacent_expected, atol=1e-7, rtol=1e-7)
        ),
    }


def _profile(device: torch.device) -> dict[str, float | int | bool | str]:
    generator = torch.Generator(device=device).manual_seed(20260843)
    base = torch.randn(
        BATCH_SIZE,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    positive = torch.zeros_like(base, dtype=torch.bool)
    for image_index in range(BATCH_SIZE):
        indices = torch.randperm(
            IMAGE_SIZE * IMAGE_SIZE,
            device=device,
            generator=generator,
        )[:64]
        positive[image_index, 0].flatten()[indices] = True

    def iteration() -> tuple[torch.Tensor, torch.Tensor]:
        logits = base.detach().clone().requires_grad_(True)
        loss = atom_center_global_peak_rank_loss(
            logits,
            positive,
            candidate_budget=CANDIDATE_BUDGET,
            boundary_window=BOUNDARY_WINDOW,
            margin=MARGIN,
        )
        loss.backward()
        return loss, logits.grad

    iteration()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    loss, gradient = iteration()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    positive_gradients = gradient[positive]
    negative_gradients = gradient[~positive]
    return {
        "device": torch.cuda.get_device_name(device),
        "dtype": "bfloat16",
        "batch_size": BATCH_SIZE,
        "height": IMAGE_SIZE,
        "width": IMAGE_SIZE,
        "loss": float(loss.detach()),
        "forward_backward_milliseconds": elapsed * 1_000.0,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "all_positive_cells_receive_gradient": bool((positive_gradients < 0).all()),
        "hard_negative_cells_receive_gradient": bool((negative_gradients > 0).any()),
        "finite_gradients": bool(torch.isfinite(gradient).all()),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("P4-E3 zero-step audit requires CUDA")
    device = torch.device("cuda")
    logical = _logical_checks(device)
    profile = _profile(device)
    checks = {
        **logical,
        "production_shape_positive_gradients": profile[
            "all_positive_cells_receive_gradient"
        ],
        "production_shape_negative_gradients": profile[
            "hard_negative_cells_receive_gradient"
        ],
        "production_shape_finite_gradients": profile["finite_gradients"],
        "peak_cuda_memory_below_2gib": profile["peak_cuda_bytes"] < 2 * 1024**3,
        "forward_backward_below_250ms": profile[
            "forward_backward_milliseconds"
        ] < 250.0,
    }
    audit = {
        "schema_version": "g3-p4e3-global-peak-rank-loss-audit-0.1",
        "status": "passed" if all(checks.values()) else "failed",
        "protocol": {
            "candidate_budget": CANDIDATE_BUDGET,
            "boundary_window": BOUNDARY_WINDOW,
            "margin": MARGIN,
            "regional_maxima_selection": "detached-3x3-local-maxima",
            "selected_logits_keep_gradient": True,
            "pixel_cluster_radius": None,
            "data_read": False,
            "confirm_v2_read": False,
            "sealed_test_read": False,
        },
        "checks": checks,
        "profile": profile,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
