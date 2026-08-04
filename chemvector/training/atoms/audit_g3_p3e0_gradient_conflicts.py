#!/usr/bin/env python3
"""Zero-update gradient norm, cosine, and clipping audit for P3-E0."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_query_set import (
    g3_atom_query_set_loss,
)
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import (
    train_g3_p3c_fixed_assignment_replay_train4937 as p3c,
)
from chemvector.training.atoms import (
    train_g3_p3d_structured_matching_replay_train4937 as p3d,
)
from chemvector.training.atoms import audit_g3_p3e0_latent_solver as p3e0
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e0-gradient-conflicts"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
BATCH_SIZE = 4
CLIP_NORM = 1.0


def flatten_gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool = True,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    return torch.cat([
        (
            torch.zeros_like(parameter, memory_format=torch.preserve_format)
            if gradient is None else gradient
        ).detach().float().flatten()
        for parameter, gradient in zip(parameters, gradients)
    ])


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(left.norm() * right.norm())
    if denominator == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def shared_parameters(model) -> list[torch.nn.Parameter]:
    excluded = ("base_model.count_head.", "existence_head.")
    return [
        value for name, value in model.named_parameters()
        if not name.startswith(excluded)
    ]


def audit_model(model, records_by_domain, assignments, *, device) -> dict:
    model.eval()
    all_parameters = list(model.parameters())
    shared = shared_parameters(model)
    losses_to_norms = defaultdict(list)
    cosines = defaultdict(list)
    clip_norms = defaultdict(list)
    domain_sums = {
        domain: defaultdict(lambda: torch.zeros(
            sum(value.numel() for value in shared), device=device
        ))
        for domain in records_by_domain
    }
    domain_batches = defaultdict(int)

    for domain, records in records_by_domain.items():
        for start in range(0, len(records), BATCH_SIZE):
            current = records[start : start + BATCH_SIZE]
            batch = p2h.collate(current, device)
            with _autocast_context(device, "bf16"):
                output = model(p2h.model_candidates(batch))
                dynamic = g3_atom_query_set_loss(
                    output,
                    batch["candidate_anchors"],
                    batch["atom_anchors"],
                    batch["full_atom_mask"],
                    unique_pointers=True,
                    query_matching="pointer",
                )
                fixed = [assignments[str(record["sample_key"])] for record in current]
                structured = p3d.structured_matching_loss(
                    output["existence_logits"], output["pointer_logits"], fixed
                )["structured_matching_loss"]
            effective_losses = {
                "object": dynamic["object_loss"],
                "pointer": dynamic["pointer_loss"],
                "coordinate_x5": 5.0 * dynamic["coordinate_loss"],
                "boundary": dynamic["existence_boundary_loss"],
                "count": dynamic["count_loss"],
                "structured_fixed": structured,
            }
            shared_gradients = {
                name: flatten_gradients(value, shared)
                for name, value in effective_losses.items()
            }
            for name, gradient in shared_gradients.items():
                losses_to_norms[name].append(float(gradient.norm()))
                domain_sums[domain][name] += gradient
            for structure_name in (
                "object", "pointer", "coordinate_x5", "boundary", "structured_fixed"
            ):
                cosines[f"count_vs_{structure_name}"].append(cosine(
                    shared_gradients["count"], shared_gradients[structure_name]
                ))
            cosines["pointer_vs_coordinate_x5"].append(cosine(
                shared_gradients["pointer"], shared_gradients["coordinate_x5"]
            ))
            cosines["object_vs_boundary"].append(cosine(
                shared_gradients["object"], shared_gradients["boundary"]
            ))

            with _autocast_context(device, "bf16"):
                p3b_total = (
                    dynamic["object_loss"]
                    + dynamic["pointer_loss"]
                    + 5.0 * dynamic["coordinate_loss"]
                    + dynamic["count_loss"]
                )
                struct_boundary_total = (
                    dynamic["object_loss"]
                    + dynamic["pointer_loss"]
                    + 5.0 * dynamic["coordinate_loss"]
                    + dynamic["existence_boundary_loss"]
                )
                p3d_total = structured + dynamic["count_loss"]
            for name, value in {
                "p3b_joint": p3b_total,
                "dynamic_structure_boundary1": struct_boundary_total,
                "p3d_structured_plus_count": p3d_total,
            }.items():
                gradient = flatten_gradients(value, all_parameters)
                clip_norms[name].append(float(gradient.norm()))
            domain_batches[domain] += 1

    cross_domain = {}
    for name in losses_to_norms:
        capacity = (
            domain_sums["capacity_easy"][name] / domain_batches["capacity_easy"]
            + domain_sums["capacity_hard"][name] / domain_batches["capacity_hard"]
        ) / 2.0
        general = domain_sums["general192"][name] / domain_batches["general192"]
        cross_domain[name] = {
            "capacity_vs_general_cosine": cosine(capacity, general),
            "capacity_mean_norm": float(capacity.norm()),
            "general_mean_norm": float(general.norm()),
        }
    return {
        "shared_parameter_count": sum(value.numel() for value in shared),
        "loss_shared_gradient_norms": {
            name: summarize(values) for name, values in losses_to_norms.items()
        },
        "shared_gradient_cosines": {
            name: summarize(values) for name, values in cosines.items()
        },
        "cross_domain_mean_gradients": cross_domain,
        "global_objective_clipping": {
            name: {
                "preclip_norm": summarize(values),
                "fraction_clipped": sum(value > CLIP_NORM for value in values) / len(values),
                "mean_scale": float(np.mean([
                    min(1.0, CLIP_NORM / max(value, 1e-12)) for value in values
                ])),
            }
            for name, values in clip_norms.items()
        },
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E0 gradient output is not empty: {OUTPUT}")
    if file_sha256(p3c.OUTPUT / "fixed_assignment_cache.json") != (
        p3d.ASSIGNMENT_CACHE_FILE_SHA256
    ):
        raise ValueError("P3-E0 gradient fixed assignment cache SHA256 differs")
    assignment_payload = json.loads(
        (p3c.OUTPUT / "fixed_assignment_cache.json").read_text(encoding="utf-8")
    )
    assignments = p3c.validate_assignment_cache(assignment_payload)
    torch.manual_seed(20260713)
    torch.cuda.manual_seed_all(20260713)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    cache, cache_audit = p2e.load_cache()
    eligible = [
        record for record in cache["train5k"]
        if torch.equal(record["atom_mask"], record["full_atom_mask"])
    ]
    domains, selection_audit = p2h.select_capacity_domains(cache["train5k"])
    capacity_keys = {
        str(record["sample_key"])
        for records in domains.values() for record in records
    }
    general = sorted(
        (
            record for record in eligible
            if str(record["sample_key"]) not in capacity_keys
        ),
        key=lambda record: str(record["sample_key"]),
    )[:p3e0.GENERAL_IMAGES]
    records_by_domain = {
        "capacity_easy": domains["easy32"],
        "capacity_hard": domains["hard32"],
        "general192": general,
    }

    results = {}
    state_checks = {}
    for name, (path, expected_sha) in p3e0.CHECKPOINTS.items():
        if file_sha256(path) != expected_sha:
            raise ValueError(f"P3-E0 gradient {name} checkpoint SHA256 differs")
        model = p3e0.load_model(path, device)
        initial_state = module_state_sha256(model)
        results[name] = audit_model(
            model, records_by_domain, assignments, device=device
        )
        state_checks[name] = module_state_sha256(model) == initial_state

    finite = all(
        math.isfinite(summary["mean"])
        for result in results.values()
        for summary in result["loss_shared_gradient_norms"].values()
    )
    checks = {
        "all_model_states_unchanged": all(state_checks.values()),
        "all_gradient_summaries_finite": finite,
        "training_updates": 0,
    }
    payload = {
        "schema_version": "g3-p3e0-gradient-conflicts-audit-0.1",
        "audit_name": AUDIT_NAME,
        "decision": (
            "gradient_conflict_audit_complete"
            if checks["all_model_states_unchanged"] and finite
            else "gradient_conflict_audit_failed"
        ),
        "checks": checks,
        "scope": {
            "capacity_easy": 32,
            "capacity_hard": 32,
            "general": len(general),
            "batch_size": BATCH_SIZE,
            "clip_norm": CLIP_NORM,
        },
        "results": results,
        "lineage": {
            "checkpoints": {
                name: {"path": str(path.resolve()), "sha256": expected}
                for name, (path, expected) in p3e0.CHECKPOINTS.items()
            },
            "assignment_cache_file_sha256": p3d.ASSIGNMENT_CACHE_FILE_SHA256,
            "assignment_entries_sha256": p3d.ASSIGNMENT_ENTRIES_SHA256,
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
            "capacity_selection": selection_audit,
            "general_sample_keys": [str(record["sample_key"]) for record in general],
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "metrics.json", payload)
    print(json.dumps({
        "decision": payload["decision"],
        "checks": checks,
        "p2w": results["p2w"],
    }, indent=2))


if __name__ == "__main__":
    main()
