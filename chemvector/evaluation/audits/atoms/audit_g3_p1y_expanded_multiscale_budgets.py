#!/usr/bin/env python3
"""Audit expanded frozen multiscale peak budgets on train5k and route100."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride1AtomCenterField,
    ContinuousStride2AtomCenterField,
)
from chemvector.evaluation.audits.atoms import audit_g3_p1w_multiscale_peak_rank as p1w
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1u_adjacent_atom_capacity as p1u
from chemvector.training.atoms import train_g3_p1v_continuous_stride1_capacity as p1v
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


DEFAULT_OUTPUT = Path("outputs/audits/chemvector-g3-p1y-expanded-multiscale-budgets")
COARSE_BUDGETS = (
    (128, 0),
    (96, 32),
    (0, 128),
    (128, 64),
    (0, 192),
    (192, 64),
    (128, 128),
    (0, 256),
    (256, 128),
    (192, 192),
    (0, 384),
    (384, 128),
    (256, 256),
    (0, 512),
    (512, 256),
    (384, 384),
    (0, 768),
    (768, 256),
    (512, 512),
    (0, 1024),
    (1024, 512),
    (768, 768),
    (1024, 1024),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P1-Y expanded audit output is not empty: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("P1-Y expanded budget audit requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(20260713)
    np.random.seed(20260713)
    torch.manual_seed(20260713)
    torch.cuda.manual_seed_all(20260713)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    train, train_audit = p1a.load_train_manifest()
    route, route_audit = p1a.load_route_manifest()
    domains = {"train5k": train, "route100": route}
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    stride2, stride2_lineage = p1w.collect_candidates(
        checkpoint=p1w.P1U_CHECKPOINT,
        checkpoint_sha256=p1w.P1U_CHECKPOINT_SHA256,
        expected_model_type=p1u.MODEL_TYPE,
        field_factory=ContinuousStride2AtomCenterField,
        domains=domains,
        collator=collator,
        device=device,
    )
    stride1, stride1_lineage = p1w.collect_candidates(
        checkpoint=p1w.P1V_CHECKPOINT,
        checkpoint_sha256=p1w.P1V_CHECKPOINT_SHA256,
        expected_model_type=p1v.MODEL_TYPE,
        field_factory=ContinuousStride1AtomCenterField,
        domains=domains,
        collator=collator,
        device=device,
    )

    results = []
    for stride2_k, stride1_k in COARSE_BUDGETS:
        current = {"stride2_k": stride2_k, "stride1_k": stride1_k, "domains": {}}
        for domain in domains:
            metrics = p1w.evaluate_budget(
                stride2[domain], stride1[domain],
                stride2_k=stride2_k, stride1_k=stride1_k,
            )
            current["domains"][domain] = metrics
        current["all_domains_exact"] = all(
            value["exact_images"] == len(domains[domain])
            for domain, value in current["domains"].items()
        )
        results.append(current)
        print(json.dumps(current, separators=(",", ":")), flush=True)

    maximum = results[-1]
    union_reachable = all(
        value["matched_atoms"] == value["target_atoms"]
        for value in maximum["domains"].values()
    )
    first_exact = next(
        (value for value in results if value["all_domains_exact"]), None
    )
    decision = (
        "expanded_fixed_budget_go"
        if first_exact is not None
        else "rank_budget_no_go_representation_reachable"
        if union_reachable
        else "top1024_per_scale_no_go_full_peak_reachability_unresolved"
    )
    audit = {
        "schema_version": "g3-p1y-expanded-multiscale-budget-0.1",
        "decision": decision,
        "maximum_peaks_per_scale": p1w.MAXIMUM_PEAKS,
        "coarse_budget_order": [
            {"stride2_k": left, "stride1_k": right}
            for left, right in COARSE_BUDGETS
        ],
        "first_all_domain_exact_budget": first_exact,
        "maximum_union_all_atoms_reachable": union_reachable,
        "results": results,
        "manifests": {"train5k": train_audit, "route100": route_audit},
        "lineage": {"stride2": stride2_lineage, "stride1": stride1_lineage},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps({
        "decision": decision,
        "first_all_domain_exact_budget": first_exact,
        "maximum_union": maximum,
    }, indent=2))


if __name__ == "__main__":
    main()
