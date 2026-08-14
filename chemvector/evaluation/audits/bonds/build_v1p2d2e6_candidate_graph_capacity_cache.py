#!/usr/bin/env python3
"""Build an exact-reachable capacity64 cache for the shared-endpoint graph model."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.components.bonds.objects.semantic_bond_endpoint import (
    SemanticBondEndpointConfig,
    SemanticBondEndpointDetector,
    decode_semantic_endpoints,
)
from chemvector.components.bonds.objects.semantic_endpoint_candidate_graph import (
    candidate_graph_node_features,
)
from chemvector.components.bonds.objects.semantic_endpoint_pair import (
    SemanticEndpointPairConfig,
    score_endpoint_pairs,
    select_local_partner_pairs,
)
from chemvector.components.bonds.objects.semantic_endpoint_pair_scorer import (
    extract_endpoint_pair_features,
)
from chemvector.components.vision.semantic_ink_separator import (
    SemanticInkSeparator,
    SemanticInkSeparatorConfig,
)
from chemvector.evaluation.audits.bonds.audit_v1p2c_endpoint_pair_oracle import (
    ENDPOINT_THRESHOLD,
    IMAGE_SIZE,
    MATCH_RADIUS_PIXELS,
    MAXIMUM_ENDPOINTS,
)
from chemvector.evaluation.audits.bonds.audit_v1p2d2e5_slots4_local_partner_oracle import (
    ENDPOINT_CHECKPOINT,
    ENDPOINT_CHECKPOINT_SHA256,
    ENDPOINT_RESULT,
    ENDPOINT_RESULT_SHA256,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.bonds.objects.train_v1p2_semantic_bond_object import (
    DEFAULT_SEPARATOR,
    SEPARATOR_SHA256,
    _bond_probability,
)
from chemvector.training.bonds.objects.train_v1p2b_semantic_bond_endpoint import (
    _unique_port_data,
)
from chemvector.training.bonds.objects.train_v1p2d2e3_style_endpoint_field import (
    MANIFEST_SHA256,
    _style_loader,
    load_style_manifest,
    split_rows,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)


AUDIT_NAME = "chemvector-v1p2d2e6-candidate-graph-capacity64-cache"
DEFAULT_OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
DEFAULT_CACHE = DEFAULT_OUTPUT / "capacity64.pt"
PARTNERS_PER_ENDPOINT = 8
CAPACITY_SAMPLES = 64
VARIANT_QUOTA = 16
SCAN_ROWS = 1_024
SEED = 20260825


def canonical_candidate_graph_targets(
    points: torch.Tensor,
    pairs: torch.Tensor,
    expected_bonds: torch.Tensor,
    *,
    match_radius_pixels: float = MATCH_RADIUS_PIXELS,
    image_size: float = IMAGE_SIZE,
) -> dict[str, torch.Tensor | int]:
    """Match nodes as a set, then label edges by their matched GT port pair."""

    if points.ndim != 2 or points.size(-1) != 2:
        raise ValueError("Candidate target points must have shape [nodes, 2]")
    if pairs.ndim != 2 or pairs.size(-1) != 2:
        raise ValueError("Candidate target pairs must have shape [edges, 2]")
    if expected_bonds.ndim != 3 or expected_bonds.shape[-2:] != (2, 2):
        raise ValueError("Candidate target Bonds must have shape [bonds, 2, 2]")
    if min(match_radius_pixels, image_size) <= 0.0:
        raise ValueError("Candidate target matching scales must be positive")
    if len(pairs) and (
        int(pairs.min()) < 0 or int(pairs.max()) >= len(points)
    ):
        raise ValueError("Candidate target pair index is outside the node set")

    points_cpu = points.detach().float().cpu()
    pairs_cpu = pairs.detach().long().cpu()
    expected_cpu = expected_bonds.detach().float().cpu()
    target_points, port_indices = _unique_port_data(expected_cpu)
    node_port_ids = torch.full((len(points_cpu),), -1, dtype=torch.long)
    if len(points_cpu) and len(target_points):
        distances = torch.cdist(points_cpu, target_points.float()) * image_size
        rows, columns = linear_sum_assignment(distances.numpy())
        for row, column in zip(rows, columns):
            if float(distances[row, column]) <= match_radius_pixels:
                node_port_ids[int(row)] = int(column)

    target_degrees = torch.bincount(
        port_indices.reshape(-1), minlength=len(target_points)
    )
    node_target_degrees = torch.zeros((len(points_cpu),), dtype=torch.long)
    matched_nodes = torch.where(node_port_ids >= 0)[0]
    if len(matched_nodes):
        node_target_degrees[matched_nodes] = target_degrees[
            node_port_ids[matched_nodes]
        ]

    bond_by_ports: dict[tuple[int, int], int] = {}
    for bond_id, ports in enumerate(port_indices.tolist()):
        key = tuple(sorted((int(ports[0]), int(ports[1]))))
        if key in bond_by_ports:
            raise ValueError("Candidate target contains duplicate GT port pairs")
        bond_by_ports[key] = bond_id
    edge_target_bond_ids = torch.full((len(pairs_cpu),), -1, dtype=torch.long)
    for edge_index, pair in enumerate(pairs_cpu.tolist()):
        begin = int(node_port_ids[int(pair[0])])
        end = int(node_port_ids[int(pair[1])])
        if min(begin, end) < 0 or begin == end:
            continue
        edge_target_bond_ids[edge_index] = bond_by_ports.get(
            tuple(sorted((begin, end))), -1
        )
    reachable = len(set(edge_target_bond_ids[edge_target_bond_ids >= 0].tolist()))
    return {
        "node_port_ids": node_port_ids,
        "node_target_degrees": node_target_degrees,
        "edge_target_bond_ids": edge_target_bond_ids,
        "unique_port_count": len(target_points),
        "reachable_bond_count": reachable,
    }


@torch.no_grad()
def build_candidate_graph(
    probability: torch.Tensor,
    decoded: list[dict[str, torch.Tensor | float]],
    expected_bonds: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    points = (
        torch.stack([item["point"] for item in decoded]).to(probability.device)
        if decoded
        else probability.new_empty((0, 2))
    )
    point_scores = probability.new_tensor(
        [float(item["score"]) for item in decoded]
    )
    pair_config = SemanticEndpointPairConfig()
    evidence = score_endpoint_pairs(
        probability, points, point_scores, pair_config
    )
    selected = select_local_partner_pairs(
        evidence["pairs"],
        evidence["line_score"],
        point_count=len(points),
        partners_per_point=PARTNERS_PER_ENDPOINT,
    )
    pairs = evidence["pairs"][selected]
    edge_features = extract_endpoint_pair_features(
        probability, points, point_scores, pairs, pair_config
    )
    node_features = candidate_graph_node_features(points, point_scores, pairs)
    targets = canonical_candidate_graph_targets(points, pairs, expected_bonds)
    return {
        "points": points.float().cpu(),
        "point_scores": point_scores.float().cpu(),
        "pairs": pairs.long().cpu(),
        "node_features": node_features.float().cpu(),
        "edge_features": edge_features.float().cpu(),
        "node_port_ids": targets["node_port_ids"],
        "node_target_degrees": targets["node_target_degrees"],
        "edge_target_bond_ids": targets["edge_target_bond_ids"],
        "unique_port_count": int(targets["unique_port_count"]),
        "reachable_bond_count": int(targets["reachable_bond_count"]),
        "bond_count": len(expected_bonds),
    }


def _verify_frozen_inputs() -> None:
    if file_sha256(DEFAULT_SEPARATOR) != SEPARATOR_SHA256:
        raise ValueError("P2-D2-E6 separator SHA256 differs")
    if file_sha256(ENDPOINT_CHECKPOINT) != ENDPOINT_CHECKPOINT_SHA256:
        raise ValueError("P2-D2-E6 endpoint checkpoint SHA256 differs")
    if file_sha256(ENDPOINT_RESULT) != ENDPOINT_RESULT_SHA256:
        raise ValueError("P2-D2-E6 endpoint result SHA256 differs")
    result = json.loads(ENDPOINT_RESULT.read_text(encoding="utf-8"))
    if result.get("decision") != "slots4_style_endpoint_go":
        raise ValueError("P2-D2-E6 requires the frozen E4 GO checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scan-rows", type=int, default=SCAN_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-every", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2-D2-E6 capacity cache requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-D2-E6 output is not empty: {args.output_dir}")
    if args.scan_rows < CAPACITY_SAMPLES:
        raise ValueError("P2-D2-E6 scan budget is smaller than capacity64")
    _verify_frozen_inputs()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    fit_rows = split_rows(load_style_manifest(), "fit")
    scan_rows = fit_rows[: args.scan_rows]
    device = torch.device("cuda")
    separator_payload = torch.load(DEFAULT_SEPARATOR, map_location=device)
    separator = SemanticInkSeparator(
        SemanticInkSeparatorConfig(**separator_payload["config"])
    ).to(device)
    separator.load_state_dict(separator_payload["model"], strict=True)
    separator.eval().requires_grad_(False)
    endpoint_payload = torch.load(ENDPOINT_CHECKPOINT, map_location=device)
    detector = SemanticBondEndpointDetector(
        SemanticBondEndpointConfig(**endpoint_payload["config"])
    ).to(device)
    detector.load_state_dict(endpoint_payload["detector"], strict=True)
    detector.eval().requires_grad_(False)

    started = time.perf_counter()
    graphs: list[dict[str, Any]] = []
    variant_counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    scanned = 0
    loader = _style_loader(
        scan_rows,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        workers=args.workers,
    )
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        probability = _bond_probability(separator, pixels)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            fields = detector(probability)
        decoded_batch = decode_semantic_endpoints(
            fields,
            threshold=ENDPOINT_THRESHOLD,
            maximum_points=MAXIMUM_ENDPOINTS,
        )
        for index, decoded in enumerate(decoded_batch):
            scanned += 1
            variant = str(batch["variants"][index])
            if variant_counts[variant] >= VARIANT_QUOTA:
                rejected["variant_quota_full"] += 1
                continue
            bond_count = int(batch["mask"][index].sum())
            expected_bonds = batch["endpoints"][index, :bond_count]
            graph = build_candidate_graph(
                probability[index : index + 1], decoded, expected_bonds
            )
            if graph["reachable_bond_count"] != bond_count:
                rejected["canonical_oracle_not_exact"] += 1
                continue
            graph.update(
                {
                    "sample_key": str(batch["keys"][index]),
                    "variant": variant,
                    "stress_operation": batch["stress_operations"][index],
                }
            )
            graphs.append(graph)
            variant_counts[variant] += 1
            if len(graphs) % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "scanned": scanned,
                            "selected": len(graphs),
                            "variants": dict(sorted(variant_counts.items())),
                        }
                    ),
                    flush=True,
                )
            if len(graphs) == CAPACITY_SAMPLES:
                break
        if len(graphs) == CAPACITY_SAMPLES:
            break
    expected_variants = {name: VARIANT_QUOTA for name in ("coordgen", "original", "rdkit", "stress")}
    if len(graphs) != CAPACITY_SAMPLES or dict(variant_counts) != expected_variants:
        raise RuntimeError(
            f"P2-D2-E6 could not fill balanced capacity64: {dict(variant_counts)}"
        )
    maximum_degree = max(
        int(graph["node_target_degrees"].max()) for graph in graphs
    )
    if maximum_degree > 8:
        raise ValueError(f"P2-D2-E6 target degree exceeds model capacity: {maximum_degree}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    cache_path = args.output_dir / DEFAULT_CACHE.name
    cache = {
        "schema_version": "chemvector-v1p2d2e6-candidate-graph-cache-0.1",
        "audit_name": AUDIT_NAME,
        "seed": args.seed,
        "separator_sha256": SEPARATOR_SHA256,
        "endpoint_checkpoint_sha256": ENDPOINT_CHECKPOINT_SHA256,
        "endpoint_result_sha256": ENDPOINT_RESULT_SHA256,
        "style_manifest_sha256": MANIFEST_SHA256,
        "endpoint_threshold": ENDPOINT_THRESHOLD,
        "maximum_endpoints": MAXIMUM_ENDPOINTS,
        "partners_per_endpoint": PARTNERS_PER_ENDPOINT,
        "match_radius_pixels": MATCH_RADIUS_PIXELS,
        "graphs": graphs,
    }
    _atomic_torch_save(cache_path, cache)
    sample_keys = [str(graph["sample_key"]) for graph in graphs]
    audit = {
        "schema_version": "chemvector-v1p2d2e6-candidate-graph-cache-audit-0.1",
        "audit_name": AUDIT_NAME,
        "status": "passed",
        "samples": len(graphs),
        "scanned_samples": scanned,
        "variants": dict(sorted(variant_counts.items())),
        "rejected": dict(sorted(rejected.items())),
        "bonds": sum(int(graph["bond_count"]) for graph in graphs),
        "unique_ports": sum(int(graph["unique_port_count"]) for graph in graphs),
        "candidate_nodes": sum(len(graph["node_features"]) for graph in graphs),
        "candidate_edges": sum(len(graph["edge_features"]) for graph in graphs),
        "canonical_oracle_image_exact": 1.0,
        "maximum_target_degree": maximum_degree,
        "sample_keys_sha256": hashlib.sha256(
            "".join(f"{key}\n" for key in sample_keys).encode()
        ).hexdigest(),
        "cache": str(cache_path),
        "cache_sha256": file_sha256(cache_path),
        "generated_png_files": 0,
        "style_calibration_samples_read": 0,
        "style_holdout_samples_read": 0,
        "fresh_calibration_samples_read": 0,
        "fresh_holdout_samples_read": 0,
        "sealed_test_samples_read": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
