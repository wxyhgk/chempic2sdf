#!/usr/bin/env python3
"""Build v0.10.13 route-select actions and audit the single-delete oracle."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import islice
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import time

import torch

from Data.sdf.chemsvg_sequence_v4 import (
    ChemSVGCoordinateCollisionError,
    serialize_chemsvg_v4,
)
from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.data import ChemVectorIterableDataset
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.direct_scene_inference import load_direct_model
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.safe_atom_suppression import (
    apply_oracle_single_delete,
    evaluate_kept_graph,
    label_single_delete_actions,
)
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_direct_atom_pair_bond import frozen_atom_objects


DATA_ROOT = Path("Data/sdf/chemvector_safe_suppression_v0_1")
DATA_COMMITMENT = "a0dcb1c2d57bea11198d874ca60f3e229705d1e3ed0b369da41fde8c02cf7542"
DEFAULT_DIRECT = "outputs/checkpoints/chemvector-v0103-direct-atom-pair-style10k/selected.pt"
DEFAULT_OUTPUT = "outputs/audits/chemvector-v01013-safe-single-delete-oracle/report.json"
DEFAULT_CACHE = "outputs/caches/chemvector-v01013-route-select-actions/base-actions.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-checkpoint", default=DEFAULT_DIRECT)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-output", default=DEFAULT_CACHE)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data_gate() -> dict:
    validation = json.loads((DATA_ROOT / "validation.json").read_text(encoding="utf-8"))
    commitment = json.loads((DATA_ROOT / "commitment.json").read_text(encoding="utf-8"))
    embedded = commitment.pop("commitment_sha256", None)
    recomputed = hashlib.sha256(_canonical_json_bytes(commitment)).hexdigest()
    if validation.get("status") != "passed":
        raise ValueError("Safe-suppression data validation is not passed")
    if embedded != DATA_COMMITMENT or recomputed != DATA_COMMITMENT:
        raise ValueError(
            f"Safe-suppression commitment differs: embedded={embedded} recomputed={recomputed}"
        )
    if validation.get("commitment_sha256") != DATA_COMMITMENT:
        raise ValueError("Validation references a different data commitment")
    confirm = DATA_ROOT / "route_confirm"
    confirm_mode = stat.S_IMODE(confirm.stat().st_mode)
    if confirm_mode != 0:
        raise ValueError(f"route_confirm must remain mode 000, found {confirm_mode:o}")
    shards = sorted((DATA_ROOT / "route_select/shards").glob("route_select-*.tar"))
    if len(shards) != 2:
        raise ValueError(f"Expected two route-select shards, found {len(shards)}")
    return {
        "status": "passed",
        "commitment_sha256": DATA_COMMITMENT,
        "route_select_shards": [str(path) for path in shards],
        "route_confirm_mode": f"{confirm_mode:03o}",
        "route_confirm_read": False,
    }


def _batches(iterator, batch_size: int):
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def _minimum_target_distance(target_anchors: torch.Tensor) -> float:
    if len(target_anchors) < 2:
        return float("inf")
    distance = torch.cdist(target_anchors.double(), target_anchors.double()) * 768.0
    distance.fill_diagonal_(float("inf"))
    return float(distance.min())


def _compact_labels(labels: dict) -> dict:
    return {
        "baseline_utility": (
            int(labels["baseline_primary"]["skeleton_exact"]),
            int(labels["baseline_primary"]["atom_exact"]),
            -(
                int(labels["baseline_primary"]["atom_fp"])
                + int(labels["baseline_primary"]["atom_fn"])
            ),
            -(
                int(labels["baseline_primary"]["bond_fp"])
                + int(labels["baseline_primary"]["bond_fn"])
            ),
        ),
        "actions": [
            {
                "removed": int(action["removed"]),
                "safe": bool(action["safe"]),
                "strict_gain": bool(action["strict_gain"]),
                "utility": tuple(int(value) for value in action["utility"]),
            }
            for action in labels["actions"]
        ],
        "winner_removed_nodes": [
            int(value) for value in labels["winner_removed_nodes"]
        ],
    }


@torch.no_grad()
def collect_samples(
    direct,
    vision,
    atom,
    bond,
    set_head,
    records,
    collator,
    config,
    atom_threshold,
    presence_threshold,
    device,
) -> list[dict]:
    batch = collator(records)
    pixels = batch["pixel_values"].to(device)
    targets = stack_targets(batch, device)
    features, atoms = frozen_atom_objects(
        vision, atom, bond, set_head, pixels, config, atom_threshold
    )
    pairs, pair_samples = enumerate_unordered_atom_pairs(atoms["atom_mask"])
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = direct(
            features,
            atoms["atom_features"],
            atoms["atom_geometry"][..., :2],
            atoms["atom_scores"],
            pairs,
            pair_samples,
        )
    probabilities = output["presence_logits"].sigmoid().float().cpu()
    chemical = output["chemical_logits"].float().argmax(-1).cpu()
    samples = []
    for batch_index, record in enumerate(records):
        proposal_indices = torch.where(atoms["atom_mask"][batch_index])[0]
        remap = {
            int(value): index for index, value in enumerate(proposal_indices.tolist())
        }
        pair_rows = torch.where(pair_samples == batch_index)[0]
        atom_count = int(targets["atom_mask"][batch_index].sum())
        bond_count = int(targets["bond_mask"][batch_index].sum())
        target_bonds = {
            tuple(sorted(int(value) for value in pair)): int(targets["bond_chemical"][batch_index, row])
            for row, pair in enumerate(
                targets["bond_atom_refs"][batch_index, :bond_count].cpu().tolist()
            )
        }
        sample = {
            "key": record["key"],
            "molecule_id": record["chemmol"]["molecule_id"],
            "variant": record["render"]["variant"],
            "anchors": atoms["atom_geometry"][batch_index, proposal_indices, :2]
            .float()
            .cpu(),
            "atom_features": atoms["atom_features"][batch_index, proposal_indices]
            .float()
            .cpu(),
            "atom_geometry": atoms["atom_geometry"][batch_index, proposal_indices]
            .float()
            .cpu(),
            "scores": atoms["atom_scores"][batch_index, proposal_indices].float().cpu(),
            "pairs": torch.tensor(
                [
                    [remap[int(left)], remap[int(right)]]
                    for left, right in pairs[pair_rows].tolist()
                ],
                dtype=torch.long,
            ),
            "probabilities": probabilities[pair_rows.cpu()],
            "chemical": chemical[pair_rows.cpu()],
            "target_anchors": targets["atom_anchors"][batch_index, :atom_count]
            .float()
            .cpu(),
            "target_bonds": target_bonds,
        }
        labels = label_single_delete_actions(
            sample, presence_threshold=presence_threshold
        )
        sample["action_labels"] = _compact_labels(labels)
        rendered_anchors = torch.tensor(
            [
                [float(item["image_x"]), float(item["image_y"])]
                for item in record["render"]["atoms"]
            ],
            dtype=torch.float64,
        ) / 768.0
        sample["minimum_target_distance_px"] = _minimum_target_distance(
            rendered_anchors
        )
        samples.append(sample)
    return samples


def _summarize(
    samples: list[dict], presence_threshold: float, *, forced_failures: int = 0
) -> dict:
    totals = defaultdict(int)
    totals["samples"] = forced_failures
    totals["coordinate_collision_failures"] = forced_failures
    cases = []
    for sample in samples:
        kept = torch.ones(len(sample["anchors"]), dtype=torch.bool)
        baseline = evaluate_kept_graph(
            sample,
            kept,
            presence_threshold=presence_threshold,
            radius_px=6.0,
        )
        labels = label_single_delete_actions(
            sample, presence_threshold=presence_threshold
        )
        oracle_kept = apply_oracle_single_delete(sample, labels)
        oracle = evaluate_kept_graph(
            sample,
            oracle_kept,
            presence_threshold=presence_threshold,
            radius_px=6.0,
        )
        totals["samples"] += 1
        totals["candidate_actions"] += len(labels["actions"])
        totals["safe_actions"] += sum(action["safe"] for action in labels["actions"])
        totals["strict_gain_actions"] += sum(
            action["strict_gain"] for action in labels["actions"]
        )
        totals["executed_actions"] += int(bool(labels["winner_removed_nodes"]))
        for prefix, metrics in (("baseline", baseline), ("oracle", oracle)):
            totals[f"{prefix}_atom_exact"] += int(metrics["atom_exact"])
            totals[f"{prefix}_skeleton_exact"] += int(metrics["skeleton_exact"])
            totals[f"{prefix}_atom_fp"] += int(metrics["atom_fp"])
            totals[f"{prefix}_atom_fn"] += int(metrics["atom_fn"])
            totals[f"{prefix}_bond_fp"] += int(metrics["bond_fp"])
            totals[f"{prefix}_bond_fn"] += int(metrics["bond_fn"])
        totals["atom_rescue"] += int(oracle["atom_exact"] and not baseline["atom_exact"])
        totals["atom_break"] += int(baseline["atom_exact"] and not oracle["atom_exact"])
        totals["skeleton_rescue"] += int(
            oracle["skeleton_exact"] and not baseline["skeleton_exact"]
        )
        totals["skeleton_break"] += int(
            baseline["skeleton_exact"] and not oracle["skeleton_exact"]
        )
        if labels["winner_removed_nodes"]:
            cases.append(
                {
                    "key": sample["key"],
                    "removed": min(labels["winner_removed_nodes"]),
                    "baseline": {
                        name: value
                        for name, value in baseline.items()
                        if not isinstance(value, set)
                    },
                    "oracle": {
                        name: value
                        for name, value in oracle.items()
                        if not isinstance(value, set)
                    },
                }
            )
    count = max(totals["samples"], 1)
    baseline_atom = totals["baseline_atom_exact"] / count
    oracle_atom = totals["oracle_atom_exact"] / count
    baseline_skeleton = totals["baseline_skeleton_exact"] / count
    oracle_skeleton = totals["oracle_skeleton_exact"] / count
    return {
        **dict(totals),
        "baseline_atom_exact_rate": baseline_atom,
        "oracle_atom_exact_rate": oracle_atom,
        "atom_exact_delta": oracle_atom - baseline_atom,
        "baseline_skeleton_exact_rate": baseline_skeleton,
        "oracle_skeleton_exact_rate": oracle_skeleton,
        "skeleton_exact_delta": oracle_skeleton - baseline_skeleton,
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Safe-suppression oracle requires CUDA")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Batch size and limit must be positive")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    data_gate = verify_data_gate()
    device = torch.device("cuda")
    direct, direct_payload = load_direct_model(args.direct_checkpoint, device)
    direct_args = direct_payload["training_args"]
    vision, atom, bond, set_head, source = load_components(
        direct_args["source_checkpoint"], direct_args["grounded_graph_checkpoint"], device
    )
    for module in (direct, vision, atom, bond, set_head):
        module.eval().requires_grad_(False)
    config = PredictedProposalConfig(**source["proposal_config"])
    collator = AtomBondSceneCollator(
        AtomBondSceneConfig(
            atom_queries=config.maximum_atoms,
            bond_queries=config.maximum_bonds,
        )
    )
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    presence_threshold = float(direct_payload["validation"]["threshold"])
    dataset = ChemVectorIterableDataset(
        data_gate["route_select_shards"],
        sequence_version="0.4",
        serialize_target=False,
        load_raster_layers=False,
    )
    iterator = iter(dataset)
    samples = []
    coordinate_collisions = []
    total_seen = 0
    for records in _batches(iterator, args.batch_size):
        if args.limit is not None:
            remaining = args.limit - total_seen
            if remaining <= 0:
                break
            records = records[:remaining]
        total_seen += len(records)
        valid_records = []
        for record in records:
            try:
                serialize_chemsvg_v4(record["chemmol"], record["render"])
            except ChemSVGCoordinateCollisionError:
                anchors = torch.tensor(
                    [
                        [float(atom["image_x"]), float(atom["image_y"])]
                        for atom in record["render"]["atoms"]
                    ],
                    dtype=torch.float64,
                ) / 768.0
                coordinate_collisions.append(
                    {
                        "key": record["key"],
                        "molecule_id": record["chemmol"]["molecule_id"],
                        "minimum_target_distance_px": _minimum_target_distance(anchors),
                    }
                )
            else:
                valid_records.append(record)
        if valid_records:
            samples.extend(
                collect_samples(
                    direct,
                    vision,
                    atom,
                    bond,
                    set_head,
                    valid_records,
                    collator,
                    config,
                    atom_threshold,
                    presence_threshold,
                    device,
                )
            )
        if total_seen % 500 < len(records):
            print(
                json.dumps(
                    {
                        "samples_seen": total_seen,
                        "samples_cached": len(samples),
                        "coordinate_collisions": len(coordinate_collisions),
                    }
                ),
                flush=True,
            )
        if args.limit is not None and total_seen >= args.limit:
            break
    expected = args.limit or 8_000
    if total_seen != expected or len(samples) + len(coordinate_collisions) != expected:
        raise ValueError(
            "Route-select coverage differs: "
            f"seen={total_seen} cached={len(samples)} collisions={len(coordinate_collisions)} "
            f"expected={expected}"
        )

    close_molecules = {
        sample["molecule_id"]
        for sample in [*samples, *coordinate_collisions]
        if sample["minimum_target_distance_px"] <= 16.0
    }
    buckets = {
        "all": samples,
        "close_distinct": [
            sample for sample in samples if sample["molecule_id"] in close_molecules
        ],
        "general": [
            sample for sample in samples if sample["molecule_id"] not in close_molecules
        ],
    }
    forced_by_bucket = {
        "all": len(coordinate_collisions),
        "close_distinct": sum(
            row["molecule_id"] in close_molecules for row in coordinate_collisions
        ),
        "general": sum(
            row["molecule_id"] not in close_molecules for row in coordinate_collisions
        ),
    }
    metrics = {
        name: _summarize(
            rows, presence_threshold, forced_failures=forced_by_bucket[name]
        )
        for name, rows in buckets.items()
    }
    oracle_gate = {
        "minimum_atom_exact_delta": 0.03,
        "minimum_skeleton_exact_delta": 0.03,
        "passed": metrics["all"]["atom_exact_delta"] >= 0.03
        and metrics["all"]["skeleton_exact_delta"] >= 0.03
        and metrics["all"]["atom_break"] == 0
        and metrics["all"]["skeleton_break"] == 0,
    }
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cache = {
        "schema_version": "safe-atom-action-cache-0.1",
        "data_commitment_sha256": DATA_COMMITMENT,
        "direct_checkpoint": args.direct_checkpoint,
        "direct_checkpoint_sha256": _sha256_file(args.direct_checkpoint),
        "source_checkpoint": direct_args["source_checkpoint"],
        "grounded_graph_checkpoint": direct_args["grounded_graph_checkpoint"],
        "code_git_revision": code_revision,
        "atom_threshold": atom_threshold,
        "presence_threshold": presence_threshold,
        "maximum_action_distance_px": 16.0,
        "primary_match_radius_px": 6.0,
        "secondary_match_radius_px": 10.0,
        "input_samples": total_seen,
        "coordinate_collisions": coordinate_collisions,
        "samples": samples,
    }
    cache_output = Path(args.cache_output)
    cache_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_output)
    report = {
        "schema_version": "safe-atom-single-delete-oracle-report-0.1",
        "data_gate": data_gate,
        "cache": {
            "path": str(cache_output),
            "sha256": _sha256_file(cache_output),
            "input_samples": total_seen,
            "cached_samples": len(samples),
            "coordinate_collision_failures": len(coordinate_collisions),
        },
        "base": {
            "direct_checkpoint": args.direct_checkpoint,
            "direct_checkpoint_sha256": _sha256_file(args.direct_checkpoint),
            "code_git_revision": code_revision,
            "atom_threshold": atom_threshold,
            "presence_threshold": presence_threshold,
        },
        "metrics": metrics,
        "oracle_gate": oracle_gate,
        "route_confirm_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    compact = {
        name: {key: value for key, value in value.items() if key != "cases"}
        for name, value in metrics.items()
    }
    print(
        json.dumps(
            {"metrics": compact, "oracle_gate": oracle_gate},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
