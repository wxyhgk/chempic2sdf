#!/usr/bin/env python3
"""Audit whether 64/96px crops reveal target-adjacent structure missing at 32px."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from chemvector.evaluation.audits.atoms import (
    audit_g3_p5e6_refine_error_decomposition as p5e6,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p4e9_recovery_oracle as oracle,
)
from chemvector.evaluation.evaluators.atoms import (
    evaluate_g3_p4e9_local_recovery_pilot20k as p4eval,
)
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.atoms import (
    train_g3_p4e9_local_recovery_pilot20k as p4train,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256
from chemvector.data import structure_labels


AUDIT_NAME = "chemvector-g3-p6e0-context-capacity"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
P5E6_RECORDS = p5e6.OUTPUT / "records.jsonl"
P5E6_RECORDS_SHA256 = (
    "262ee83ff863bc7ea6ef80f34928d2f69e6761786769e39ad9879809c4a87c9a"
)
CROP_SIZES = (32, 64, 96)
MINIMUM_FAILURE_ADDED_RATE = 0.40
IMAGE_SIZE = 768


def in_crop(anchor: torch.Tensor, source: torch.Tensor, crop_size: int) -> bool:
    half = crop_size * 0.5
    delta = torch.abs((anchor.float() - source.float()) * IMAGE_SIZE)
    return bool((delta <= half).all())


def select_context_size(rates: dict[int, dict[str, float]]) -> int | None:
    for crop_size in (64, 96):
        if all(
            rates[crop_size][split] >= MINIMUM_FAILURE_ADDED_RATE
            for split in ("calibration", "holdout")
        ):
            return crop_size
    return None


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P6-E0 output already exists: {OUTPUT}")
    for path, expected in (
        (P5E6_RECORDS, P5E6_RECORDS_SHA256),
        (p4train.ORACLE_ACTIONS, p5e6.p5e1.ORACLE_ACTIONS_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P6-E0 input SHA256 differs: {path}")
    p5_records = {
        (str(row["sample_key"]), int(row["base_source_id"])): row
        for row in _jsonl(P5E6_RECORDS)
    }
    records = []
    for split in ("calibration", "holdout"):
        actions = [
            row
            for row in p4eval._jsonl(p4train.ORACLE_ACTIONS)
            if row["route"] == "ADD_REFINE" and row["split"] == split
        ]
        by_key: dict[str, list[dict]] = defaultdict(list)
        for action in actions:
            by_key[str(action["sample_key"])].append(action)
        path = p4eval.ORACLE_CALIBRATION if split == "calibration" else p4eval.ORACLE_HOLDOUT
        manifest, _ = p4eval._manifest(path)
        selected = [row for row in manifest if row.sample_key in by_key]
        for sample in p1a.iter_manifest_samples(selected, accepted_split="train"):
            key = str(sample["key"])
            labels = structure_labels(sample)
            atoms = sorted(labels["atoms"], key=lambda atom: int(atom["atom_id"]))
            anchors = torch.tensor(
                p1a.G3AtomCollator.atom_anchors(sample), dtype=torch.float32
            )
            id_to_index = {
                int(atom["atom_id"]): index for index, atom in enumerate(atoms)
            }
            for action in by_key[key]:
                identity = (key, int(action["base_source_id"]))
                if identity not in p5_records:
                    raise RuntimeError(f"P6-E0 lacks P5-E6 record: {identity}")
                target_index = int(action["target_index"])
                target_atom_id = int(atoms[target_index]["atom_id"])
                neighbors = []
                incident_ports = []
                for bond in labels["bonds"]:
                    begin = int(bond["begin_atom"])
                    end = int(bond["end_atom"])
                    if begin == target_atom_id:
                        neighbors.append(id_to_index[end])
                        incident_ports.append(torch.tensor(bond["end"]) / IMAGE_SIZE)
                    elif end == target_atom_id:
                        neighbors.append(id_to_index[begin])
                        incident_ports.append(torch.tensor(bond["begin"]) / IMAGE_SIZE)
                source = torch.tensor(action["base_anchor"], dtype=torch.float32)
                neighbor_counts = {
                    crop_size: sum(
                        in_crop(anchors[index], source, crop_size) for index in neighbors
                    )
                    for crop_size in CROP_SIZES
                }
                port_counts = {
                    crop_size: sum(
                        in_crop(port.float(), source, crop_size) for port in incident_ports
                    )
                    for crop_size in CROP_SIZES
                }
                record = p5_records[identity]
                records.append(
                    {
                        "split": split,
                        "sample_key": key,
                        "base_source_id": int(action["base_source_id"]),
                        "success_6px": bool(record["success_6px"]),
                        "target_degree": len(neighbors),
                        "neighbor_counts": neighbor_counts,
                        "incident_opposite_port_counts": port_counts,
                        "added_neighbor_64": neighbor_counts[64] > neighbor_counts[32],
                        "added_neighbor_96": neighbor_counts[96] > neighbor_counts[32],
                        "all_neighbors_visible_32": neighbor_counts[32] == len(neighbors),
                        "all_neighbors_visible_64": neighbor_counts[64] == len(neighbors),
                        "all_neighbors_visible_96": neighbor_counts[96] == len(neighbors),
                    }
                )
    expected = {"calibration": 31, "holdout": 67}
    summaries = {}
    rates = {64: {}, 96: {}}
    for split, count in expected.items():
        current = [row for row in records if row["split"] == split]
        if len(current) != count:
            raise RuntimeError(f"P6-E0 {split} count differs")
        failures = [row for row in current if not row["success_6px"]]
        successes = [row for row in current if row["success_6px"]]
        summaries[split] = {
            "actions": len(current),
            "failures": len(failures),
            "failure_added_neighbor_64": sum(row["added_neighbor_64"] for row in failures),
            "failure_added_neighbor_96": sum(row["added_neighbor_96"] for row in failures),
            "success_added_neighbor_64": sum(row["added_neighbor_64"] for row in successes),
            "all_neighbors_visible": {
                str(size): sum(row[f"all_neighbors_visible_{size}"] for row in current)
                for size in CROP_SIZES
            },
            "mean_neighbor_count": {
                str(size): float(np.mean([row["neighbor_counts"][size] for row in current]))
                for size in CROP_SIZES
            },
        }
        for size in (64, 96):
            rates[size][split] = (
                sum(row[f"added_neighbor_{size}"] for row in failures) / len(failures)
                if failures
                else 0.0
            )
    selected_size = select_context_size(rates)
    checks = {
        "calibration_actions_31": summaries["calibration"]["actions"] == 31,
        "holdout_actions_67": summaries["holdout"]["actions"] == 67,
        "selected_context_exists": selected_size is not None,
        "dev_confirm_external_sealed_not_read": True,
    }
    decision = "larger_context_capacity_go" if all(checks.values()) else "larger_context_capacity_no_go"
    OUTPUT.mkdir(parents=True)
    oracle._atomic_jsonl(OUTPUT / "records.jsonl", records)
    audit = {
        "schema_version": "g3-p6e0-context-capacity-0.1",
        "decision": decision,
        "checks": checks,
        "selected_context_size": selected_size,
        "failure_added_neighbor_rates": rates,
        "summaries": summaries,
        "protocol": {
            "training_performed": False,
            "crop_sizes": CROP_SIZES,
            "minimum_failure_added_rate": MINIMUM_FAILURE_ADDED_RATE,
            "context_signal": "target incident neighbor anchor newly enters crop",
            "dev_confirm_external_sealed_read": False,
        },
        "lineage": {
            "p5e6_records_sha256": P5E6_RECORDS_SHA256,
            "oracle_actions_sha256": p5e6.p5e1.ORACLE_ACTIONS_SHA256,
        },
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
