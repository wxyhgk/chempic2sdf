#!/usr/bin/env python3
"""Audit the frozen top-8 endpoint-pair candidate generator on fresh_dev."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from chemvector.components.bonds.objects.semantic_bond_endpoint import (
    SemanticBondEndpointConfig,
    SemanticBondEndpointDetector,
)
from chemvector.components.vision.semantic_ink_separator import (
    SemanticInkSeparator,
    SemanticInkSeparatorConfig,
)
from chemvector.core.data import ChemVectorIterableDataset
from chemvector.evaluation.audits.bonds.audit_v1p2c_endpoint_pair_oracle import (
    DEFAULT_ENDPOINT_CHECKPOINT,
    ENDPOINT_CHECKPOINT_SHA256,
    ENDPOINT_THRESHOLD,
)
from chemvector.evaluation.audits.bonds.audit_v1p2d0_local_partner_oracle import (
    GO_GATES,
    PARTNER_COUNTS,
    ROUTES,
    run_audit,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.bonds.objects.train_v1p2_semantic_bond_object import (
    DEFAULT_SEPARATOR,
    SEPARATOR_SHA256,
    V1BondObjectCollator,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


AUDIT_NAME = "chemvector-v1p2d2e0-fresh-candidate-oracle"
DEFAULT_OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
FRESH_ROOT = Path("Data/sdf/chemvector_joint_v1/fresh_eval_12k")
SOURCE_COMMITMENT = (
    "13a456afb2c475a73fc4dca82df5fe62d4a31277d6b584637d9cd92736024512"
)
SAMPLES_CSV_SHA256 = (
    "b46571e188620e8e117e294648ae1a5a936cc8e0958c88da6ad830af0e9d0619"
)
SPLIT_SEED = 20260821
CALIBRATION_MOLECULES = 100
HOLDOUT_MOLECULES = 100
EXPECTED_IMAGES = 8_000
EXPECTED_MOLECULES = 2_000
EXPECTED_VARIANTS = frozenset({"coordgen", "original", "rdkit", "stress"})
SOURCE_REJECTIONS = {
    "chembl:CHEMBL5856698": (
        "rdkit/stress layouts place atom 15 and atom 19 at identical visual "
        "coordinates; ChemScene v0.2 cannot represent the sample losslessly"
    )
}


def _rank_molecule(molecule_id: str) -> tuple[str, str]:
    return (
        hashlib.sha256(f"{SPLIT_SEED}:{molecule_id}".encode()).hexdigest(),
        molecule_id,
    )


def split_fresh_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) != EXPECTED_IMAGES:
        raise ValueError(f"P2-D2-E0 fresh_dev image count differs: {len(rows)}")
    if {str(row["split"]) for row in rows} != {"fresh_dev"}:
        raise ValueError("P2-D2-E0 source includes a non-fresh_dev split")
    if {str(row["font_family_group"]) for row in rows} != {"heldout"}:
        raise ValueError("P2-D2-E0 source includes a non-held-out font group")
    by_molecule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_molecule[str(row["molecule_id"])].append(row)
    if len(by_molecule) != EXPECTED_MOLECULES:
        raise ValueError("P2-D2-E0 fresh_dev molecule count differs")
    if any(
        len(group) != 4
        or {str(row["variant"]) for row in group} != EXPECTED_VARIANTS
        for group in by_molecule.values()
    ):
        raise ValueError("P2-D2-E0 molecule variants differ")
    ranked = [
        molecule_id
        for molecule_id in sorted(by_molecule, key=_rank_molecule)
        if molecule_id not in SOURCE_REJECTIONS
    ]
    calibration_ids = set(ranked[:CALIBRATION_MOLECULES])
    holdout_ids = set(
        ranked[CALIBRATION_MOLECULES : CALIBRATION_MOLECULES + HOLDOUT_MOLECULES]
    )
    calibration = [
        row for row in rows if str(row["molecule_id"]) in calibration_ids
    ]
    holdout = [row for row in rows if str(row["molecule_id"]) in holdout_ids]
    calibration.sort(key=lambda row: str(row["sample_key"]))
    holdout.sort(key=lambda row: str(row["sample_key"]))
    if (len(calibration), len(holdout)) != (400, 400):
        raise ValueError("P2-D2-E0 calibration/holdout image counts differ")
    return calibration, holdout


def load_fresh_split() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict]:
    commitment = json.loads(
        (FRESH_ROOT / "commitment.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (FRESH_ROOT / "validation.json").read_text(encoding="utf-8")
    )
    if commitment.get("commitment_sha256") != SOURCE_COMMITMENT:
        raise ValueError("P2-D2-E0 fresh_dev commitment differs")
    if file_sha256(FRESH_ROOT / "samples.csv") != SAMPLES_CSV_SHA256:
        raise ValueError("P2-D2-E0 fresh_dev samples.csv SHA256 differs")
    if validation.get("status") != "passed":
        raise ValueError("P2-D2-E0 fresh_dev validation did not pass")
    with (FRESH_ROOT / "samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    calibration, holdout = split_fresh_rows(rows)
    return calibration, holdout, commitment


def _fresh_loader(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    if shuffle:
        raise ValueError("P2-D2-E0 Oracle loader must not shuffle")
    keys = {str(row["sample_key"]) for row in rows}
    shard_names = sorted({str(row["output_shard"]) for row in rows})
    if not shard_names or any(
        not name.startswith("shards/fresh_dev-") for name in shard_names
    ):
        raise ValueError("P2-D2-E0 source references an unexpected shard")
    shards = [str(FRESH_ROOT / name) for name in shard_names]
    dataset = ChemVectorIterableDataset(
        shards,
        shardshuffle=False,
        sample_shuffle_buffer=0,
        sample_shuffle_seed=seed,
        include_keys=keys,
        serialize_target=False,
        load_raster_layers=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=V1BondObjectCollator(),
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _molecule_ids_sha256(rows: list[dict[str, Any]]) -> str:
    molecule_ids = sorted({str(row["molecule_id"]) for row in rows})
    return hashlib.sha256(
        "".join(f"{value}\n" for value in molecule_ids).encode()
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-checkpoint", type=Path, default=DEFAULT_ENDPOINT_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2-D2-E0 fresh candidate Oracle requires CUDA")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-D2-E0 output is not empty: {args.output_dir}")
    if file_sha256(args.endpoint_checkpoint) != ENDPOINT_CHECKPOINT_SHA256:
        raise ValueError("P2-D2-E0 endpoint checkpoint SHA256 differs")
    if file_sha256(DEFAULT_SEPARATOR) != SEPARATOR_SHA256:
        raise ValueError("P2-D2-E0 separator SHA256 differs")

    calibration, holdout, commitment = load_fresh_split()
    device = torch.device("cuda")
    separator_payload = torch.load(DEFAULT_SEPARATOR, map_location=device)
    separator = SemanticInkSeparator(
        SemanticInkSeparatorConfig(**separator_payload["config"])
    ).to(device)
    separator.load_state_dict(separator_payload["model"], strict=True)
    separator.eval().requires_grad_(False)
    endpoint_payload = torch.load(args.endpoint_checkpoint, map_location=device)
    detector = SemanticBondEndpointDetector(
        SemanticBondEndpointConfig(**endpoint_payload["config"])
    ).to(device)
    detector.load_state_dict(endpoint_payload["detector"], strict=True)
    detector.eval().requires_grad_(False)

    started = time.perf_counter()
    summary, output_rows = run_audit(
        separator,
        detector,
        calibration,
        device,
        workers=args.workers,
        log_every=args.log_every,
        loader_factory=_fresh_loader,
    )
    winner = max(
        ROUTES,
        key=lambda route: (
            summary["routes"][route]["top8"]["pair_recall"],
            summary["routes"][route]["top8"]["image_exact"],
        ),
    )
    winner_top8 = summary["routes"][winner]["top8"]
    crossed = summary["by_stress_operation"].get("crossed_bonds", {})
    crossed_top8 = crossed.get(winner, {}).get("top8", {})
    checks = {
        "calibration_samples_exact400": summary["samples"] == 400,
        "winner_top8_pair_recall_min95pct": winner_top8["pair_recall"]
        >= GO_GATES["winner_top8_pair_recall"],
        "winner_top8_image_exact_min30pct": winner_top8["image_exact"]
        >= GO_GATES["winner_top8_image_exact"],
        "winner_top8_stress_pair_recall_min90pct": summary["by_variant"][
            "stress"
        ][winner]["top8"]["pair_recall"]
        >= GO_GATES["winner_top8_stress_pair_recall"],
        "winner_top8_crossed_pair_recall_min85pct": bool(crossed_top8)
        and crossed_top8["pair_recall"]
        >= GO_GATES["winner_top8_crossed_pair_recall"],
        "source_commitment_matches": commitment["commitment_sha256"]
        == SOURCE_COMMITMENT,
        "heldout_font_only": True,
        "no_maximum_bond_length_cutoff": True,
        "fresh_holdout_samples_read_zero": True,
        "sealed_test_samples_read_zero": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    split = {
        "schema_version": "chemvector-v1p2d2e0-fresh-split-0.1",
        "seed": SPLIT_SEED,
        "calibration_molecules": CALIBRATION_MOLECULES,
        "calibration_samples": len(calibration),
        "calibration_molecule_ids_sha256": _molecule_ids_sha256(calibration),
        "holdout_molecules": HOLDOUT_MOLECULES,
        "holdout_samples": len(holdout),
        "holdout_molecule_ids_sha256": _molecule_ids_sha256(holdout),
        "fresh_holdout_tar_members_read": 0,
        "sealed_test_samples_read": 0,
        "source_rejections": SOURCE_REJECTIONS,
    }
    _atomic_json(args.output_dir / "split.json", split)
    result = {
        "schema_version": "chemvector-v1p2d2e0-fresh-candidate-oracle-0.1",
        "audit_name": AUDIT_NAME,
        "decision": (
            "fresh_candidate_oracle_go"
            if all(checks.values())
            else "fresh_candidate_oracle_no_go"
        ),
        "checks": checks,
        "gates": GO_GATES,
        "winner": winner,
        "endpoint_threshold": ENDPOINT_THRESHOLD,
        "partner_counts": PARTNER_COUNTS,
        "endpoint_checkpoint_sha256": ENDPOINT_CHECKPOINT_SHA256,
        "separator_sha256": SEPARATOR_SHA256,
        "source_commitment": SOURCE_COMMITMENT,
        "source_samples_csv_sha256": SAMPLES_CSV_SHA256,
        "source_rejections": SOURCE_REJECTIONS,
        "rows_sha256": file_sha256(rows_path),
        "split": split,
        "summary": summary,
        "fresh_holdout_samples_read": 0,
        "sealed_test_samples_read": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output_dir / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
