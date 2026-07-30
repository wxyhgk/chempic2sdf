#!/usr/bin/env python3
"""Run one frozen v0.10.15 structural-twin candidate on route-confirm."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from itertools import islice
import json
import math
import os
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
from chemvector.audit_safe_atom_suppression_oracle import (
    DATA_COMMITMENT,
    DATA_ROOT,
    collect_samples,
    verify_data_gate,
)
from chemvector.audit_structural_twin_certificate import (
    RANKING_SHA256,
    _sha256_file,
    _summarize,
    evaluate_structural_twin_samples,
)
from chemvector.data import ChemVectorIterableDataset
from chemvector.infer_direct_atom_pair_bond import load_direct_model
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.safe_atom_action_model import SafeAtomActionConfig, SafeAtomActionModel
from chemvector.structural_twin_certificate import StructuralTwinCertificateConfig
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.train_bond_presence_calibration import load_components


DEFAULT_MANIFEST = (
    "outputs/candidates/chemvector-v01015-structural-twin/candidate_manifest.json"
)
DEFAULT_OUTPUT = (
    "outputs/audits/chemvector-v01015-structural-twin-route-confirm/report.json"
)
DEFAULT_CACHE = (
    "outputs/caches/chemvector-v01015-route-confirm-actions/base-actions.pt"
)
EXPECTED_CONFIRM_SAMPLES = 42_000
EXPECTED_CONFIRM_SHARDS = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-output", default=DEFAULT_CACHE)
    return parser.parse_args()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


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


def _load_frozen_manifest(path: Path) -> tuple[dict, str, str]:
    manifest_sha = _sha256_file(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    embedded_sha = manifest.pop("manifest_sha256", None)
    recomputed_sha = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    if embedded_sha != recomputed_sha:
        raise ValueError("Candidate manifest commitment differs")
    if manifest.get("schema_version") != "structural-twin-candidate-0.1":
        raise ValueError("Candidate manifest schema differs")
    if manifest.get("status") != "FROZEN-ROUTE-CONFIRM-CANDIDATE":
        raise ValueError("Candidate manifest is not frozen")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if manifest.get("code_git_revision") != revision:
        raise ValueError("Candidate code revision differs from HEAD")
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked_status:
        raise RuntimeError("Tracked worktree must be clean before route-confirm")
    if manifest.get("data_commitment_sha256") != DATA_COMMITMENT:
        raise ValueError("Candidate references another data commitment")
    if manifest.get("ranking_checkpoint_sha256") != RANKING_SHA256:
        raise ValueError("Candidate ranking checkpoint differs")
    route_select_path = Path(manifest["route_select_report"])
    if _sha256_file(route_select_path) != manifest.get("route_select_report_sha256"):
        raise ValueError("Candidate route-select report differs")
    route_select = json.loads(route_select_path.read_text(encoding="utf-8"))
    if route_select.get("status") != "ROUTE-SELECT-GO" or not route_select.get(
        "gate", {}
    ).get("passed"):
        raise ValueError("Candidate did not pass route-select")
    if route_select.get("code_git_revision") != revision:
        raise ValueError("Route-select report was not produced by candidate revision")
    expected_config = asdict(StructuralTwinCertificateConfig())
    if manifest.get("certificate") != expected_config:
        raise ValueError("Candidate structural certificate differs")
    if manifest.get("maximum_actions_per_image") != 1:
        raise ValueError("Candidate action limit differs")
    return manifest, embedded_sha, manifest_sha


def _load_models(manifest: dict, device: torch.device):
    direct_path = manifest["direct_checkpoint"]
    if _sha256_file(direct_path) != manifest["direct_checkpoint_sha256"]:
        raise ValueError("Frozen direct checkpoint differs")
    ranking_path = manifest["ranking_checkpoint"]
    if _sha256_file(ranking_path) != RANKING_SHA256:
        raise ValueError("Frozen ranking checkpoint differs")
    direct, direct_payload = load_direct_model(direct_path, device)
    direct_args = direct_payload["training_args"]
    for name in ("source_checkpoint", "grounded_graph_checkpoint"):
        if direct_args[name] != manifest[name]:
            raise ValueError(f"Frozen {name} path differs")
        if _sha256_file(manifest[name]) != manifest[f"{name}_sha256"]:
            raise ValueError(f"Frozen {name} content differs")
    vision, atom, bond, set_head, source = load_components(
        direct_args["source_checkpoint"],
        direct_args["grounded_graph_checkpoint"],
        device,
    )
    ranking_payload = torch.load(ranking_path, map_location="cpu", weights_only=False)
    ranking = SafeAtomActionModel(SafeAtomActionConfig(**ranking_payload["config"]))
    ranking.load_state_dict(ranking_payload["model"])
    for model in (direct, vision, atom, bond, set_head, ranking):
        model.eval().requires_grad_(False)
    return direct, direct_payload, vision, atom, bond, set_head, source, ranking


def _collect_confirm_samples(args, manifest, models, device):
    data_gate = verify_data_gate()
    confirm_root = DATA_ROOT / "route_confirm"
    entry_mode = stat.S_IMODE(confirm_root.stat().st_mode)
    if entry_mode != 0:
        raise RuntimeError("route-confirm was not sealed at entry")
    direct, direct_payload, vision, atom, bond, set_head, source, _ = models
    config = PredictedProposalConfig(**source["proposal_config"])
    collator = AtomBondSceneCollator(
        AtomBondSceneConfig(
            atom_queries=config.maximum_atoms,
            bond_queries=config.maximum_bonds,
        )
    )
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    presence_threshold = float(direct_payload["validation"]["threshold"])
    if atom_threshold != manifest["atom_threshold"]:
        raise ValueError("Frozen atom threshold differs")
    if presence_threshold != manifest["presence_threshold"]:
        raise ValueError("Frozen presence threshold differs")
    samples = []
    coordinate_collisions = []
    total_seen = 0
    commitment = json.loads(
        (DATA_ROOT / "commitment.json").read_text(encoding="utf-8")
    )
    tar_entries = [
        row
        for row in commitment["tars"]
        if row["path"].startswith("route_confirm/shards/")
    ]
    if len(tar_entries) != EXPECTED_CONFIRM_SHARDS:
        raise ValueError("Committed route-confirm shard count differs")
    shards_root = confirm_root / "shards"
    opened_shards = []
    os.chmod(confirm_root, 0o500)
    try:
        if stat.S_IMODE(shards_root.stat().st_mode) != 0:
            raise RuntimeError("route-confirm shards directory was not sealed")
        os.chmod(shards_root, 0o500)
        shards = []
        for entry in tar_entries:
            path = DATA_ROOT / Path(entry["path"]).relative_to("route_confirm")
            if stat.S_IMODE(path.stat().st_mode) != 0:
                raise RuntimeError(f"route-confirm shard was not sealed: {path}")
            os.chmod(path, 0o400)
            opened_shards.append(path)
            if path.stat().st_size != entry["bytes"]:
                raise ValueError(f"Committed route-confirm shard size differs: {path}")
            if _sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Committed route-confirm shard SHA differs: {path}")
            shards.append(path)
        dataset = ChemVectorIterableDataset(
            [str(path) for path in shards],
            sequence_version="0.4",
            serialize_target=False,
            load_raster_layers=False,
        )
        for records in _batches(iter(dataset), args.batch_size):
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
                            "minimum_target_distance_px": _minimum_target_distance(
                                anchors
                            ),
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
    finally:
        for path in reversed(opened_shards):
            os.chmod(path, 0o000)
        if shards_root.exists():
            os.chmod(shards_root, 0o000)
        os.chmod(confirm_root, 0o000)
    restored_mode = stat.S_IMODE(confirm_root.stat().st_mode)
    if restored_mode != 0:
        raise RuntimeError("route-confirm permissions were not restored")
    if total_seen != EXPECTED_CONFIRM_SAMPLES:
        raise ValueError(
            f"Confirm coverage differs: {total_seen} != {EXPECTED_CONFIRM_SAMPLES}"
        )
    if len(samples) + len(coordinate_collisions) != total_seen:
        raise ValueError("Confirm cached/collision coverage differs")
    return (
        data_gate,
        samples,
        coordinate_collisions,
        atom_threshold,
        presence_threshold,
        restored_mode,
    )


def _safe_precision_lower(executed: int, harmful: int, alpha: float = 0.05) -> float:
    if executed < 1:
        return 0.0
    if harmful:
        return 0.0
    return math.pow(alpha, 1.0 / executed)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    output = Path(args.output)
    cache_output = Path(args.cache_output)
    for path in (output, cache_output):
        if path.exists():
            raise RuntimeError(f"Frozen confirm output already exists: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("Structural-twin confirm requires CUDA")
    started = time.perf_counter()
    manifest_path = Path(args.candidate_manifest)
    manifest, embedded_manifest_sha, manifest_file_sha = _load_frozen_manifest(
        manifest_path
    )
    torch.manual_seed(manifest["seed"])
    torch.cuda.manual_seed_all(manifest["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    models = _load_models(manifest, device)
    (
        data_gate,
        samples,
        coordinate_collisions,
        atom_threshold,
        presence_threshold,
        restored_mode,
    ) = _collect_confirm_samples(args, manifest, models, device)
    cache = {
        "schema_version": "structural-twin-confirm-cache-0.1",
        "data_commitment_sha256": DATA_COMMITMENT,
        "candidate_manifest_sha256": embedded_manifest_sha,
        "input_samples": EXPECTED_CONFIRM_SAMPLES,
        "coordinate_collisions": coordinate_collisions,
        "atom_threshold": atom_threshold,
        "presence_threshold": presence_threshold,
        "samples": samples,
    }
    cache_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_output)
    ranking = models[-1]
    prepared, certified, decisions = evaluate_structural_twin_samples(
        samples,
        ranking,
        presence_threshold=presence_threshold,
        config=StructuralTwinCertificateConfig(),
    )
    close_molecules = {
        sample["molecule_id"]
        for sample in [*samples, *coordinate_collisions]
        if sample["minimum_target_distance_px"] <= 16.0
    }
    all_molecules = {sample["molecule_id"] for sample in samples}
    all_molecules.update(row["molecule_id"] for row in coordinate_collisions)
    general_molecules = all_molecules - close_molecules
    forced = {
        "all": len(coordinate_collisions),
        "close_distinct": sum(
            row["molecule_id"] in close_molecules for row in coordinate_collisions
        ),
        "general": sum(
            row["molecule_id"] in general_molecules for row in coordinate_collisions
        ),
    }
    metrics = {
        name: _summarize(
            decisions, molecule_ids=molecules, forced_failures=forced[name]
        )
        for name, molecules in (
            ("all", all_molecules),
            ("close_distinct", close_molecules),
            ("general", general_molecules),
        )
    }
    overall = metrics["all"]
    safe_lower = _safe_precision_lower(
        overall["executed_actions"], overall["harmful_actions"]
    )
    zero_loss_fields = (
        "harmful_actions",
        "atom_break",
        "skeleton_break",
        "lost_matched_atom_ids",
        "lost_expressible_bond_ids",
        "lost_covered_bond_ids",
        "lost_typed_bond_ids",
    )
    structure_gate = {
        "passed": overall["executed_actions"] >= 2_995
        and all(overall[name] == 0 for name in zero_loss_fields)
        and all(
            all(metrics[bucket][name] == 0 for name in zero_loss_fields)
            for bucket in ("close_distinct", "general")
        )
        and safe_lower >= 0.999
        and overall["atom_exact_delta"] >= 0.03
        and overall["skeleton_exact_delta"] >= 0.03,
        "minimum_executed_actions": 2_995,
        "minimum_safe_precision_lower_95": 0.999,
        "safe_precision_lower_95": safe_lower,
        "minimum_atom_exact_delta": 0.03,
        "minimum_skeleton_exact_delta": 0.03,
    }
    report = {
        "schema_version": "structural-twin-route-confirm-report-0.1",
        "status": (
            "STRUCTURE-CONFIRM-GO"
            if structure_gate["passed"]
            else "STRUCTURE-CONFIRM-NO-GO"
        ),
        "candidate_manifest": {
            "path": str(manifest_path),
            "commitment_sha256": embedded_manifest_sha,
            "file_sha256": manifest_file_sha,
        },
        "data_gate": data_gate,
        "certificate": asdict(StructuralTwinCertificateConfig()),
        "cache": {
            "path": str(cache_output),
            "sha256": _sha256_file(cache_output),
            "cached_samples": len(samples),
            "coordinate_collision_failures": len(coordinate_collisions),
        },
        "action_audit": {
            "candidate_actions": len(prepared["features"]),
            "certified_actions": int(certified.sum()),
            "certified_safe_actions": int(
                (certified & prepared["safe_targets"]).sum()
            ),
            "certified_unsafe_actions": int(
                (certified & ~prepared["safe_targets"]).sum()
            ),
        },
        "metrics": metrics,
        "structure_gate": structure_gate,
        "route_confirm_read": True,
        "route_confirm_mode_restored": f"{restored_mode:03o}",
        "sealed_test_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output, report)
    compact = {
        name: {key: value for key, value in value.items() if key != "cases"}
        for name, value in metrics.items()
    }
    print(
        json.dumps(
            {
                "status": report["status"],
                "action_audit": report["action_audit"],
                "metrics": compact,
                "structure_gate": structure_gate,
                "route_confirm_mode_restored": report[
                    "route_confirm_mode_restored"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
