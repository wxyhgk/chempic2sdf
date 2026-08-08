#!/usr/bin/env python3
"""Online calibration and held-out evaluation for one P4-E9 recovery pilot."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from chemvector.components.recovery.local_atom_recovery import (
    LocalAtomRecoveryConfig,
    LocalAtomRecoveryHead,
)
from chemvector.training.atoms.g3_p4_local_recovery_data import (
    CANDIDATES,
    build_local_recovery_observations,
)
from chemvector.training.atoms.train_g3_p4e9_local_recovery_pilot20k import (
    ORACLE_ACTIONS,
    ORACLE_AUDIT,
)


# Keep evaluation split paths explicit so the evaluator remains auditable.
ORACLE_ROOT = Path(
    "outputs/audits/chemvector-g3-p4e9-recovery-oracle-pilot20k"
)
ORACLE_CALIBRATION = ORACLE_ROOT / "calibration.jsonl"
ORACLE_HOLDOUT = ORACLE_ROOT / "holdout.jsonl"
ORACLE_IMAGES = ORACLE_ROOT / "images.jsonl"
P4E8_AUDIT = Path(
    "outputs/audits/chemvector-g3-p4e8-local-multiplicity-components/audit.json"
)
P4E8_AUDIT_SHA256 = (
    "7062f586af88ff122b31538890c8bc72a9af7b8dfb54d3f9f5841b889784480b"
)
ROUTE_PROTOCOL = {
    "split": {
        "oracle_route": "ADD_SPLIT",
        "model_type": "chemvector-g3-p4e9a-local-binary-split-pilot20k",
    },
    "refine": {
        "oracle_route": "ADD_REFINE",
        "model_type": "chemvector-g3-p4e9b-structural-anchor-refine-pilot20k",
    },
}
ONLINE_BATCH_SIZE = 8
TOP128 = 128
TOP1024 = 1024
MATCH_RADIUS_PIXELS = 6.0
MINIMUM_ORACLE_RECOVERY = 0.50
MINIMUM_SHARED_SPLIT_RECOVERY = 14
SEED = 20260851


@dataclass(frozen=True, slots=True)
class EvaluationManifestRecord:
    sample_key: str
    molecule_id: str
    inchikey: str
    shard: str
    variant: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _manifest(path: Path) -> tuple[list[EvaluationManifestRecord], list[dict]]:
    rows = _jsonl(path)
    records = []
    seen = set()
    required = {"sample_key", "molecule_id", "inchikey", "shard", "variant"}
    for row in rows:
        if required - row.keys():
            raise ValueError(f"P4-E9 evaluation manifest is incomplete: {path}")
        key = str(row["sample_key"])
        if key in seen:
            raise ValueError(f"P4-E9 evaluation manifest duplicates {key}")
        seen.add(key)
        records.append(
            EvaluationManifestRecord(
                sample_key=key,
                molecule_id=str(row["molecule_id"]),
                inchikey=str(row["inchikey"]),
                shard=str(row["shard"]),
                variant=str(row["variant"]),
            )
        )
    if not records:
        raise ValueError(f"P4-E9 evaluation manifest is empty: {path}")
    return records, rows


def _prefix_peaks(peak: Mapping[str, object], count: int) -> dict:
    output = {}
    for name in ("anchors", "scores", "source_ids"):
        value = peak.get(name)
        if not isinstance(value, torch.Tensor) or len(value) < count:
            raise ValueError(f"P4-E9 peak field {name} lacks a top-{count} prefix")
        output[name] = value[:count]
    return output


def _actions_by_sample(actions: list[dict], route: str, split: str) -> dict[str, list[dict]]:
    oracle_route = ROUTE_PROTOCOL[route]["oracle_route"]
    output: dict[str, list[dict]] = {}
    for action in actions:
        if str(action.get("route")) != oracle_route or str(action.get("split")) != split:
            continue
        output.setdefault(str(action["sample_key"]), []).append(action)
    return output


def _route_output(
    model_output: Mapping[str, torch.Tensor], route: str
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = "split" if route == "split" else "refine"
    return model_output[f"{prefix}_logits"], model_output[f"{prefix}_offsets"]


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def records_sha256(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(f"{record['sample_key']}\n".encode("utf-8"))
        digest.update(f"{record['variant']}\n".encode("utf-8"))
        for name in (
            "base_anchors",
            "source_ids",
            "target_anchors",
            "logits",
            "offsets",
            "top1024_anchors",
            "top1024_source_ids",
        ):
            if name not in record:
                continue
            value = record[name].detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.numpy().tobytes())
        digest.update(
            json.dumps(
                record.get("oracle_actions", []),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return digest.hexdigest()


@torch.no_grad()
def collect_online_records(
    manifest: Sequence[EvaluationManifestRecord],
    *,
    route: str,
    route_actions: Mapping[str, list[dict]] | None,
    dynamic_oracle: bool,
    include_top1024: bool,
    model: LocalAtomRecoveryHead,
    field,
    atom_head,
    vision,
    frozen_block,
    device: torch.device,
) -> tuple[list[dict], dict]:
    """Run frozen P4-E3 and the route head; GT is used only after forward."""

    from chemvector.components.atoms.objects.g3_atom_center_field import (
        decode_atom_center_field,
    )
    from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
        extract_regional_maximum_objects,
    )
    from chemvector.evaluation.audits.atoms.build_g3_p4e9_recovery_oracle import (
        recovery_actions,
    )
    from chemvector.training.atoms import (
        train_g3_p1a_canonical_atom_selector as p1a,
    )
    from chemvector.training.atoms import (
        train_g3_p1t_continuous_stride2_center_field as p1t,
    )
    from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
        _autocast_context,
    )

    if route not in ROUTE_PROTOCOL:
        raise ValueError(f"Unknown P4-E9 evaluation route: {route}")
    collator = p1a.G3AtomCollator(maximum_atoms=CANDIDATES, image_size=768)
    model.eval()
    records = []
    sample_keys = []
    started = time.perf_counter()
    maximum = TOP1024 if include_top1024 else TOP128
    for batch in p1a.iter_manifest_batches(
        list(manifest),
        collator,
        batch_size=ONLINE_BATCH_SIZE,
        accepted_split="train",
    ):
        objects, stroke = p1t.model_inputs(
            atom_head,
            vision,
            frozen_block,
            batch["pixel_values"],
            device=device,
            training=False,
        )
        with _autocast_context(device, "bf16"):
            field_output = field(objects, stroke)
        all_peaks = extract_regional_maximum_objects(
            decode_atom_center_field(field_output), maximum_objects=maximum
        )
        top128 = [_prefix_peaks(peak, TOP128) for peak in all_peaks]
        observations = build_local_recovery_observations(
            batch["pixel_values"].to(device), field_output, top128
        )
        with _autocast(device):
            model_output = model(
                observations["image_crops"],
                observations["score_crops"],
                observations["candidate_features"],
                observations["candidate_scores"],
                observations["candidate_ranks"],
                observations["candidate_mask"],
            )
        logits, offsets = _route_output(model_output, route)
        for index, key_value in enumerate(batch["keys"]):
            key = str(key_value)
            expected = batch["atom_anchors"][index][batch["atom_mask"][index]].float()
            if dynamic_oracle:
                actions = recovery_actions(
                    top128[index]["anchors"].cpu(),
                    top128[index]["source_ids"].cpu(),
                    expected.cpu(),
                )
                actions = [
                    action
                    for action in actions
                    if str(action["route"]) == ROUTE_PROTOCOL[route]["oracle_route"]
                ]
            else:
                actions = list((route_actions or {}).get(key, []))
            record = {
                "sample_key": key,
                "variant": str(batch["variants"][index]),
                "base_anchors": top128[index]["anchors"].float().cpu(),
                "source_ids": top128[index]["source_ids"].long().cpu(),
                "target_anchors": expected.cpu(),
                "logits": logits[index].float().cpu(),
                "offsets": offsets[index].float().cpu(),
                "oracle_actions": actions,
            }
            if include_top1024:
                record["top1024_anchors"] = all_peaks[index]["anchors"].float().cpu()
                record["top1024_source_ids"] = all_peaks[index]["source_ids"].long().cpu()
            records.append(record)
            sample_keys.append(key)
    if len(records) != len(manifest) or sample_keys != [item.sample_key for item in manifest]:
        raise RuntimeError("P4-E9 online evaluation stream is incomplete or reordered")
    return records, {
        "images": len(records),
        "records_sha256": records_sha256(records),
        "elapsed_seconds": time.perf_counter() - started,
        "dynamic_oracle": dynamic_oracle,
        "top1024_retained": include_top1024,
    }


def _selected_indices(logits: torch.Tensor, threshold: float) -> list[int]:
    if logits.ndim != 1 or not logits.is_floating_point():
        raise ValueError("P4-E9 replay logits must be a floating vector")
    if not bool(torch.isfinite(logits).all()) or not math.isfinite(float(threshold)):
        raise ValueError("P4-E9 replay logits and threshold must be finite")
    eligible = torch.where(logits.double() >= float(threshold))[0]
    if not len(eligible):
        return []
    order = torch.argsort(logits[eligible], descending=True, stable=True)
    return eligible[order[:16]].tolist()


def replay_shared_split_components(
    dev_records: list[dict], components: list[dict], threshold: float
) -> dict:
    """Replay the 28 P4-E8 top1024 shared components by exact source identity."""

    from chemvector.evaluation.audits.atoms.audit_g3_p4e6_missing_peak_sources import (
        maximum_cardinality_assignments,
    )

    shared = [
        item
        for item in components
        if item.get("kind") == "shared_deficient"
        and int(item.get("gt_count", 0)) == 2
        and int(item.get("candidate_count", 0)) == 1
        and int(item.get("deficiency", 0)) == 1
    ]
    if len(shared) != 28:
        raise ValueError(f"P4-E9 expected 28 P4-E8 shared components, found {len(shared)}")
    by_key = {str(record["sample_key"]): record for record in dev_records}
    source_available = selected_sources = recovered = 0
    details = []
    for component in shared:
        key = str(component["sample_key"])
        record = by_key.get(key)
        if record is None:
            raise ValueError(f"P4-E9 shared replay lacks dev record: {key}")
        candidate_indices = component["candidate_indices"]
        gt_indices = component["gt_indices"]
        if len(candidate_indices) != 1 or len(gt_indices) != 2:
            raise ValueError("P4-E9 shared replay component shape differs")
        top1024_index = int(candidate_indices[0])
        if top1024_index >= len(record["top1024_source_ids"]):
            raise ValueError("P4-E9 shared replay top1024 index is unavailable")
        source_id = int(record["top1024_source_ids"][top1024_index])
        base_rows = torch.where(record["source_ids"] == source_id)[0]
        available = len(base_rows) == 1
        source_available += int(available)
        selected = False
        current_recovered = False
        if available:
            base_index = int(base_rows[0])
            selected = base_index in _selected_indices(record["logits"], threshold)
            selected_sources += int(selected)
            expected = record["target_anchors"][gt_indices]
            candidate = record["top1024_anchors"][top1024_index : top1024_index + 1]
            assignments = maximum_cardinality_assignments(
                candidate, expected, radius_pixels=MATCH_RADIUS_PIXELS
            )
            unmatched = [index for index, value in enumerate(assignments) if value is None]
            if len(unmatched) != 1:
                raise RuntimeError("P4-E9 shared replay lacks one canonical secondary GT")
            addition = (
                record["base_anchors"][base_index]
                + record["offsets"][base_index]
            ).clamp(0.0, 1.0)
            distance = float(
                torch.linalg.vector_norm(
                    (addition - expected[unmatched[0]]) * 768.0
                )
            )
            current_recovered = selected and distance <= MATCH_RADIUS_PIXELS
            recovered += int(current_recovered)
        details.append(
            {
                "sample_key": key,
                "top1024_candidate_index": top1024_index,
                "source_id": source_id,
                "source_available_top128": available,
                "source_selected": selected,
                "recovered": current_recovered,
            }
        )
    return {
        "components": len(shared),
        "source_available_top128": source_available,
        "selected_sources": selected_sources,
        "recovered": recovered,
        "minimum_recovered": MINIMUM_SHARED_SPLIT_RECOVERY,
        "gate": recovered >= MINIMUM_SHARED_SPLIT_RECOVERY,
        "details": details,
    }


def route_capability_decision(
    route: str,
    calibration: dict,
    holdout: dict,
    dev: dict,
    *,
    repeat_deterministic: bool,
    shared_split: dict | None,
) -> tuple[str, dict[str, bool]]:
    if route not in ROUTE_PROTOCOL:
        raise ValueError(f"Unknown P4-E9 decision route: {route}")

    def recovery_half(metrics: dict) -> bool:
        return metrics["oracle_actions"] > 0 and (
            metrics["recovered_oracle"] / metrics["oracle_actions"]
            >= MINIMUM_ORACLE_RECOVERY
        )

    checks = {
        "calibration_zero_useless": bool(calibration["zero_useless"]),
        "holdout_zero_useless": bool(holdout["zero_useless"]),
        "dev_zero_useless": bool(dev["zero_useless"]),
        "calibration_recover_oracle_min50pct": recovery_half(calibration),
        "holdout_recover_oracle_min50pct": recovery_half(holdout),
        "dev_recover_oracle_min50pct": recovery_half(dev),
        "dev_repeat_deterministic": repeat_deterministic,
    }
    if route == "split":
        checks.update(
            {
                "p4e8_shared_recovered_min14": bool(
                    shared_split and shared_split["gate"]
                ),
                "dev_top128_matched_net_gain": dev["matched_gain"] > 0,
                "dev_top128_exact_net_gain": dev["exact_gain"] > 0,
            }
        )
    decision = (
        f"{route}_local_recovery_pilot_go"
        if all(checks.values())
        else f"{route}_local_recovery_pilot_no_go"
    )
    return decision, checks


def _validate_sha(value: str, name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must contain 64 hex characters")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error


def load_route_checkpoint(
    route: str, checkpoint_sha256: str, oracle_audit_sha256: str, device: torch.device
) -> tuple[LocalAtomRecoveryHead, dict, Path]:
    protocol = ROUTE_PROTOCOL[route]
    checkpoint = Path("outputs/checkpoints") / protocol["model_type"] / "step-005000.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"P4-E9 checkpoint is missing: {checkpoint}")
    if file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("P4-E9 checkpoint SHA256 differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("checkpoint_schema_version")
        != "g3-p4e9-local-recovery-pilot20k-0.1"
        or payload.get("model_type") != protocol["model_type"]
        or payload.get("route") != route
        or payload.get("oracle_route") != protocol["oracle_route"]
        or payload.get("lineage", {}).get("oracle_audit_sha256")
        != oracle_audit_sha256
    ):
        raise ValueError("P4-E9 checkpoint protocol or lineage differs")
    model = LocalAtomRecoveryHead(
        LocalAtomRecoveryConfig(**payload["config"])
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, payload, checkpoint


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True, choices=tuple(ROUTE_PROTOCOL))
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--oracle-audit-sha256", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    _validate_sha(args.checkpoint_sha256, "checkpoint SHA256")
    _validate_sha(args.oracle_audit_sha256, "Oracle audit SHA256")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P4-E9 online evaluation requires CUDA")
    required = (
        ORACLE_AUDIT,
        ORACLE_ACTIONS,
        ORACLE_IMAGES,
        ORACLE_CALIBRATION,
        ORACLE_HOLDOUT,
        P4E8_AUDIT,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P4-E9 evaluation inputs are incomplete: {missing}")
    if file_sha256(ORACLE_AUDIT) != args.oracle_audit_sha256:
        raise ValueError("P4-E9 Oracle audit SHA256 differs")
    if file_sha256(P4E8_AUDIT) != P4E8_AUDIT_SHA256:
        raise ValueError("P4-E9 P4-E8 audit SHA256 differs")
    oracle_audit = json.loads(ORACLE_AUDIT.read_text(encoding="utf-8"))
    if oracle_audit.get("decision") != "recovery_oracle_go" or not all(
        oracle_audit.get("protocol_checks", {}).values()
    ):
        raise ValueError("P4-E9 Oracle has not passed its protocol gate")
    oracle_route = ROUTE_PROTOCOL[args.route]["oracle_route"]
    if not oracle_audit.get("route_capacity", {}).get(oracle_route, False):
        raise ValueError(f"P4-E9 Oracle route is unsupported: {oracle_route}")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model, checkpoint_payload, checkpoint = load_route_checkpoint(
        args.route,
        args.checkpoint_sha256,
        args.oracle_audit_sha256,
        device,
    )

    output_dir = checkpoint.parent
    benchmark_path = output_dir / "benchmark.json"
    final_path = output_dir / "final_metrics.json"
    if benchmark_path.exists() or final_path.exists():
        raise RuntimeError("P4-E9 online evaluation outputs already exist")
    checkpoint_audit_path = output_dir / "audit.json"
    if not checkpoint_audit_path.is_file():
        raise FileNotFoundError("P4-E9 checkpoint audit is missing")
    checkpoint_audit = json.loads(
        checkpoint_audit_path.read_text(encoding="utf-8")
    )
    if (
        checkpoint_audit.get("model_type")
        != ROUTE_PROTOCOL[args.route]["model_type"]
        or checkpoint_audit.get("route") != args.route
        or checkpoint_audit.get("oracle_route")
        != ROUTE_PROTOCOL[args.route]["oracle_route"]
        or checkpoint_audit.get("checkpoint", {}).get("sha256")
        != args.checkpoint_sha256
        or checkpoint_audit.get("lineage", {}).get("oracle_audit_sha256")
        != args.oracle_audit_sha256
    ):
        raise ValueError("P4-E9 checkpoint audit identity or lineage differs")

    # Heavy structural dependencies are loaded only for an actual online run.
    from chemvector.evaluation.audits.atoms import (
        audit_g3_p4e6_missing_peak_sources as p4e6,
    )
    from chemvector.evaluation.audits.atoms import (
        build_g3_p3e21_shared_cache as p3e21,
    )
    from chemvector.training.graph.scenes.g3_structural_data import (
        load_g3_sample_manifest,
    )
    from chemvector.evaluation import g3_p3e21_cycle as p3_cycle
    from chemvector.evaluation.evaluators.atoms.g3_p4_local_recovery import (
        calibrate_zero_useless_threshold,
        evaluate_threshold,
    )

    if (
        checkpoint_payload.get("lineage", {}).get("p4e3_checkpoint_sha256")
        != p4e6.P4E3_CHECKPOINT_SHA256
        or oracle_audit.get("lineage", {}).get("p4e3_checkpoint_sha256")
        != p4e6.P4E3_CHECKPOINT_SHA256
    ):
        raise ValueError("P4-E9 frozen P4-E3 lineage differs")

    calibration_manifest, _ = _manifest(ORACLE_CALIBRATION)
    holdout_manifest, _ = _manifest(ORACLE_HOLDOUT)
    dev_manifest, dev_manifest_audit = load_g3_sample_manifest(
        p3e21.MANIFEST_ROOT / "dev.jsonl",
        expected_sha256=p3e21.MANIFESTS["dev"][0],
        expected_samples=p3e21.MANIFESTS["dev"][1],
        expected_molecules=p3e21.MANIFESTS["dev"][2],
        expected_variants=p3_cycle.VARIANTS,
    )
    actions = _jsonl(ORACLE_ACTIONS)
    calibration_actions = _actions_by_sample(actions, args.route, "calibration")
    holdout_actions = _actions_by_sample(actions, args.route, "holdout")
    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    started = time.perf_counter()
    calibration_records, calibration_stream = collect_online_records(
        calibration_manifest,
        route=args.route,
        route_actions=calibration_actions,
        dynamic_oracle=False,
        include_top1024=False,
        model=model,
        field=field,
        atom_head=atom_head,
        vision=vision,
        frozen_block=frozen_block,
        device=device,
    )
    threshold, calibration_audit = calibrate_zero_useless_threshold(
        calibration_records
    )
    holdout_records, holdout_stream = collect_online_records(
        holdout_manifest,
        route=args.route,
        route_actions=holdout_actions,
        dynamic_oracle=False,
        include_top1024=False,
        model=model,
        field=field,
        atom_head=atom_head,
        vision=vision,
        frozen_block=frozen_block,
        device=device,
    )
    holdout_metrics = evaluate_threshold(holdout_records, threshold)
    dev_records, dev_stream = collect_online_records(
        dev_manifest,
        route=args.route,
        route_actions=None,
        dynamic_oracle=True,
        include_top1024=args.route == "split",
        model=model,
        field=field,
        atom_head=atom_head,
        vision=vision,
        frozen_block=frozen_block,
        device=device,
    )
    dev_metrics = evaluate_threshold(dev_records, threshold)
    repeated_dev_records, repeated_dev_stream = collect_online_records(
        dev_manifest,
        route=args.route,
        route_actions=None,
        dynamic_oracle=True,
        include_top1024=args.route == "split",
        model=model,
        field=field,
        atom_head=atom_head,
        vision=vision,
        frozen_block=frozen_block,
        device=device,
    )
    repeated_dev_metrics = evaluate_threshold(repeated_dev_records, threshold)
    repeat_deterministic = (
        dev_stream["records_sha256"] == repeated_dev_stream["records_sha256"]
        and dev_metrics == repeated_dev_metrics
    )
    shared_split = None
    if args.route == "split":
        p4e8 = json.loads(P4E8_AUDIT.read_text(encoding="utf-8"))
        shared_split = replay_shared_split_components(
            dev_records, p4e8["components"]["6"], threshold
        )
    decision, checks = route_capability_decision(
        args.route,
        calibration_audit["metrics"],
        holdout_metrics,
        dev_metrics,
        repeat_deterministic=repeat_deterministic,
        shared_split=shared_split,
    )

    benchmark = {
        "schema_version": "g3-p4e9-local-recovery-benchmark-0.1",
        "model_type": ROUTE_PROTOCOL[args.route]["model_type"],
        "route": args.route,
        "threshold": threshold,
        "calibration": calibration_audit,
        "holdout": holdout_metrics,
        "dev": dev_metrics,
        "repeated_dev": repeated_dev_metrics,
        "repeat_deterministic": repeat_deterministic,
        "shared_split": shared_split,
        "streams": {
            "calibration": calibration_stream,
            "holdout": holdout_stream,
            "dev": dev_stream,
            "repeated_dev": repeated_dev_stream,
        },
    }
    _atomic_json(benchmark_path, benchmark)
    final = {
        "schema_version": "g3-p4e9-local-recovery-final-metrics-0.1",
        "decision": decision,
        "checks": checks,
        "model_type": ROUTE_PROTOCOL[args.route]["model_type"],
        "route": args.route,
        "threshold": threshold,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "oracle_audit_sha256": args.oracle_audit_sha256,
        "benchmark": str(benchmark_path.resolve()),
        "benchmark_sha256": file_sha256(benchmark_path),
        "vision_checkpoint_sha256": vision_sha,
        "dev_manifest": dev_manifest_audit,
        "elapsed_seconds": time.perf_counter() - started,
        "confirm_v2_read": False,
        "tag_created": False,
    }
    _atomic_json(final_path, final)

    checkpoint_audit.setdefault("protocol", {})[
        "calibration_holdout_dev_benchmark_run"
    ] = True
    checkpoint_audit.update(
        {
            "status": "evaluated",
            "decision": decision,
            "threshold": threshold,
            "online_evaluation": {
                "benchmark": str(benchmark_path.resolve()),
                "benchmark_sha256": file_sha256(benchmark_path),
                "final_metrics": str(final_path.resolve()),
                "final_metrics_sha256": file_sha256(final_path),
                "checks": checks,
                "confirm_v2_read": False,
                "tag_created": False,
            },
        }
    )
    _atomic_json(checkpoint_audit_path, checkpoint_audit)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
