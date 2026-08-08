#!/usr/bin/env python3
"""Build the P4-E9 fit cache and train one route-specific local recovery pilot."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np
import torch

from chemvector.components.recovery.local_atom_recovery import (
    LocalAtomRecoveryHead,
)
from chemvector.training.atoms.g3_p4_local_recovery_data import (
    CANDIDATES,
    build_local_recovery_observations,
    build_local_recovery_targets,
    deterministic_recovery_sample_mask,
    local_recovery_loss,
)


ORACLE_ROOT = Path(
    "outputs/audits/chemvector-g3-p4e9-recovery-oracle-pilot20k"
)
ORACLE_AUDIT = ORACLE_ROOT / "audit.json"
ORACLE_ACTIONS = ORACLE_ROOT / "actions.jsonl"
ORACLE_IMAGES = ORACLE_ROOT / "images.jsonl"
ORACLE_FIT = ORACLE_ROOT / "fit.jsonl"
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
CAPACITY_POSITIVES = 128
CAPACITY_NEGATIVES = 384
CAPACITY_STEPS = 2_000
PILOT_STEPS = 5_000
BATCH_SIZE = 128
POSITIVE_BATCH_SIZE = 32
NEGATIVE_BATCH_SIZE = 96
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
CACHE_BATCH_SIZE = 8
SEED = 20260850


@dataclass(frozen=True, slots=True)
class FitManifestRecord:
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


def tensor_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_oracle(
    expected_sha256: str, route: str
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    if route not in ROUTE_PROTOCOL:
        raise ValueError(f"Unknown P4-E9 recovery route: {route}")
    if len(expected_sha256) != 64:
        raise ValueError("P4-E9 Oracle audit SHA256 must contain 64 hex characters")
    try:
        int(expected_sha256, 16)
    except ValueError as error:
        raise ValueError("P4-E9 Oracle audit SHA256 must be hexadecimal") from error
    required = (ORACLE_AUDIT, ORACLE_ACTIONS, ORACLE_IMAGES, ORACLE_FIT)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P4-E9 Oracle outputs are incomplete: {missing}")
    actual_sha256 = file_sha256(ORACLE_AUDIT)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"P4-E9 Oracle audit SHA256 differs: {actual_sha256} != {expected_sha256}"
        )
    audit = json.loads(ORACLE_AUDIT.read_text(encoding="utf-8"))
    if audit.get("decision") != "recovery_oracle_go":
        raise ValueError("P4-E9 Oracle has not passed its protocol gate")
    oracle_route = ROUTE_PROTOCOL[route]["oracle_route"]
    if not bool(audit.get("route_capacity", {}).get(oracle_route, False)):
        raise ValueError(f"P4-E9 Oracle route is unsupported: {oracle_route}")
    checks = audit.get("protocol_checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("P4-E9 Oracle protocol checks are incomplete or failed")
    actions = _jsonl(ORACLE_ACTIONS)
    images = _jsonl(ORACLE_IMAGES)
    fit = _jsonl(ORACLE_FIT)
    return audit, actions, images, fit


def _selected_fit_inputs(
    route: str,
    actions: list[dict],
    images: list[dict],
    fit: list[dict],
) -> tuple[list[FitManifestRecord], dict[str, list[dict]], dict[str, set[int]], dict]:
    oracle_route = ROUTE_PROTOCOL[route]["oracle_route"]
    fit_keys = {str(row["sample_key"]) for row in fit}
    if len(fit_keys) != len(fit):
        raise ValueError("P4-E9 fit manifest contains duplicate sample keys")
    images_by_key = {
        str(row["sample_key"]): row
        for row in images
        if str(row.get("split")) == "fit"
    }
    if set(images_by_key) != fit_keys:
        raise ValueError("P4-E9 fit manifest and image Oracle identities differ")
    route_actions = [
        row
        for row in actions
        if str(row.get("split")) == "fit"
        and str(row.get("route")) == oracle_route
    ]
    actions_by_key: dict[str, list[dict]] = {}
    for row in route_actions:
        key = str(row["sample_key"])
        if key not in fit_keys:
            raise ValueError("P4-E9 fit action references a non-fit sample")
        actions_by_key.setdefault(key, []).append(row)
    if len(route_actions) < CAPACITY_POSITIVES:
        raise ValueError("P4-E9 fit route lacks 128 capacity positives")
    unsupported_by_key = {}
    for key, row in images_by_key.items():
        values = row.get("unsupported_source_ids")
        if not isinstance(values, list):
            raise ValueError("P4-E9 image Oracle lacks explicit unsupported sources")
        unsupported_by_key[key] = {int(value) for value in values}
    metadata = {str(row["sample_key"]): row for row in fit}
    selected_keys = set(actions_by_key)
    manifest = []
    for row in fit:
        key = str(row["sample_key"])
        if key not in selected_keys:
            continue
        required = {"sample_key", "molecule_id", "inchikey", "shard", "variant"}
        if required - row.keys():
            raise ValueError("P4-E9 fit manifest row lacks source identity fields")
        manifest.append(
            FitManifestRecord(
                sample_key=key,
                molecule_id=str(row["molecule_id"]),
                inchikey=str(row["inchikey"]),
                shard=str(row["shard"]),
                variant=str(row["variant"]),
            )
        )
    if len(manifest) != len(selected_keys):
        raise RuntimeError("P4-E9 selected fit manifest is incomplete")
    return manifest, actions_by_key, unsupported_by_key, {
        "fit_images": len(fit),
        "selected_positive_images": len(manifest),
        "route_actions": len(route_actions),
        "molecules": len({metadata[key]["molecule_id"] for key in selected_keys}),
        "variants": sorted({metadata[key]["variant"] for key in selected_keys}),
    }


def _unsupported_mask(
    source_ids: torch.Tensor, unsupported: Sequence[set[int]]
) -> torch.Tensor:
    if source_ids.ndim != 2 or source_ids.size(0) != len(unsupported):
        raise ValueError("P4-E9 unsupported mask inputs differ")
    mask = torch.ones_like(source_ids, dtype=torch.bool)
    for batch_index, values in enumerate(unsupported):
        if values:
            blocked = source_ids.new_tensor(sorted(values))
            mask[batch_index] = ~torch.isin(source_ids[batch_index], blocked)
    return mask


def _append_cache_rows(
    cache: dict[str, list],
    observations: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    candidate_mask: torch.Tensor,
    sample_keys: Sequence[str],
    route: str,
) -> None:
    prefix = route
    action_targets = targets[f"{prefix}_targets"]
    offset_targets = targets[f"{prefix}_offsets"]
    for batch_index, sample_key in enumerate(sample_keys):
        current_targets = action_targets[batch_index : batch_index + 1]
        current_mask = candidate_mask[batch_index : batch_index + 1]
        current_ids = observations["candidate_source_ids"][
            batch_index : batch_index + 1
        ]
        selected = deterministic_recovery_sample_mask(
            current_targets,
            current_mask,
            current_ids,
            [sample_key],
            seed=SEED,
            round_index=0,
        )[0]
        rows = torch.where(selected)[0]
        if int(current_targets.sum()) < 1 or len(rows) != 4 * int(current_targets.sum()):
            raise RuntimeError("P4-E9 per-image cache sampling does not satisfy 1:3")
        for name in (
            "image_crops",
            "score_crops",
            "candidate_features",
            "candidate_scores",
            "candidate_ranks",
        ):
            cache[name].append(observations[name][batch_index, rows].half().cpu())
        cache["action_targets"].append(action_targets[batch_index, rows].cpu())
        cache["offset_targets"].append(offset_targets[batch_index, rows].float().cpu())
        cache["source_ids"].append(
            observations["candidate_source_ids"][batch_index, rows].cpu()
        )
        cache["sample_keys"].extend([sample_key] * len(rows))


@torch.no_grad()
def build_training_cache(
    route: str,
    manifest: Sequence[FitManifestRecord],
    actions_by_key: dict[str, list[dict]],
    unsupported_by_key: dict[str, set[int]],
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict]:
    """Run frozen P4-E3 only on fit images containing this route's positives."""

    # Heavy structural imports remain local so protocol/checkpoint inspection does
    # not require the RDKit training environment.
    from chemvector.components.atoms.objects.g3_atom_center_field import (
        decode_atom_center_field,
    )
    from chemvector.evaluation.audits.atoms.audit_g3_p1n_regional_maxima import (
        extract_regional_maximum_objects,
    )
    from chemvector.evaluation.audits.atoms import (
        audit_g3_p4e6_missing_peak_sources as p4e6,
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

    field, atom_head, vision, frozen_block, vision_sha = p4e6._load_model(device)
    collator = p1a.G3AtomCollator(maximum_atoms=CANDIDATES, image_size=768)
    cache: dict[str, list] = {
        "image_crops": [],
        "score_crops": [],
        "candidate_features": [],
        "candidate_scores": [],
        "candidate_ranks": [],
        "action_targets": [],
        "offset_targets": [],
        "source_ids": [],
        "sample_keys": [],
    }
    started = time.perf_counter()
    for batch in p1a.iter_manifest_batches(
        list(manifest),
        collator,
        batch_size=CACHE_BATCH_SIZE,
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
        peaks = extract_regional_maximum_objects(
            decode_atom_center_field(field_output), maximum_objects=CANDIDATES
        )
        observations = build_local_recovery_observations(
            batch["pixel_values"].to(device), field_output, peaks
        )
        current_actions = [
            action
            for key in batch["keys"]
            for action in actions_by_key[str(key)]
        ]
        targets = build_local_recovery_targets(
            [str(key) for key in batch["keys"]],
            observations["candidate_anchors"],
            observations["candidate_source_ids"],
            observations["candidate_mask"],
            current_actions,
        )
        supported = _unsupported_mask(
            observations["candidate_source_ids"],
            [unsupported_by_key[str(key)] for key in batch["keys"]],
        )
        candidate_mask = observations["candidate_mask"] & supported
        prefix = route
        if bool((targets[f"{prefix}_targets"] & ~candidate_mask).any()):
            raise RuntimeError("P4-E9 Oracle marked a positive source unsupported")
        _append_cache_rows(
            cache,
            observations,
            targets,
            candidate_mask,
            [str(key) for key in batch["keys"]],
            route,
        )
    tensor_names = (
        "image_crops",
        "score_crops",
        "candidate_features",
        "candidate_scores",
        "candidate_ranks",
        "action_targets",
        "offset_targets",
        "source_ids",
    )
    payload: dict[str, object] = {
        name: torch.cat(cache[name], dim=0) for name in tensor_names
    }
    payload["sample_keys"] = cache["sample_keys"]
    rows = len(payload["sample_keys"])
    positives = int(payload["action_targets"].sum())
    negatives = rows - positives
    if positives < CAPACITY_POSITIVES or negatives != 3 * positives:
        raise RuntimeError("P4-E9 training cache lacks capacity or fixed 1:3 balance")
    digest = hashlib.sha256()
    for name in tensor_names:
        value = payload[name].contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    for key in payload["sample_keys"]:
        digest.update(f"{key}\n".encode("utf-8"))
    return payload, {
        "rows": rows,
        "positives": positives,
        "negatives": negatives,
        "images": len(set(payload["sample_keys"])),
        "unsupported_masked": sum(len(unsupported_by_key[item.sample_key]) for item in manifest),
        "sha256": digest.hexdigest(),
        "vision_checkpoint_sha256": vision_sha,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _hash_ranked_indices(cache: dict[str, object], positive: bool) -> list[int]:
    targets = cache["action_targets"]
    indices = torch.where(targets if positive else ~targets)[0].tolist()
    return sorted(
        indices,
        key=lambda index: (
            hashlib.sha256(
                f"{SEED}:{cache['sample_keys'][index]}:{int(cache['source_ids'][index])}"
                .encode("utf-8")
            ).digest(),
            str(cache["sample_keys"][index]),
            int(cache["source_ids"][index]),
        ),
    )


def capacity_subset(cache: dict[str, object]) -> dict[str, object]:
    positives = _hash_ranked_indices(cache, True)[:CAPACITY_POSITIVES]
    negatives = _hash_ranked_indices(cache, False)[:CAPACITY_NEGATIVES]
    if len(positives) != CAPACITY_POSITIVES or len(negatives) != CAPACITY_NEGATIVES:
        raise ValueError("P4-E9 cache cannot form the fixed capacity subset")
    indices = torch.tensor([*positives, *negatives], dtype=torch.long)
    output = {}
    row_count = len(cache["action_targets"])
    for name, value in cache.items():
        if isinstance(value, torch.Tensor) and value.size(0) == row_count:
            output[name] = value[indices]
        elif isinstance(value, list) and len(value) == row_count:
            output[name] = [value[index] for index in indices.tolist()]
        else:
            output[name] = value
    return output


def balanced_batch_schedule(
    cache: dict[str, object], *, steps: int, seed: int
) -> list[torch.Tensor]:
    if steps < 1:
        raise ValueError("P4-E9 training schedule must contain steps")
    positives = torch.where(cache["action_targets"])[0]
    negatives = torch.where(~cache["action_targets"])[0]
    if len(positives) < POSITIVE_BATCH_SIZE or len(negatives) < NEGATIVE_BATCH_SIZE:
        raise ValueError("P4-E9 cache is too small for a balanced batch")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    positive_order = negative_order = torch.empty(0, dtype=torch.long)
    positive_cursor = negative_cursor = 0

    def take(pool, order, cursor, count):
        values = []
        while len(values) < count:
            if cursor == len(order):
                order = pool[torch.randperm(len(pool), generator=generator)]
                cursor = 0
            amount = min(count - len(values), len(order) - cursor)
            values.extend(order[cursor : cursor + amount].tolist())
            cursor += amount
        return torch.tensor(values, dtype=torch.long), order, cursor

    batches = []
    for _ in range(steps):
        positive, positive_order, positive_cursor = take(
            positives,
            positive_order,
            positive_cursor,
            POSITIVE_BATCH_SIZE,
        )
        negative, negative_order, negative_cursor = take(
            negatives,
            negative_order,
            negative_cursor,
            NEGATIVE_BATCH_SIZE,
        )
        combined = torch.cat((positive, negative))
        combined = combined[torch.randperm(BATCH_SIZE, generator=generator)]
        batches.append(combined)
    return batches


def route_parameters(model: LocalAtomRecoveryHead, route: str) -> list[torch.nn.Parameter]:
    if route not in ROUTE_PROTOCOL:
        raise ValueError(f"Unknown P4-E9 route: {route}")
    other = "refine" if route == "split" else "split"
    getattr(model, f"{other}_action").requires_grad_(False)
    getattr(model, f"{other}_offset").requires_grad_(False)
    parameters = [value for value in model.parameters() if value.requires_grad]
    if not parameters:
        raise RuntimeError("P4-E9 route has no trainable parameters")
    return parameters


def _cache_batch(
    cache: dict[str, object], indices: torch.Tensor, device: torch.device
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor], torch.Tensor]:
    if indices.shape != (BATCH_SIZE,):
        raise ValueError("P4-E9 optimizer batch must contain exactly 128 actions")
    model_inputs = tuple(
        cache[name][indices].to(device).float().unsqueeze(0)
        for name in (
            "image_crops",
            "score_crops",
            "candidate_features",
            "candidate_scores",
            "candidate_ranks",
        )
    )
    candidate_mask = torch.ones((1, BATCH_SIZE), device=device, dtype=torch.bool)
    model_inputs = (*model_inputs, candidate_mask)
    prefix = "split" if cache["route"] == "split" else "refine"
    action_targets = cache["action_targets"][indices].to(device).unsqueeze(0)
    offset_targets = cache["offset_targets"][indices].to(device).unsqueeze(0)
    targets = {
        f"{prefix}_targets": action_targets,
        f"{prefix}_offsets": offset_targets,
        "candidate_mask": candidate_mask,
    }
    return model_inputs, targets, candidate_mask


def _autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def train_phase(
    model: LocalAtomRecoveryHead,
    cache: dict[str, object],
    *,
    route: str,
    steps: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, dict, dict]:
    parameters = route_parameters(model, route)
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    schedule = balanced_batch_schedule(cache, steps=steps, seed=seed)
    losses_sum = {"loss": 0.0, "action_loss": 0.0, "offset_loss": 0.0}
    last = {}
    model.train()
    for step, indices in enumerate(schedule, start=1):
        optimizer.zero_grad(set_to_none=True)
        inputs, targets, sample_mask = _cache_batch(cache, indices, device)
        with _autocast(device):
            output = model(*inputs)
            losses = local_recovery_loss(
                output,
                targets,
                sample_mask,
                route=ROUTE_PROTOCOL[route]["oracle_route"],
            )
        losses["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("P4-E9 gradient norm is not finite")
        optimizer.step()
        last = {
            **{name: float(value.detach()) for name, value in losses.items()},
            "gradient_norm": float(gradient_norm.detach()),
            "step": step,
        }
        for name in losses_sum:
            losses_sum[name] += float(losses[name].detach())
        if step % 250 == 0 or step == steps:
            print(
                json.dumps(
                    {
                        "phase_steps": steps,
                        "step": step,
                        "loss": last["loss"],
                        "action_loss": last["action_loss"],
                        "offset_loss": last["offset_loss"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = {
        "steps": steps,
        "exposure": steps * BATCH_SIZE,
        "mean_losses": {name: value / steps for name, value in losses_sum.items()},
        "last": last,
        "schedule_sha256": hashlib.sha256(
            torch.stack(schedule).numpy().tobytes()
        ).hexdigest(),
    }
    return optimizer, last, summary


@torch.no_grad()
def evaluate_cache(
    model: LocalAtomRecoveryHead,
    cache: dict[str, object],
    *,
    route: str,
    device: torch.device,
) -> dict:
    model.eval()
    prefix = route
    true_positive = false_positive = true_negative = false_negative = 0
    offset_error = offset_count = 0
    for start in range(0, len(cache["action_targets"]), BATCH_SIZE):
        rows = torch.arange(start, min(start + BATCH_SIZE, len(cache["action_targets"])))
        if len(rows) < BATCH_SIZE:
            padding = rows.new_full((BATCH_SIZE - len(rows),), int(rows[-1]))
            rows = torch.cat((rows, padding))
        inputs, targets, _ = _cache_batch(cache, rows, device)
        with _autocast(device):
            output = model(*inputs)
        count = min(BATCH_SIZE, len(cache["action_targets"]) - start)
        expected = targets[f"{prefix}_targets"][0, :count]
        predicted = output[f"{prefix}_logits"][0, :count] >= 0.0
        true_positive += int((predicted & expected).sum())
        false_positive += int((predicted & ~expected).sum())
        true_negative += int((~predicted & ~expected).sum())
        false_negative += int((~predicted & expected).sum())
        if bool(expected.any()):
            difference = (
                output[f"{prefix}_offsets"][0, :count][expected]
                - targets[f"{prefix}_offsets"][0, :count][expected]
            )
            offset_error += float(
                torch.linalg.vector_norm(difference.float() * 768.0, dim=-1).sum()
            )
            offset_count += int(expected.sum())
    return {
        "rows": len(cache["action_targets"]),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "accuracy": (true_positive + true_negative)
        / max(true_positive + false_positive + true_negative + false_negative, 1),
        "positive_recall": true_positive / max(true_positive + false_negative, 1),
        "offset_mae_pixels": offset_error / max(offset_count, 1),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True, choices=tuple(ROUTE_PROTOCOL))
    parser.add_argument("--oracle-audit-sha256", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P4-E9 pilot training requires CUDA")
    route_protocol = ROUTE_PROTOCOL[args.route]
    output_dir = Path("outputs/checkpoints") / route_protocol["model_type"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"P4-E9 output directory is not empty: {output_dir}")
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

    oracle_audit, actions, images, fit = validate_oracle(
        args.oracle_audit_sha256, args.route
    )
    manifest, actions_by_key, unsupported_by_key, selection_audit = (
        _selected_fit_inputs(args.route, actions, images, fit)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cache, cache_audit = build_training_cache(
        args.route,
        manifest,
        actions_by_key,
        unsupported_by_key,
        device=device,
    )
    cache["route"] = args.route
    cache_path = output_dir / "fit-cache.pt"
    _atomic_torch_save(cache_path, cache)

    capacity = capacity_subset(cache)
    capacity["route"] = args.route
    torch.manual_seed(SEED)
    capacity_model = LocalAtomRecoveryHead().to(device)
    capacity_initial_sha = tensor_state_sha256(capacity_model)
    _, _, capacity_training = train_phase(
        capacity_model,
        capacity,
        route=args.route,
        steps=CAPACITY_STEPS,
        seed=SEED + 1,
        device=device,
    )
    capacity_final_sha = tensor_state_sha256(capacity_model)
    capacity_metrics = evaluate_cache(
        capacity_model, capacity, route=args.route, device=device
    )
    del capacity_model

    # The pilot is deliberately reset instead of warm-starting from capacity.
    torch.manual_seed(SEED)
    model = LocalAtomRecoveryHead().to(device)
    pilot_initial_sha = tensor_state_sha256(model)
    optimizer, last_losses, pilot_training = train_phase(
        model,
        cache,
        route=args.route,
        steps=PILOT_STEPS,
        seed=SEED + 2,
        device=device,
    )
    fit_metrics = evaluate_cache(model, cache, route=args.route, device=device)
    reset_check = pilot_initial_sha == capacity_initial_sha
    if not reset_check or capacity_final_sha == pilot_initial_sha:
        raise RuntimeError("P4-E9 capacity-to-pilot reset protocol failed")

    checkpoint_payload = {
        "checkpoint_schema_version": "g3-p4e9-local-recovery-pilot20k-0.1",
        "model_type": route_protocol["model_type"],
        "route": args.route,
        "oracle_route": route_protocol["oracle_route"],
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": PILOT_STEPS,
        "samples_seen": PILOT_STEPS * BATCH_SIZE,
        "training_args": {
            "batch_size": BATCH_SIZE,
            "positive_batch_size": POSITIVE_BATCH_SIZE,
            "negative_batch_size": NEGATIVE_BATCH_SIZE,
            "steps": PILOT_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "seed": SEED,
        },
        "last_losses": last_losses,
        "fit_metrics": fit_metrics,
        "lineage": {
            "oracle_audit": str(ORACLE_AUDIT.resolve()),
            "oracle_audit_sha256": args.oracle_audit_sha256,
            "p4e3_checkpoint": oracle_audit["lineage"]["p4e3_checkpoint"],
            "p4e3_checkpoint_sha256": oracle_audit["lineage"][
                "p4e3_checkpoint_sha256"
            ],
            "fit_cache": str(cache_path.resolve()),
            "fit_cache_sha256": file_sha256(cache_path),
            "observation_cache_sha256": cache_audit["sha256"],
        },
        "status": "trained_pending_calibration_holdout_dev_benchmark",
    }
    checkpoint = output_dir / "step-005000.pt"
    _atomic_torch_save(checkpoint, checkpoint_payload)
    _atomic_torch_save(output_dir / "last.pt", checkpoint_payload)
    audit = {
        "schema_version": "g3-p4e9-local-recovery-pilot20k-audit-0.1",
        "status": "trained_pending_online_benchmark",
        "model_type": route_protocol["model_type"],
        "route": args.route,
        "oracle_route": route_protocol["oracle_route"],
        "protocol": {
            "capacity": {
                "positives": CAPACITY_POSITIVES,
                "negatives": CAPACITY_NEGATIVES,
                "steps": CAPACITY_STEPS,
            },
            "pilot": {
                "steps": PILOT_STEPS,
                "batch_size": BATCH_SIZE,
                "positive_negative_ratio": "1:3",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "max_grad_norm": MAX_GRAD_NORM,
            },
            "capacity_reset_before_pilot": reset_check,
            "unsupported_sources_loss_masked": True,
            "fit_only": True,
            "calibration_holdout_dev_benchmark_run": False,
            "confirm_v2_read": False,
            "tag_created": False,
        },
        "selection": selection_audit,
        "cache": {
            **cache_audit,
            "path": str(cache_path.resolve()),
            "file_sha256": file_sha256(cache_path),
        },
        "capacity": {
            "initial_state_sha256": capacity_initial_sha,
            "final_state_sha256": capacity_final_sha,
            "training": capacity_training,
            "metrics": capacity_metrics,
        },
        "pilot": {
            "initial_state_sha256": pilot_initial_sha,
            "final_state_sha256": tensor_state_sha256(model),
            "training": pilot_training,
            "fit_metrics": fit_metrics,
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": file_sha256(checkpoint),
        },
        "lineage": checkpoint_payload["lineage"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(output_dir / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
