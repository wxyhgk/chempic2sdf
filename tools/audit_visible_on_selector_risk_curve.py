#!/usr/bin/env python3
"""Read-only selector retention risk curves on the visible O/N pilot.

The frozen initial forward produces raw AtomObject proposals and set-head
presence logits.  This audit replays those predictions under fixed,
target-free selector threshold offsets and fixed top-K retention policies.
Targets are joined only after each policy mask is formed, for reporting.  No
weights, thresholds, or checkpoints are fitted or changed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.core.data import ChemVectorIterableDataset
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.audit_visible_on_frozen_reachability import (
    DATA, IMAGE_SIZE, INITIAL_SHA, MANIFEST, OUT as REACHABILITY_OUT,
    _metadata, _rotation_bucket, _target_rows, digest,
)

OUT = ROOT / os.environ.get(
    "CHEMVECTOR_SELECTOR_CURVE_OUT",
    "outputs/audits/chemvector-visible-on-selector-risk-curve-20261017-r1",
)
BATCH_SIZE = 16
STRICT_RADIUS = 6.0
REACH_RADIUS = 16.0
OTHER_TARGET_MIN = 20.0
MARGIN = 4.0
BASELINE_THRESHOLD = -0.25
THRESHOLD_OFFSETS = (-0.50, -0.25, 0.0, 0.25, 0.50, 0.75, 1.00)
TOP_K = (48, 64, 80, 96, 112, 128)


def _empty() -> Counter:
    return Counter()


def _status(candidates: torch.Tensor, targets: list[dict]) -> tuple[list[bool], list[bool], list[bool], int]:
    """Return strict/6px/16px target statuses under one candidate mask."""
    if not targets:
        return [], [], [], 0
    target_xy = torch.tensor([t["xy"] for t in targets], dtype=torch.float32)
    distance = torch.cdist(candidates.float(), target_xy)
    any6 = [(distance[:, i] <= STRICT_RADIUS).any().item() for i in range(len(targets))]
    reach16 = [(distance[:, i] <= REACH_RADIUS).any().item() for i in range(len(targets))]
    strict = [False] * len(targets)
    owners: set[int] = set()
    for ti in range(len(targets)):
        nearby = torch.where(distance[:, ti] <= STRICT_RADIUS)[0].tolist()
        if len(nearby) != 1:
            continue
        ci = nearby[0]
        own = float(distance[ci, ti])
        other = torch.cat((distance[ci, :ti], distance[ci, ti + 1:]))
        nearest_other = float(other.min()) if len(other) else float("inf")
        if nearest_other <= OTHER_TARGET_MIN or nearest_other - own < MARGIN or ci in owners:
            continue
        owners.add(ci)
        strict[ti] = True
    return strict, any6, reach16, len(owners)


def _update(counter: Counter, *, candidates: torch.Tensor, targets: list[dict],
            strict: list[bool], any6: list[bool], reach16: list[bool]) -> None:
    counter["images"] += 1
    counter["candidate_count"] += int(len(candidates))
    counter["target_count"] += len(targets)
    counter["strict_owned"] += sum(strict)
    counter["reachable_6_any"] += sum(any6)
    counter["reachable_16"] += sum(reach16)


def _rates(counter: Counter) -> dict:
    targets = max(int(counter["target_count"]), 1)
    candidates = max(int(counter["candidate_count"]), 1)
    return {
        "images": int(counter["images"]),
        "targets": int(counter["target_count"]),
        "candidates": int(counter["candidate_count"]),
        "coverage": counter["strict_owned"] / targets,
        "reachable_6_any": counter["reachable_6_any"] / targets,
        "reachable_16": counter["reachable_16"] / targets,
        "candidate_precision": counter["strict_owned"] / candidates,
        "retention_per_target": counter["candidate_count"] / max(int(counter["images"]), 1),
    }


def _target_bucket(target: dict, meta: dict, sample: dict) -> list[tuple[str, str]]:
    element = str(target["element"])
    role = str(target["role"])
    if role == "text" and element in {"O", "N"}:
        role_key = element
    elif role == "implicit" and element == "C" and int(target["degree"]) == 2:
        role_key = "implicit_C_degree_2"
    else:
        role_key = "other"
    params = sample["layers"].get("render_parameters", {})
    variant = str(meta.get("variant") or params.get("layout") or "unknown")
    return [("focus", role_key), ("profile", variant)]


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("read_only") is not True or manifest.get("training_performed") is not False:
        raise ValueError("pilot manifest is not read-only")
    if manifest.get("sealed_test_read") is not False:
        raise ValueError("pilot manifest is marked as sealed-test consuming")
    if digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("initial checkpoint changed")
    metadata = _metadata(manifest)
    expected_keys = set(metadata)
    shards = sorted(str(path) for path in (DATA / "shards").glob("train-*.tar"))
    if not shards:
        raise FileNotFoundError("no public train shards found")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval().requires_grad_(False)
    state_before = state_digest(model)
    atom_threshold = float(payload["provenance"]["atom_threshold"])
    if abs(atom_threshold - BASELINE_THRESHOLD) > 1e-9:
        raise RuntimeError(f"unexpected baseline selector threshold: {atom_threshold}")
    policies: dict[str, Counter] = {}
    strata: dict[str, Counter] = defaultdict(Counter)
    for offset in THRESHOLD_OFFSETS:
        policies[f"threshold_offset:{offset:+.2f}"] = _empty()
    for k in TOP_K:
        policies[f"top_k:{k}"] = _empty()
    # Keep split-level counters alongside the compact aggregate counters.  The
    # latter are diagnostic only; a gate must never mistake train+holdout for
    # holdout.  defaultdict is used so a future public split is retained
    # without changing the replay path.
    split_policies: dict[str, defaultdict[str, Counter]] = {
        name: defaultdict(_empty) for name in policies
    }
    seen: set[str] = set()
    batch: list[dict] = []

    def consume() -> None:
        if not batch:
            return
        images = torch.stack([image_tensor(s["image"]) for s in batch]).to(device)
        with torch.inference_mode():
            outputs = structure_forward(model, images, config, atom_threshold)
        raw = outputs[3]
        set_output = outputs[5]
        for index, sample in enumerate(batch):
            key = str(sample["key"])
            meta = metadata[key]
            targets = _target_rows(sample)
            raw_mask = raw["atom_mask"][index].bool()
            raw_geometry = raw["atom_geometry"][index, :, :2].detach().cpu() * IMAGE_SIZE
            raw_scores = raw["atom_scores"][index].detach().cpu()
            presence = set_output["atom_presence_logits"][index].detach().cpu()
            active = torch.where(raw_mask)[0]
            active_cpu = active.detach().cpu()
            active_presence = presence[active_cpu]
            active_scores = raw_scores[active_cpu]
            for offset in THRESHOLD_OFFSETS:
                policy = f"threshold_offset:{offset:+.2f}"
                mask = raw_mask & (set_output["atom_presence_logits"][index] >= BASELINE_THRESHOLD + offset)
                # Geometry is audited on CPU; the routing mask originates on
                # the model device (CUDA in the normal run).
                candidates = raw_geometry[mask.detach().cpu()]
                strict, any6, reach16, _ = _status(candidates, targets)
                _update(policies[policy], candidates=candidates, targets=targets,
                        strict=strict, any6=any6, reach16=reach16)
                _update(split_policies[policy][str(meta["split"])], candidates=candidates,
                        targets=targets, strict=strict, any6=any6, reach16=reach16)
                for ti, target in enumerate(targets):
                    for kind, value in _target_bucket(target, meta, sample):
                        c = strata[f"{policy}|{meta['split']}|{kind}|{value}"]
                        c["targets"] += 1; c["strict_owned"] += int(strict[ti])
                        c["reachable_6_any"] += int(any6[ti]); c["reachable_16"] += int(reach16[ti])
            # Top-K is a fixed target-free retention policy over active raw
            # proposals, ranked only by frozen set-head presence logits.
            order = torch.argsort(active_presence, descending=True)
            for k in TOP_K:
                keep = active[order[: min(k, len(order))].to(active.device)]
                mask = torch.zeros_like(raw_mask); mask[keep] = True
                candidates = raw_geometry[mask.detach().cpu()]
                strict, any6, reach16, _ = _status(candidates, targets)
                policy = f"top_k:{k}"
                _update(policies[policy], candidates=candidates, targets=targets,
                        strict=strict, any6=any6, reach16=reach16)
                _update(split_policies[policy][str(meta["split"])], candidates=candidates,
                        targets=targets, strict=strict, any6=any6, reach16=reach16)
                for ti, target in enumerate(targets):
                    for kind, value in _target_bucket(target, meta, sample):
                        c = strata[f"{policy}|{meta['split']}|{kind}|{value}"]
                        c["targets"] += 1; c["strict_owned"] += int(strict[ti])
                        c["reachable_6_any"] += int(any6[ti]); c["reachable_16"] += int(reach16[ti])
        batch.clear()

    for sample in ChemVectorIterableDataset(
        shards, include_keys=expected_keys, sequence_version="0.4", load_raster_layers=False,
    ):
        key = str(sample["key"])
        if key not in metadata:
            continue
        if key in seen:
            raise ValueError(f"duplicate loaded key: {key}")
        if str(sample["meta"].get("split")) != "train":
            raise ValueError(f"non-train source leaked into pilot: {key}")
        if str(sample["meta"].get("molecule_id")) != str(metadata[key]["molecule_id"]):
            raise ValueError(f"pilot molecule identity mismatch: {key}")
        seen.add(key); batch.append(sample)
        if len(batch) >= BATCH_SIZE:
            consume()
    consume()
    missing = sorted(expected_keys - seen)
    if missing:
        raise ValueError(f"missing pilot keys: {missing[:8]}")
    state_after = state_digest(model)
    if state_before != state_after or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("frozen model state changed during risk-curve replay")
    train_ids = {g["molecule_id"] for g in manifest["train"]["molecules"]}
    holdout_ids = {g["molecule_id"] for g in manifest["holdout"]["molecules"]}
    policy_report = {
        name: {
            "counts": dict(counter),
            "rates": _rates(counter),
            "split": {
                split: {"counts": dict(split_counter), "rates": _rates(split_counter)}
                for split, split_counter in sorted(split_policies[name].items())
            },
        }
        for name, counter in policies.items()
    }
    # The aggregate policy counters are useful for compact comparison; strata
    # retain the train/holdout split and focus labels for decision-making.
    strata_report = {}
    for key, counter in sorted(strata.items()):
        n = max(int(counter["targets"]), 1)
        strata_report[key] = {**dict(counter),
                              "strict_recall": counter["strict_owned"] / n,
                              "reachable_6_any_recall": counter["reachable_6_any"] / n,
                              "reachable_16_recall": counter["reachable_16"] / n}
    result = {
        "schema_version": "chemvector-visible-on-selector-risk-curve-0.2",
        "complete": True, "read_only": True, "training_performed": False,
        "optimizer_steps": 0, "gpu_used": device.type == "cuda", "device": str(device),
        "sealed_test_read": False, "production_checkpoint_written": False,
        "manifest": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": digest(MANIFEST),
        "source_data": str(DATA.relative_to(ROOT)), "source_shard_count": len(shards),
        "images": len(seen), "expected_images": len(expected_keys), "missing_keys": len(missing),
        "baseline_selector_threshold": BASELINE_THRESHOLD,
        "threshold_offsets": list(THRESHOLD_OFFSETS), "top_k_policies": list(TOP_K),
        "policy_contract": {"raw_proposals_frozen": True, "presence_logits_frozen": True,
                            "routing_target_free": True, "targets_posthoc_only": True,
                            "threshold_selection": "fixed_predeclared_offsets",
                            "top_k_selection": "fixed_predeclared_k_on_presence_logits"},
        "policies": policy_report, "strata": strata_report,
        "identity": {"train_molecules": len(train_ids), "holdout_molecules": len(holdout_ids),
                     "overlap": len(train_ids & holdout_ids)},
        "initial_checkpoint_sha256": INITIAL_SHA,
        "initial_checkpoint_sha256_after": digest(INITIAL),
        "model_state_before": state_before, "model_state_after": state_after,
        "gate": {"initial_unchanged": state_before == state_after,
                 "identity_disjoint": not (train_ids & holdout_ids),
                 "all_keys_loaded": not missing, "public_train_only": True,
                 "sealed_test_read": False, "training_admissible": False,
                 "gpu_training_go": False, "production_go": False,
                 "next": "review fixed risk curve; no selector training or merge"},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT / "report.json"), "images": len(seen),
                      "device": str(device), "policies": {k: v["rates"] for k, v in policy_report.items()},
                      "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
