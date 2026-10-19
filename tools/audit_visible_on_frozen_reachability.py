#!/usr/bin/env python3
"""Read-only frozen-initial reachability audit for the visible O/N pilot.

The pilot is a metadata-bound manifest carved from public ``train`` shards.
This audit only runs the frozen initial model in eval/no-grad mode.  It never
fits a threshold, creates an optimizer, writes a checkpoint, or reads a
sealed/internal split.  Target geometry is joined *after* inference solely
for post-hoc reachability accounting.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
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

INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
MANIFEST = ROOT / "outputs/audits/chemvector-visible-on-data-first-pilot-20261017-r1/manifest.json"
DATA = ROOT / "Data/sdf/chemvision_layers_v0_2/font_pilot_10k"
OUT = ROOT / os.environ.get(
    "CHEMVECTOR_VISIBLE_REACHABILITY_OUT",
    "outputs/audits/chemvector-visible-on-frozen-reachability-20261017-r1",
)
IMAGE_SIZE = 768.0
STRICT_RADIUS = 6.0
REACH_RADIUS = 16.0
OTHER_TARGET_MIN = 20.0
MARGIN = 4.0
DEFAULT_BATCH = 16


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _metadata(manifest: dict) -> dict[str, dict]:
    result = {}
    for split in ("train", "holdout"):
        for group in manifest[split]["molecules"]:
            for row in group["rows"]:
                key = str(row["sample_key"])
                if key in result:
                    raise ValueError(f"duplicate pilot sample key: {key}")
                value = dict(row)
                value["split"] = split
                result[key] = value
    return result


def _degree(layers: dict) -> dict[int, int]:
    degree: Counter[int] = Counter()
    for bond in layers.get("bonds", []):
        degree[int(bond["begin_atom"])] += 1
        degree[int(bond["end_atom"])] += 1
    return dict(degree)


def _rotation_bucket(value: float) -> str:
    if value < -4.0:
        return "neg"
    if value > 4.0:
        return "pos"
    return "near_zero"


def _target_rows(sample: dict) -> list[dict]:
    layers = sample["layers"]
    degrees = _degree(layers)
    out = []
    for atom in layers.get("atoms", []):
        element = str(atom.get("element", ""))
        visible = bool(atom.get("visible_text"))
        role = "text" if visible else "implicit"
        out.append({
            "atom_id": int(atom["atom_id"]),
            "element": element,
            "role": role,
            "degree": min(int(degrees.get(int(atom["atom_id"]), 0)), 3),
            "xy": (float(atom["center"][0]), float(atom["center"][1])),
        })
    return out


def _candidate_stats(candidates: torch.Tensor, targets: list[dict]) -> dict:
    """Compute target-centric reachability and conservative ownership.

    The strict ownership contract mirrors the existing raw replay: exactly one
    nearby candidate, no candidate near another target, a four-pixel margin,
    and no candidate reuse.  ``reachable_16`` is an upper-bound diagnostic and
    is intentionally not used to select or alter candidates.
    """
    target_count = len(targets)
    if target_count:
        target_xy = torch.tensor([t["xy"] for t in targets], dtype=torch.float32)
        distance = torch.cdist(candidates.cpu().float(), target_xy)
    else:
        distance = torch.empty((len(candidates), 0), dtype=torch.float32)
    strict_owned: set[int] = set()
    unique_matches: dict[int, int] = {}
    counts = Counter()
    unknown = Counter()
    for target_index, target in enumerate(targets):
        row = distance[:, target_index] if len(candidates) else torch.empty(0)
        nearby_strict = torch.where(row <= STRICT_RADIUS)[0].tolist()
        nearby_reach = torch.where(row <= REACH_RADIUS)[0].tolist()
        counts["target_count"] += 1
        counts["reachable_16"] += int(bool(nearby_reach))
        counts["reachable_6_any"] += int(bool(nearby_strict))
        if len(nearby_strict) != 1:
            unknown["target_zero_or_multiple_within_6"] += 1
            continue
        candidate_index = nearby_strict[0]
        all_dist = distance[candidate_index]
        own = float(all_dist[target_index])
        other = torch.cat((all_dist[:target_index], all_dist[target_index + 1:]))
        nearest_other = float(other.min()) if len(other) else float("inf")
        if nearest_other <= OTHER_TARGET_MIN:
            unknown["candidate_near_other_target"] += 1
            continue
        if nearest_other - own < MARGIN:
            unknown["target_margin_small"] += 1
            continue
        if candidate_index in strict_owned:
            unknown["candidate_reused"] += 1
            continue
        strict_owned.add(candidate_index)
        unique_matches[target_index] = candidate_index
        counts["strict_owned"] += 1
    counts["candidate_count"] = int(len(candidates))
    counts["strict_candidate_true"] = int(len(strict_owned))
    counts["candidate_far_from_all_targets"] = int(sum(
        bool(distance[i].numel() == 0 or float(distance[i].min()) > OTHER_TARGET_MIN)
        for i in range(len(candidates))
    ))
    return {"counts": dict(counts), "unknown": dict(unknown),
            "strict_target_indices": sorted(unique_matches)}


def _merge(total: Counter, unknown: Counter, rows: list[dict], stats: dict) -> None:
    counts = stats["counts"]
    total.update({k: int(v) for k, v in counts.items()})
    unknown.update({k: int(v) for k, v in stats["unknown"].items()})


def _rates(c: Counter) -> dict:
    targets = max(int(c["target_count"]), 1)
    candidates = max(int(c["candidate_count"]), 1)
    return {
        "targets": int(c["target_count"]),
        "candidates": int(c["candidate_count"]),
        "strict_recall": c["strict_owned"] / targets,
        "reachable_6_any_recall": c["reachable_6_any"] / targets,
        "reachable_16_recall": c["reachable_16"] / targets,
        "strict_candidate_precision": c["strict_candidate_true"] / candidates,
    }


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
    if not DATA.is_dir() or "font_pilot_10k" not in DATA.name:
        raise RuntimeError("unexpected source dataset; refusing non-pilot data")
    metadata = _metadata(manifest)
    expected_keys = set(metadata)
    shards = sorted(str(path) for path in (DATA / "shards").glob("train-*.tar"))
    if not shards:
        raise FileNotFoundError("no public train shards found")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval().requires_grad_(False)
    before = state_digest(model)
    threshold = float(payload["provenance"]["atom_threshold"])
    by_arm_split: dict[str, Counter] = defaultdict(Counter)
    by_stratum: dict[str, Counter] = defaultdict(Counter)
    unknown_by_arm_split: dict[str, Counter] = defaultdict(Counter)
    seen: set[str] = set()
    rows_seen = 0
    batch_samples: list[dict] = []

    def flush() -> None:
        nonlocal rows_seen
        if not batch_samples:
            return
        images = torch.stack([image_tensor(s["image"]) for s in batch_samples]).to(device)
        with torch.inference_mode():
            outputs = structure_forward(model, images, config, threshold)
        raw = outputs[3]
        selected = outputs[6]
        for index, sample in enumerate(batch_samples):
            key = str(sample["key"])
            meta = metadata[key]
            targets = _target_rows(sample)
            split = str(meta["split"])
            params = sample["layers"].get("render_parameters", {})
            # The pilot is variant/layout based rather than oaXX based.  Keep
            # this explicit instead of inventing a profile ID that is absent
            # from the source contract.
            profile = str(meta.get("profile_id") or meta.get("variant") or params.get("layout") or "unknown")
            rotation = _rotation_bucket(float(meta.get("rotation_degrees", params.get("rotation_degrees", 0.0))))
            font_bucket = str(meta.get("font_bucket") or "unknown")
            for arm, arm_output in (("raw", raw), ("selected", selected)):
                mask = arm_output["atom_mask"][index].bool()
                candidates = arm_output["atom_geometry"][index, mask, :2].detach().cpu() * IMAGE_SIZE
                stats = _candidate_stats(candidates, targets)
                arm_split = f"{arm}|{split}"
                _merge(by_arm_split[arm_split], unknown_by_arm_split[arm_split], [], stats)
                # Attribute the same target rows to stable post-hoc strata.
                # Each target gets its own denominator; candidates stay image-level.
                distance = torch.cdist(candidates.float(), torch.tensor([t["xy"] for t in targets])) if targets else torch.empty((len(candidates), 0))
                owned = set(stats["strict_target_indices"])
                for ti, target in enumerate(targets):
                    label = target["element"] if target["element"] in {"O", "N"} and target["role"] == "text" else "all"
                    if label == "all" and target["role"] != "implicit":
                        continue
                    strata = [
                        ("profile", profile),
                        ("element_role", label),
                        ("element_degree", f"{target['element']}_{target['role']}_degree_{target['degree']}"),
                        ("font", str(meta.get("font_family_id", "unknown"))),
                        ("font_bucket", font_bucket),
                        ("rotation_bucket", rotation),
                        ("variant", str(meta.get("variant", "unknown"))),
                    ]
                    for kind, value in strata:
                        counter = by_stratum[f"{arm}|{split}|{kind}|{value}"]
                        counter["targets"] += 1
                        if ti in owned:
                            counter["strict_owned"] += 1
                        if distance.numel() and bool((distance[:, ti] <= REACH_RADIUS).any()):
                            counter["reachable_16"] += 1
                        if distance.numel() and bool((distance[:, ti] <= STRICT_RADIUS).any()):
                            counter["reachable_6_any"] += 1
            rows_seen += 1
            if rows_seen % 100 == 0:
                print(json.dumps({"images": rows_seen, "device": str(device)}), flush=True)
        batch_samples.clear()

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
        seen.add(key)
        batch_samples.append(sample)
        if len(batch_samples) >= DEFAULT_BATCH:
            flush()
    flush()
    missing = sorted(expected_keys - seen)
    if missing:
        raise ValueError(f"missing pilot keys: {missing[:8]}")
    after = state_digest(model)
    if before != after or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen model state changed during replay")

    split_report = {}
    for arm_split, counter in sorted(by_arm_split.items()):
        arm, split = arm_split.split("|", 1)
        split_report.setdefault(split, {})[arm] = {
            "counts": dict(counter), "rates": _rates(counter),
            "unknown": dict(unknown_by_arm_split[arm_split]),
        }
    strata_report = {}
    for key, counter in sorted(by_stratum.items()):
        targets = max(int(counter["targets"]), 1)
        strata_report[key] = {**dict(counter),
                              "strict_recall": counter["strict_owned"] / targets,
                              "reachable_16_recall": counter["reachable_16"] / targets,
                              "reachable_6_any_recall": counter["reachable_6_any"] / targets}
    train_ids = {row["molecule_id"] for row in manifest["train"]["molecules"]}
    holdout_ids = {row["molecule_id"] for row in manifest["holdout"]["molecules"]}
    result = {
        "schema_version": "chemvector-visible-on-frozen-reachability-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "optimizer_steps": 0, "gpu_used": device.type == "cuda",
        "device": str(device), "sealed_test_read": False,
        "production_checkpoint_written": False,
        "manifest": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": digest(MANIFEST),
        "source_data": str(DATA.relative_to(ROOT)), "source_shard_count": len(shards),
        "images": rows_seen, "expected_images": len(expected_keys), "missing_keys": len(missing),
        "threshold": threshold,
        "ownership_contract": {"strict_radius_px": STRICT_RADIUS, "reach_radius_px": REACH_RADIUS,
                                "other_target_min_px": OTHER_TARGET_MIN, "margin_px": MARGIN,
                                "hungarian_used": False, "oracle_candidates_injected": False},
        "split": split_report, "strata": strata_report,
        "identity": {"train_molecules": len(train_ids), "holdout_molecules": len(holdout_ids),
                     "overlap": len(train_ids & holdout_ids)},
        "initial_checkpoint_sha256": INITIAL_SHA,
        "initial_checkpoint_sha256_after": digest(INITIAL),
        "model_state_before": before, "model_state_after": after,
        "gate": {"initial_unchanged": before == after, "identity_disjoint": not (train_ids & holdout_ids),
                 "all_keys_loaded": not missing, "public_train_only": True,
                 "sealed_test_read": False, "training_admissible": False,
                 "gpu_training_go": False, "production_go": False,
                 "next": "review profile/element/degree strata; no training admission"},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT / "report.json"), "images": rows_seen,
                      "device": str(device), "split": split_report, "gate": result["gate"]},
                     ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
