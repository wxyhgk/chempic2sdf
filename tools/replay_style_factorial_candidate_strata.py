"""Frozen-initial candidate replay for the style-factorial stream.

This is read-only inference.  It does not build an optimizer, call backward,
write a checkpoint, or authorize adaptation.  Each raw atom proposal is
matched one-to-one to the committed rendered atom anchors and assigned to an
explicit selector stratum:

* strict_positive: raw match <= 6 px, but the set head drops it;
* protected: the set head keeps a matched candidate (not a loss row);
* near_unknown: unmatched and nearest atom <= 10 px (masked from loss);
* hard_negative: unmatched and nearest atom > 10 px.

The replay is intentionally separate from the later visual-gate training.  It
produces per-split/per-style counts so that the visual gate can be evaluated
without silently mixing the ``stress`` style into the core stream.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch

from Data.sdf.chemsvg_sequence_v4 import ChemSVGCoordinateCollisionError
from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.data import ChemVectorIterableDataset
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "outputs/audits/chemvector-style-factorial-core-stream-20261007-r1"
DATA = ROOT / "Data/sdf/chemvector_joint_v1/style_train_100k"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r1/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
BATCH_SIZE = 8
NEAR_UNKNOWN_PX = 10.0
CORE_VARIANTS = ("coordgen", "original", "rdkit")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["rows"]


def replay_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def empty_counts() -> Counter:
    return Counter({"raw_candidates": 0, "strict_positive": 0, "protected": 0,
                    "near_unknown": 0, "hard_negative": 0,
                    "selected_unmatched": 0, "mapping_anomaly": 0})


def classify_batch(batch_samples: list[dict], tensors: dict[str, torch.Tensor],
                   raw: dict[str, torch.Tensor], set_output: dict[str, torch.Tensor],
                   selected: dict[str, torch.Tensor], config: AtomBondSceneConfig) -> tuple[list[dict], list[dict]]:
    raw_map, _ = match_atom_targets(raw, tensors, config)
    selected_map, _ = match_atom_targets(selected, tensors, config)
    out: list[dict] = []
    candidate_cache: list[dict] = []
    image_size = float(config.image_size)
    for i, sample in enumerate(batch_samples):
        target_count = int(tensors["atom_mask"][i].sum())
        targets = tensors["atom_anchors"][i, :target_count, :2]
        raw_indices = torch.where(raw["atom_mask"][i])[0]
        counts = empty_counts()
        distance_bins = Counter()
        for j in raw_indices.tolist():
            counts["raw_candidates"] += 1
            candidate = raw["atom_geometry"][i, j, :2]
            if target_count:
                nearest = float(torch.linalg.vector_norm((targets - candidate) * image_size, dim=-1).min().detach().cpu())
            else:
                nearest = float("inf")
            raw_matched = int(raw_map[i, j]) >= 0
            selected_kept = bool(selected["atom_mask"][i, j])
            selected_matched = int(selected_map[i, j]) >= 0
            if selected_kept != selected_matched:
                counts["mapping_anomaly"] += 1
            label = "selected_unmatched"
            if raw_matched and selected_matched:
                counts["protected"] += 1
                distance_bins["protected"] += 1
                label = "protected"
            elif raw_matched and not selected_matched:
                # raw matcher is <=6 px by contract; no GT-nearest oracle is
                # used beyond this matching decision.
                counts["strict_positive"] += 1
                distance_bins["strict_positive"] += 1
                label = "strict_positive"
            elif not raw_matched and not selected_matched:
                if nearest <= NEAR_UNKNOWN_PX:
                    counts["near_unknown"] += 1
                    distance_bins["near_unknown"] += 1
                    label = "near_unknown"
                else:
                    counts["hard_negative"] += 1
                    distance_bins["hard_negative"] += 1
                    label = "hard_negative"
            else:
                counts["selected_unmatched"] += 1
                distance_bins["selected_unmatched"] += 1
                label = "selected_unmatched"
            # This record is optional and written by the caller as JSONL.  It
            # intentionally contains no target atom id or rendered layers:
            # nearest distance is an audit stratum, not an oracle assignment
            # exposed to a model.
            candidate_cache.append({
                "sample_key": sample["key"], "molecule_id": sample["molecule_id"],
                "replay_split": sample["replay_split"], "variant": sample["variant"],
                "profile": sample["profile"], "slot": int(j), "label": label,
                "raw_matched": raw_matched, "selected_kept": selected_kept,
                "selected_matched": selected_matched,
                "raw_score": float(raw["atom_scores"][i, j].detach().cpu()),
                "presence_logit": float(set_output["atom_presence_logits"][i, j].detach().cpu()),
                "x_norm": float(candidate[0].detach().cpu()), "y_norm": float(candidate[1].detach().cpu()),
                "x_px": float(candidate[0].detach().cpu() * image_size),
                "y_px": float(candidate[1].detach().cpu() * image_size),
                "nearest_distance_px": nearest,
            })
        out.append({"sample_key": sample["key"], "molecule_id": sample["molecule_id"],
                    "variant": sample["variant"], "profile": sample["profile"],
                    "target_atoms": target_count, "strata": dict(counts),
                    "distance_bins": dict(distance_bins)})
    return out, candidate_cache


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    cache_output = Path(args.cache_output) if args.cache_output else None
    if output.exists():
        raise FileExistsError(output)
    if cache_output is not None and cache_output.exists():
        raise FileExistsError(cache_output)
    cache_handle = None
    if cache_output is not None:
        cache_output.parent.mkdir(parents=True, exist_ok=True)
        cache_handle = cache_output.open("x", encoding="utf-8")
    device = replay_device(args.device)
    manifests = {
        "train_core": CORE / "train-core-manifest.json",
        "dev_core": CORE / "dev-core-manifest.json",
        "dev_stress": CORE / "dev-stress-manifest.json",
    }
    wanted: dict[str, dict] = {}
    for split, path in manifests.items():
        for row in load_rows(path):
            if row["sample_key"] in wanted:
                raise ValueError(f"duplicate sample key: {row['sample_key']}")
            wanted[row["sample_key"]] = {**row, "replay_split": split}
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    if digest(INITIAL) != INITIAL_SHA:
        raise ValueError("initial checkpoint SHA mismatch before replay")
    model.to(device).eval().requires_grad_(False)
    scene_config = AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds)
    collator = AtomBondSceneCollator(scene_config)
    threshold = float(payload["provenance"]["atom_threshold"])
    shards = sorted(str(p) for p in (DATA / "shards").glob("train-*.tar"))
    if len(shards) != 20:
        raise ValueError(f"expected 20 train shards, got {len(shards)}")
    rows: list[dict] = []
    skipped: list[str] = []
    seen: set[str] = set()
    pending: list[dict] = []

    def flush(samples: list[dict]) -> None:
        if not samples:
            return
        try:
            batch = collator(samples)
        except ChemSVGCoordinateCollisionError:
            skipped.extend(s["key"] for s in samples)
            return
        tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.inference_mode():
            _, _, _, raw, _, set_output, selected = structure_forward(model, tensors["pixel_values"], config, threshold)
        sample_rows, candidate_rows = classify_batch(samples, tensors, raw, set_output, selected, config)
        rows.extend(sample_rows)
        if cache_handle is not None:
            for candidate_row in candidate_rows:
                cache_handle.write(json.dumps(candidate_row, ensure_ascii=False, separators=(",", ":")) + "\n")

    for sample in ChemVectorIterableDataset(shards, sequence_version="0.4",
                                            serialize_target=False, load_raster_layers=False):
        key = sample["key"]
        if key not in wanted:
            continue
        if key in seen:
            raise ValueError(f"duplicate sample from shards: {key}")
        seen.add(key)
        manifest_row = wanted[key]
        # Keep only metadata needed for the output; all model inputs remain in
        # the in-memory collator batch and are not persisted.
        pending.append({"key": key, "molecule_id": manifest_row["molecule_id"],
                        "replay_split": manifest_row["replay_split"],
                        "variant": manifest_row["variant"], "profile": manifest_row["profile"],
                        "image": sample["image"], "target_sequence": sample["target_sequence"],
                        "chemmol": sample["chemmol"], "render": sample["render"],
                        "meta": sample["meta"], "layers": sample.get("layers")})
        if len(pending) == BATCH_SIZE:
            flush(pending)
            pending = []
    flush(pending)
    if cache_handle is not None:
        cache_handle.close()
    if seen != set(wanted):
        raise ValueError(f"manifest samples missing from shards: {len(set(wanted) - seen)}")
    if set(skipped) & set(wanted):
        for key in skipped:
            seen.add(key)
    by_split_variant: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(empty_counts))
    by_split: dict[str, Counter] = defaultdict(empty_counts)
    per_sample = []
    for row in rows:
        split = wanted[row["sample_key"]]["replay_split"]
        variant = row["variant"]
        for name, value in row["strata"].items():
            by_split_variant[split][variant][name] += int(value)
            by_split[split][name] += int(value)
        per_sample.append({"sample_key": row["sample_key"], "molecule_id": row["molecule_id"],
                           "variant": variant, "profile": row["profile"],
                           "target_atoms": row["target_atoms"], "strata": row["strata"]})
    report = {
        "schema_version": "chemvector-style-factorial-candidate-strata-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "optimizer_steps": 0, "backward_called": False, "checkpoint_written": False,
        "gpu_used": device.type == "cuda", "device": str(device), "sealed_test_read": False,
        "initial_sha256": INITIAL_SHA, "initial_sha256_after": digest(INITIAL),
        "manifest_sha256": {k: digest(v) for k, v in manifests.items()},
        "candidate_cache": ({"path": str(cache_output.resolve().relative_to(ROOT.resolve())),
                             "sha256": digest(cache_output)} if cache_output is not None else None),
        "replay_samples": len(rows), "manifest_samples": len(wanted),
        "coordinate_collision_samples_skipped": len(skipped), "skipped_sample_keys": sorted(skipped),
        "batch_size": BATCH_SIZE, "atom_threshold": threshold, "strict_radius_px": config.atom_match_radius_px,
        "near_unknown_radius_px": NEAR_UNKNOWN_PX,
        "by_split_variant": {s: {v: dict(c) for v, c in sorted(groups.items())}
                             for s, groups in sorted(by_split_variant.items())},
        "by_split": {s: dict(c) for s, c in sorted(by_split.items())},
        "per_sample": per_sample,
        "label_contract": {"strict_positive": "raw <=6px and selected unmatched",
                            "protected": "raw <=6px and selected matched",
                            "near_unknown": "raw unmatched and nearest <=10px (loss weight 0)",
                            "hard_negative": "raw unmatched and nearest >10px",
                            "strict_group_unknown_excluded": True},
        "gate": {"all_manifest_samples_seen": len(seen) == len(wanted),
                 "all_replayable_samples_replayed": len(rows) + len(skipped) == len(wanted),
                 "initial_unchanged": digest(INITIAL) == INITIAL_SHA,
                 "per_style_strata_reported": all(v in by_split_variant.get("dev_core", {}) for v in CORE_VARIANTS),
                 "training_admissible": False,
                 "next": "fit CPU visual gate on train_core strict positives + hard negatives only; evaluate dev_core per style and dev_stress independently"}}
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "device": str(device),
                      "replay_samples": len(rows), "skipped": len(skipped),
                      "by_split_variant": report["by_split_variant"], "gate": report["gate"]},
                     indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--cache-output", default=None,
                        help="optional candidate-level JSONL path for CPU gate fitting")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
