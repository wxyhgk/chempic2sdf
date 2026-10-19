"""Forward-only endpoint reachability audit on the style-factorial dev pool."""

from __future__ import annotations

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
POOL = ROOT / "outputs/audits/chemvector-style-factorial-target-pool-20261007-r1"
DATA = ROOT / "Data/sdf/chemvector_joint_v1/style_train_100k"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-endpoint-reachability-20261007-r1/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
BATCH_SIZE = 4


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads((POOL / "dev-manifest.json").read_text(encoding="utf-8"))
    wanted = {r["sample_key"]: r for r in manifest["rows"]}
    if len(wanted) != 1600:
        raise ValueError(f"expected 1600 dev samples, got {len(wanted)}")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    if digest(INITIAL) != INITIAL_SHA:
        raise ValueError("initial checkpoint changed")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval().requires_grad_(False)
    scene_config = AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds)
    collator = AtomBondSceneCollator(scene_config)
    threshold = float(payload["provenance"]["atom_threshold"])
    shards = sorted(str(p) for p in (DATA / "shards").glob("train-*.tar"))
    if len(shards) != 20:
        raise ValueError(f"expected 20 train shards, got {len(shards)}")
    rows = []
    skipped_collision: list[str] = []
    pending = []
    seen = set()

    def flush(batch_samples: list[dict]) -> None:
        if not batch_samples:
            return
        try:
            batch = collator(batch_samples)
        except ChemSVGCoordinateCollisionError:
            # The source commitment records a tiny rejected subset.  Keep the
            # identity visible in the report, but do not fabricate targets or
            # silently count it as a model miss.
            skipped_collision.extend(sample["key"] for sample in batch_samples)
            return
        tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.inference_mode():
            _, _, _, raw, _, set_output, selected = structure_forward(
                model, tensors["pixel_values"], config, threshold)
        raw_map, _ = match_atom_targets(raw, tensors, config)
        selected_map, _ = match_atom_targets(selected, tensors, config)
        for i, sample in enumerate(batch_samples):
            target_count = int(tensors["atom_mask"][i].sum())
            raw_ids = {int(x) for x in raw_map[i].detach().cpu().tolist() if int(x) >= 0}
            selected_ids = {int(x) for x in selected_map[i].detach().cpu().tolist() if int(x) >= 0}
            row = wanted[sample["key"]]
            rows.append({"sample_key": sample["key"], "molecule_id": row["molecule_id"],
                         "profile": row["profile"], "variant": row["variant"],
                         "target_atoms": target_count, "raw_matched": len(raw_ids),
                         "selected_matched": len(selected_ids),
                         "raw_gap": target_count - len(raw_ids),
                         "selector_drop": len(raw_ids - selected_ids)})

    for sample in ChemVectorIterableDataset(shards, sequence_version="0.4",
                                            serialize_target=False, load_raster_layers=False):
        key = sample["key"]
        if key not in wanted:
            continue
        if key in seen:
            raise ValueError(f"duplicate sample: {key}")
        seen.add(key)
        pending.append(sample)
        if len(pending) == BATCH_SIZE:
            flush(pending); pending = []
    flush(pending)
    if seen != set(wanted):
        raise ValueError(f"manifest samples missing from shards: {len(set(wanted) - seen)}")

    totals = {name: sum(int(r[name]) for r in rows) for name in ("target_atoms", "raw_matched", "selected_matched", "raw_gap", "selector_drop")}
    by_profile: dict[str, dict] = defaultdict(lambda: Counter())
    by_variant: dict[str, dict] = defaultdict(lambda: Counter())
    for row in rows:
        for bucket, key in ((by_profile[row["profile"]["font_bucket"]], "font_bucket"),
                            (by_variant[row["variant"]], "variant")):
            for name in ("target_atoms", "raw_matched", "selected_matched", "raw_gap", "selector_drop"):
                bucket[name] += row[name]
    report = {
        "schema_version": "chemvector-style-factorial-endpoint-reachability-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "optimizer_steps": 0, "gpu_used": device.type == "cuda", "device": str(device),
        "sealed_test_read": False, "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA, "initial_sha256_after": digest(INITIAL),
        "pool_manifest_sha256": digest(POOL / "dev-manifest.json"),
        "samples": len(rows), "manifest_samples": len(wanted),
        "coordinate_collision_samples_skipped": len(skipped_collision),
        "skipped_sample_keys": sorted(skipped_collision),
        "molecules": len({r["molecule_id"] for r in rows}),
        "batch_size": BATCH_SIZE, "atom_threshold": threshold, "totals": totals,
        "by_font_bucket": {k: dict(v) for k, v in sorted(by_profile.items())},
        "by_variant": {k: dict(v) for k, v in sorted(by_variant.items())},
        "gate": {"all_manifest_samples_seen": len(seen) == len(wanted),
                 "all_replayable_samples_replayed": len(rows) + len(skipped_collision) == len(wanted),
                 "initial_unchanged": digest(INITIAL) == INITIAL_SHA,
                 "raw_reachability_reported": True, "training_admissible": False,
                 "next": "compare endpoint coverage against fixed baseline; only then consider a bounded visual branch"},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "device": str(device), "samples": len(rows), "totals": totals,
                      "by_font_bucket": report["by_font_bucket"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
