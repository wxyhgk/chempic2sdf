"""CPU smoke for explicit residual-to-predicted-bond-port assignment.

The frozen initial model supplies bond endpoints.  Baseline occupancy is built
only from the model-visible ``selected_kept`` flag; protected/strict labels
are consulted after assignment for diagnostics. Residual points are assigned
to at most one unoccupied endpoint by ``attach_residual_candidates`` within a
fixed 12 px radius. No pointwise scorer is trained and no GT geometry is used
to form assignments.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.components.graph.proposals.residual_bond_attachment import attach_residual_candidates
from chemvector.data import ChemVectorIterableDataset
from chemvector.train_atom_bond_scene import AtomBondSceneCollator
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/audits/chemvector-style-factorial-candidate-strata-20261010-r2/candidates.jsonl"
DATA = ROOT / "Data/sdf/chemvector_joint_v1/style_train_100k/shards"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-predicted-port-attachment-cpu-20261015-r1/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE_BATCH = 4
MAX_PER_VARIANT = 20
RADIUS = 12.0
DUPLICATE = 6.0
IMAGE_SIZE = 768.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cache() -> dict[str, list[dict]]:
    groups = defaultdict(list)
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            groups[row["sample_key"]].append(row)
    return groups


def choose(groups: dict[str, list[dict]]) -> set[str]:
    by = defaultdict(list)
    for key, rows in groups.items():
        by[(rows[0]["replay_split"], rows[0]["variant"])].append(key)
    chosen = set()
    for split, variants in (("train_core", ("coordgen", "original", "rdkit")),
                            ("dev_core", ("coordgen", "original", "rdkit")),
                            ("dev_stress", ("stress",))):
        for variant in variants:
            chosen.update(sorted(by[(split, variant)])[:MAX_PER_VARIANT])
    return chosen


def one_sample(rows: list[dict], bonds: dict[str, torch.Tensor], selected: dict[str, torch.Tensor]) -> dict:
    points = torch.tensor([[r["x_norm"], r["y_norm"]] for r in rows], dtype=torch.float32)
    selected_mask = torch.tensor([bool(r["selected_kept"]) for r in rows], dtype=torch.bool)
    baseline = points[selected_mask]
    residual = points[~selected_mask]
    labels = [r["label"] for r in rows]
    residual_labels = [label for label, keep in zip(labels, selected_mask.tolist()) if not keep]
    endpoints = bonds["bond_endpoints"][0].detach().cpu()
    bond_mask = bonds["bond_mask"][0].detach().cpu().bool()
    occupied = torch.zeros((len(endpoints), 2), dtype=torch.bool)
    if len(baseline) and bool(bond_mask.any()):
        distances = torch.cdist(endpoints[bond_mask].reshape(-1, 2) * IMAGE_SIZE, baseline * IMAGE_SIZE)
        occupied_flat = distances.min(dim=1).values <= DUPLICATE
        occupied[bond_mask] = occupied_flat.reshape(-1, 2)
    result = attach_residual_candidates(residual, endpoints, bond_mask, radius_px=RADIUS,
                                        image_size=IMAGE_SIZE, occupied_endpoints=occupied)
    accepted = result["matched"].tolist()
    accepted_labels = [label for label, keep in zip(residual_labels, accepted) if keep]
    return {"residual": len(residual), "baseline_selected": len(baseline), "ports": int(bond_mask.sum()) * 2,
            "occupied_ports": int(occupied.sum()), "attached": len(accepted_labels),
            "strict_selected": sum(label == "strict_positive" for label in accepted_labels),
            "hard_negative_selected": sum(label == "hard_negative" for label in accepted_labels),
            "near_unknown_selected": sum(label == "near_unknown" for label in accepted_labels),
            "selected_unmatched": sum(label == "selected_unmatched" for label in accepted_labels)}


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    groups = load_cache(); chosen = choose(groups)
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    if digest(INITIAL) != INITIAL_SHA: raise ValueError("initial SHA changed")
    model.to("cpu").eval().requires_grad_(False)
    collator = AtomBondSceneCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds))
    shards = sorted(str(p) for p in DATA.glob("train-*.tar")); seen = set(); pending = []; rows_out = []

    def flush(items: list[dict]) -> None:
        if not items: return
        batch = collator(items); tensors = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.inference_mode():
            _, _, _, _, bonds, _, selected = structure_forward(model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"])
        for index, sample in enumerate(items):
            b = {k: v[index:index + 1] for k, v in bonds.items()}; s = {k: v[index:index + 1] for k, v in selected.items()}
            row = one_sample(groups[sample["key"]], b, s); row.update(sample_key=sample["key"], replay_split=groups[sample["key"]][0]["replay_split"], variant=groups[sample["key"]][0]["variant"]); rows_out.append(row)

    for sample in ChemVectorIterableDataset(shards, sequence_version="0.4", serialize_target=False, load_raster_layers=False):
        if sample["key"] not in chosen: continue
        seen.add(sample["key"]); pending.append(sample)
        if len(pending) == IMAGE_BATCH: flush(pending); pending = []
    flush(pending)
    if seen != chosen: raise ValueError(f"missing chosen samples: {len(chosen - seen)}")
    totals = defaultdict(lambda: defaultdict(int))
    for row in rows_out:
        key = f"{row['replay_split']}/{row['variant']}"
        for name in ("residual", "baseline_selected", "ports", "occupied_ports", "attached", "strict_selected", "hard_negative_selected", "near_unknown_selected", "selected_unmatched"): totals[key][name] += row[name]
    metrics = {}
    for key, c in totals.items():
        metrics[key] = {**dict(c), "strict_precision_conservative": c["strict_selected"] / max(c["attached"], 1), "assignment_recall_not_available": True}
    result = {"schema_version": "chemvector-style-factorial-predicted-port-attachment-cpu-0.1", "complete": True, "smoke": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "initial_sha256": INITIAL_SHA, "cache_sha256": digest(CACHE), "sample_limits_per_variant": MAX_PER_VARIANT, "radius_px": RADIUS, "duplicate_veto_px": DUPLICATE, "occupancy_source": "selected_kept model output only", "labels_used_after_assignment_only": True, "metrics": metrics, "rows": len(rows_out), "gate": {"training_admissible": False, "next": "if precision is useful, scale a bounded endpoint assignment audit; do not add a pointwise selector gate"}}
    output.parent.mkdir(parents=True, exist_ok=False); output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n"); print(json.dumps({"output": str(output), "metrics": metrics, "gate": result["gate"]}, indent=2, ensure_ascii=False)); return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", default=str(OUTPUT)); run(parser.parse_args())
