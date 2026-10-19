"""CPU-only frozen raw-atom radius contract audit for oa13/oa33.

The release replay historically reported two different endpoint contracts:
the curriculum matcher uses a strict 6 px radius while the r17 endpoint and
selector sidecars use a 16 px post-hoc identity radius.  This audit runs the
same frozen raw proposal decoder on a tiny, identity-disjoint train/holdout
slice and reports both radii side by side.  Target coordinates are joined
only after inference for attribution; they never affect proposal generation.

This is diagnostic only: no optimizer/checkpoint is created or written and no
sealed test content is read.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/manifest.json"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
OUT = ROOT / "outputs/audits/chemvector-r17-oa13-oa33-raw-radius-contract-cpu-v1"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
PROFILES = ("oa00", "oa13", "oa33")
MOLECULES_PER_SPLIT = 8
RADII = (6.0, 16.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_rows() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["rows"]
    chosen: list[dict] = []
    for split in ("train", "holdout"):
        molecule_ids = sorted({str(row["molecule_id"]) for row in manifest
                               if row["split"] == split},
                              key=lambda value: hashlib.sha256(
                                  ("r17-raw-radius-contract-v1:" + value).encode()).hexdigest())
        keep = set(molecule_ids[:MOLECULES_PER_SPLIT])
        chosen.extend(row for row in manifest
                      if row["split"] == split and row["molecule_id"] in keep
                      and row["profile_id"] in PROFILES)
    chosen.sort(key=lambda row: row["sample_key"])
    if len(chosen) != 2 * MOLECULES_PER_SPLIT * len(PROFILES):
        raise ValueError("identity slice is not complete")
    return chosen


def radius_mapping(points: torch.Tensor, targets: torch.Tensor, radius: float) -> dict[int, int]:
    if not len(points) or not len(targets):
        return {}
    distances = torch.cdist(points.float() * 768.0, targets.float() * 768.0)
    candidates = [(float(distances[i, j]), i, j)
                  for i in range(distances.size(0))
                  for j in range(distances.size(1))
                  if float(distances[i, j]) <= radius]
    result: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in sorted(candidates):
        if i not in result and j not in used:
            result[i] = j
            used.add(j)
    return result


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("initial checkpoint SHA changed")
    rows = choose_rows()
    train_rows = [row for row in rows if row["split"] == "train"]
    holdout_rows = [row for row in rows if row["split"] == "holdout"]
    train_connections = {str(row.get("connection_identity")) for row in train_rows}
    holdout_connections = {str(row.get("connection_identity")) for row in holdout_rows}
    train_scaffolds = {str(row.get("legacy_scaffold")) for row in train_rows}
    holdout_scaffolds = {str(row.get("legacy_scaffold")) for row in holdout_rows}
    if train_connections & holdout_connections or train_scaffolds & holdout_scaffolds:
        raise RuntimeError("identity slice has connection/scaffold overlap")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    device = torch.device("cpu")
    model.to(device).eval().requires_grad_(False)
    before = state_digest(model)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                            bond_queries=config.maximum_bonds),
        purpose="diagnostic",
    )
    totals: dict[tuple[str, str], Counter] = defaultdict(Counter)
    with torch.inference_mode():
        for offset in range(0, len(rows), 8):
            items = rows[offset:offset + 8]
            samples = [load_sample(SOURCE, row["sample_key"], purpose="diagnostic")
                       for row in items]
            batch = collator(samples)
            tensors = {key: value.to(device) for key, value in batch.items()
                       if isinstance(value, torch.Tensor)}
            _, _, _, raw, _, _, _ = structure_forward(
                model, tensors["pixel_values"], config,
                payload["provenance"]["atom_threshold"])
            strict_mapping, _ = match_atom_targets(raw, tensors, config)
            for index, row in enumerate(items):
                sample = samples[index]
                atom_count = int(tensors["atom_mask"][index].sum())
                targets = tensors["atom_anchors"][index, :atom_count]
                active = raw["atom_mask"][index]
                points = raw["atom_geometry"][index, active, :2]
                graph_edges = {tuple(sorted((int(bond["begin_atom"]),
                                              int(bond["end_atom"]))))
                               for bond in sample["graph"]["bonds"]}
                block = totals[(str(row["split"]), str(row["profile_id"]))]
                block["images"] += 1
                block["target_atoms"] += atom_count
                for radius in RADII:
                    mapping = (strict_mapping[index].tolist()
                               if radius == 6.0 else
                               radius_mapping(points, targets, radius).values())
                    mapped = ({int(value) for value in mapping
                               if 0 <= int(value) < atom_count})
                    block[f"raw_matched_{int(radius)}"] += len(mapped)
                    block[f"reachable_bonds_{int(radius)}"] += sum(
                        int(left in mapped and right in mapped)
                        for left, right in graph_edges)
                    if radius == 6.0:
                        # The strict matcher is the same function used by the
                        # production curriculum path; keep the element split
                        # only as attribution, never as an input to inference.
                        for atom, atom_id in zip(sample["graph"]["atoms"], range(atom_count)):
                            kind = "visible" if sample["layers"]["atoms"][atom_id].get("visible_text") else "implicit"
                            block[f"target_{kind}"] += 1
                            block[f"matched_{kind}_6"] += int(atom_id in mapped)
    after = state_digest(model)
    if before != after or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen CPU model state changed")
    serial = {f"{split}|{profile}": dict(value)
              for (split, profile), value in sorted(totals.items())}
    report = {
        "schema_version": "chemvector-r17-oa13-oa33-raw-radius-contract-cpu-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "optimizer_steps": 0, "gpu_used": False, "device": "cpu",
        "sealed_test_read": False, "production_checkpoint_written": False,
        "initial_checkpoint_sha256": INITIAL_SHA,
        "initial_checkpoint_sha256_after": digest(INITIAL),
        "manifest_sha256": digest(MANIFEST), "images": len(rows),
        "profiles": list(PROFILES), "molecules_per_split": MOLECULES_PER_SPLIT,
        "radii_px": list(RADII), "rows": serial,
        "contract": {
            "strict_6px": "match_atom_targets(raw,tensors,config)",
            "identity_16px": "greedy post-hoc raw geometry to target anchors",
            "target_geometry_posthoc_only": True,
            "oracle_candidates_injected": False,
        },
        "identity": {"train_molecules": MOLECULES_PER_SPLIT,
                     "holdout_molecules": MOLECULES_PER_SPLIT,
                     "molecule_overlap": 0,
                     "connection_identity_overlap": len(train_connections & holdout_connections),
                     "scaffold_overlap": len(train_scaffolds & holdout_scaffolds)},
        "state_before": before, "state_after": after,
        "gate": {"checkpoint_unchanged": before == after and digest(INITIAL) == INITIAL_SHA,
                 "sealed_test_read": False, "training_performed": False,
                 "production_go": False,
                 "next": "if 16px closes gaps, fix radius contract before atom training; otherwise train raw atom head on visible O/N only"},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "rows": serial,
                      "gate": report["gate"]}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
