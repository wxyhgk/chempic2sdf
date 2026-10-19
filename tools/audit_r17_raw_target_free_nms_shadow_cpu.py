"""Target-free spatial-NMS shadow for the frozen raw atom decoder.

NMS decisions use only raw proposal anchors and scores.  Ground-truth anchors
are joined after NMS for a 16 px identity audit, so this script cannot improve
its own result by looking at targets.  It is a bounded CPU diagnostic, not a
decoder or checkpoint change.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.audit_r17_oa13_oa33_raw_radius_contract_cpu import choose_rows, digest


SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
OUT = ROOT / "outputs/audits/chemvector-r17-raw-target-free-nms-shadow-cpu-v1"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
NMS_RADII = (0.0, 4.0, 6.0, 8.0, 10.0)
MATCH_RADIUS = 16.0


def nms(points: torch.Tensor, scores: torch.Tensor, radius: float) -> torch.Tensor:
    order = torch.argsort(scores, descending=True)
    kept: list[int] = []
    for index in order.tolist():
        if not kept or bool((torch.linalg.vector_norm(
                points[kept] - points[index], dim=1) * 768.0 > radius).all()):
            kept.append(index)
    return torch.tensor(kept, dtype=torch.long)


def match(points: torch.Tensor, targets: torch.Tensor) -> set[int]:
    distance = torch.cdist(points.float() * 768.0, targets.float() * 768.0)
    pairs = [(float(distance[i, j]), i, j)
             for i in range(distance.size(0)) for j in range(distance.size(1))
             if float(distance[i, j]) <= MATCH_RADIUS]
    result: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in sorted(pairs):
        if i not in result and j not in used:
            result[i] = j
            used.add(j)
    return set(result.values())


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("initial checkpoint SHA changed")
    rows = choose_rows()
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cpu().eval().requires_grad_(False)
    before = state_digest(model)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                            bond_queries=config.maximum_bonds), purpose="diagnostic")
    stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    with torch.inference_mode():
        for offset in range(0, len(rows), 8):
            items = rows[offset:offset + 8]
            samples = [load_sample(SOURCE, row["sample_key"], purpose="diagnostic")
                       for row in items]
            batch = collator(samples)
            tensors = {key: value for key, value in batch.items()
                       if isinstance(value, torch.Tensor)}
            outputs = structure_forward(model, tensors["pixel_values"], config,
                                       payload["provenance"]["atom_threshold"])
            raw = outputs[3]
            for index, row in enumerate(items):
                block = stats[(str(row["split"]), str(row["profile_id"]))]
                count = int(tensors["atom_mask"][index].sum())
                targets = tensors["atom_anchors"][index, :count]
                active = raw["atom_mask"][index]
                points = raw["atom_geometry"][index, active, :2]
                scores = raw["atom_scores"][index, active]
                edges = {(min(int(bond["begin_atom"]), int(bond["end_atom"])),
                          max(int(bond["begin_atom"]), int(bond["end_atom"])))
                         for bond in samples[index]["graph"]["bonds"]}
                block["images"] += 1
                block["target_atoms"] += count
                for radius in NMS_RADII:
                    kept = torch.arange(len(points)) if radius == 0 else nms(points, scores, radius)
                    mapped = match(points[kept], targets)
                    label = f"nms{int(radius)}"
                    block[f"{label}_candidates"] += len(kept)
                    block[f"{label}_matched_16"] += len(mapped)
                    block[f"{label}_edges_16"] += sum(int(left in mapped and right in mapped)
                                                        for left, right in edges)
    after = state_digest(model)
    if before != after or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen CPU state changed")
    serial = {f"{split}|{profile}": dict(value)
              for (split, profile), value in sorted(stats.items())}
    report = {
        "schema_version": "chemvector-r17-raw-target-free-nms-shadow-cpu-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "optimizer_steps": 0, "gpu_used": False, "device": "cpu",
        "sealed_test_read": False, "production_checkpoint_written": False,
        "initial_checkpoint_sha256": INITIAL_SHA,
        "initial_checkpoint_sha256_after": digest(INITIAL),
        "images": len(rows), "nms_radii_px": list(NMS_RADII),
        "posthoc_match_radius_px": MATCH_RADIUS, "rows": serial,
        "contract": {"nms_target_free": True, "target_geometry_posthoc_only": True,
                     "oracle_candidates_injected": False},
        "state_before": before, "state_after": after,
        "gate": {"checkpoint_unchanged": before == after and digest(INITIAL) == INITIAL_SHA,
                 "sealed_test_read": False, "training_performed": False,
                 "production_go": False,
                 "next": "run identity/precision/sanitize replay before considering any NMS radius"},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "rows": serial,
                      "gate": report["gate"]}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
