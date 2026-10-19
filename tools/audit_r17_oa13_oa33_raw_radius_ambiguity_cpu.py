"""Measure ambiguity introduced by the 16 px raw-identity audit radius.

This is a companion to ``audit_r17_oa13_oa33_raw_radius_contract_cpu.py``.
It does not alter decoding or train anything.  For a small identity-disjoint
slice, target coordinates are joined after frozen inference and only used to
measure whether a 16 px radius is one-to-one, ambiguous, or margin-safe.
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


MANIFEST = ROOT / "outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/manifest.json"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
OUT = ROOT / "outputs/audits/chemvector-r17-oa13-oa33-raw-radius-ambiguity-cpu-v1"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
PROFILES = ("oa00", "oa13", "oa33")
MOLECULES_PER_SPLIT = 8
RADIUS = 16.0
MARGIN = 4.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_rows() -> list[dict]:
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))["rows"]
    selected = []
    for split in ("train", "holdout"):
        ids = sorted({str(row["molecule_id"]) for row in rows if row["split"] == split},
                     key=lambda value: hashlib.sha256(
                         ("r17-raw-radius-contract-v1:" + value).encode()).hexdigest())
        keep = set(ids[:MOLECULES_PER_SPLIT])
        selected.extend(row for row in rows
                        if row["split"] == split and row["molecule_id"] in keep
                        and row["profile_id"] in PROFILES)
    selected.sort(key=lambda row: row["sample_key"])
    if len(selected) != 48:
        raise ValueError("identity slice is not complete")
    return selected


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("initial checkpoint SHA changed")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    device = torch.device("cpu")
    model.to(device).eval().requires_grad_(False)
    before = state_digest(model)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                            bond_queries=config.maximum_bonds), purpose="diagnostic")
    rows = choose_rows()
    stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    with torch.inference_mode():
        for offset in range(0, len(rows), 8):
            items = rows[offset:offset + 8]
            samples = [load_sample(SOURCE, row["sample_key"], purpose="diagnostic") for row in items]
            batch = collator(samples)
            tensors = {key: value.to(device) for key, value in batch.items()
                       if isinstance(value, torch.Tensor)}
            _, _, _, raw, _, _, _ = structure_forward(
                model, tensors["pixel_values"], config,
                payload["provenance"]["atom_threshold"])
            for index, row in enumerate(items):
                block = stats[(str(row["split"]), str(row["profile_id"]))]
                n = int(tensors["atom_mask"][index].sum())
                target = tensors["atom_anchors"][index, :n].float() * 768.0
                active = raw["atom_mask"][index]
                points = raw["atom_geometry"][index, active, :2].float() * 768.0
                distance = torch.cdist(points, target)
                block["images"] += 1
                block["target_atoms"] += n
                block["raw_candidates"] += len(points)
                for candidate in range(distance.size(0)):
                    nearby = distance[candidate] <= RADIUS
                    count = int(nearby.sum())
                    block["candidate_within_16"] += int(count > 0)
                    block["candidate_ambiguous"] += int(count > 1)
                    if count:
                        values = torch.sort(distance[candidate])[0]
                        block["candidate_margin_ge_4"] += int(float(values[1] - values[0]) >= MARGIN
                                                             if len(values) > 1 else 1)
                for target_index in range(distance.size(1)):
                    nearby = distance[:, target_index] <= RADIUS
                    count = int(nearby.sum())
                    block["target_covered_16"] += int(count > 0)
                    block["target_ambiguous"] += int(count > 1)
                    if count:
                        values = torch.sort(distance[:, target_index])[0]
                        block["target_margin_ge_4"] += int(float(values[1] - values[0]) >= MARGIN
                                                          if len(values) > 1 else 1)
    after = state_digest(model)
    if before != after or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen CPU state changed")
    serial = {f"{split}|{profile}": dict(value)
              for (split, profile), value in sorted(stats.items())}
    report = {
        "schema_version": "chemvector-r17-oa13-oa33-raw-radius-ambiguity-cpu-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "optimizer_steps": 0, "gpu_used": False, "device": "cpu",
        "sealed_test_read": False, "production_checkpoint_written": False,
        "initial_checkpoint_sha256": INITIAL_SHA,
        "initial_checkpoint_sha256_after": digest(INITIAL),
        "manifest_sha256": digest(MANIFEST), "images": len(rows),
        "radius_px": RADIUS, "ambiguity_margin_px": MARGIN,
        "rows": serial,
        "contract": {"target_geometry_posthoc_only": True,
                     "oracle_candidates_injected": False,
                     "candidate_ambiguity": "number of GT targets within radius per raw proposal",
                     "target_ambiguity": "number of raw proposals within radius per GT target"},
        "state_before": before, "state_after": after,
        "gate": {"checkpoint_unchanged": before == after and digest(INITIAL) == INITIAL_SHA,
                 "sealed_test_read": False, "training_performed": False,
                 "production_go": False,
                 "next": "admit 16px identity only if ambiguous candidates remain explicitly abstained"},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "rows": serial,
                      "gate": report["gate"]}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
