#!/usr/bin/env python3
"""Frozen-initial, raw-proposal reachability audit for the r4 O/N panel.

Diagnostic only: the model is kept in eval/no-grad mode, no optimizer is
created, and no proposal is added or relabeled.  Target ownership is accepted
only when one raw proposal and one O/N target are mutually unambiguous under
the fixed geometry contract; Hungarian/nearest assignment is forbidden.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r4"
RENDER_AUDIT = ROOT / "outputs/audits/chemvector-identity-aligned-visible-text-render-20261017-r4/report.json"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-raw-reachability-20261017-r4/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE_SIZE = 768.0
RADIUS = 16.0
OTHER_TARGET_MIN = 20.0
MARGIN = 4.0
STRICT = 6.0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    render_audit = json.loads(RENDER_AUDIT.read_text(encoding="utf-8"))
    if not render_audit["gate"]["accounting_exact"] or not render_audit["gate"]["accepted_artifacts_complete"]:
        raise ValueError("render artifact gate failed")
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != render_audit["accepted_images"]:
        raise ValueError("accepted row count mismatch")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    if digest(INITIAL) != INITIAL_SHA or not torch.cuda.is_available():
        raise RuntimeError("initial checkpoint or CUDA binding failed")
    device = torch.device("cuda:0")
    model.to(device).eval().requires_grad_(False)
    before = state_digest(model)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="diagnostic",
    )
    total = Counter()
    by_cell = defaultdict(Counter)
    unknown = Counter()
    split_ids = defaultdict(set)
    threshold = float(payload["provenance"]["atom_threshold"])
    for index, row in enumerate(rows, 1):
        sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
        batch = collator([sample])
        tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.inference_mode():
            outputs = structure_forward(model, tensors["pixel_values"], config, threshold)
        raw = outputs[3]
        active = torch.where(raw["atom_mask"][0])[0]
        coords = raw["atom_geometry"][0, active, :2].float() * IMAGE_SIZE
        atom_ids = list(batch["atom_ids"][0])
        visible_on_ids = {
            int(atom["atom_id"])
            for atom in sample["layers"]["atoms"]
            if atom.get("visible_text") and atom.get("element") in {"O", "N"}
        }
        target_slots = [slot for slot, atom_id in enumerate(atom_ids) if atom_id in visible_on_ids]
        target_xy = tensors["atom_anchors"][0, target_slots].float() * IMAGE_SIZE
        distance = torch.cdist(coords, target_xy) if target_slots else torch.empty((len(active), 0), device=device)
        split_ids[row["split"]].add(row["molecule_id"])
        total["images"] += 1
        total["target_on"] += len(target_slots)
        total["raw_candidates"] += len(active)
        accepted_candidates = set()
        for target_index, target_slot in enumerate(target_slots):
            element = next(atom["element"] for atom in sample["layers"]["atoms"] if atom["atom_id"] == atom_ids[target_slot])
            nearby = torch.where(distance[:, target_index] <= RADIUS)[0].tolist()
            if len(nearby) != 1:
                unknown["target_zero_or_multiple_candidates"] += 1
                continue
            candidate_index = nearby[0]
            row_distances = distance[candidate_index]
            other = torch.cat((row_distances[:target_index], row_distances[target_index + 1:]))
            nearest_other = float(other.min().item()) if len(other) else float("inf")
            own = float(row_distances[target_index].item())
            if nearest_other <= OTHER_TARGET_MIN:
                unknown["candidate_near_other_target"] += 1
                continue
            if nearest_other - own < MARGIN:
                unknown["target_margin_small"] += 1
                continue
            if candidate_index in accepted_candidates:
                unknown["candidate_reused"] += 1
                continue
            accepted_candidates.add(candidate_index)
            kind = "strict" if own <= STRICT else "refinable"
            cell = (row["split"], row["profile_id"], element, kind)
            by_cell[cell]["selected"] += 1
            total["raw_unique_on"] += 1
            total[f"raw_{kind}"] += 1
        for candidate_index in range(len(active)):
            nearest = float(distance[candidate_index].min().item()) if target_slots else float("inf")
            if candidate_index in accepted_candidates:
                continue
            if nearest > OTHER_TARGET_MIN:
                total["far_negative_candidates"] += 1
            else:
                unknown["unowned_near_candidate"] += 1
        if index % 100 == 0:
            print(json.dumps({"images": index, "raw_unique_on": total["raw_unique_on"]}), flush=True)
    after = state_digest(model)
    if before != after or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("frozen model state changed during replay")
    by_cell_json = {"|".join(map(str, key)): dict(value) for key, value in sorted(by_cell.items())}
    report = {
        "schema_version": "chemvector-identity-aligned-raw-reachability-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "optimizer_steps": 0, "gpu_used": True, "device": str(device),
        "sealed_test_read": False, "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA, "initial_sha256_after": digest(INITIAL),
        "state_before": before, "state_after": after,
        "data_summary_sha256": digest(DATA / "summary.json"),
        "render_audit_sha256": digest(RENDER_AUDIT),
        "threshold": threshold,
        "ownership_contract": {"radius_px": RADIUS, "other_target_min_px": OTHER_TARGET_MIN,
                               "margin_px": MARGIN, "strict_px": STRICT,
                               "hungarian_used": False, "nearest_forced": False,
                               "oracle_candidates_injected": False},
        "totals": dict(total), "unknown": dict(sorted(unknown.items())),
        "by_cell": by_cell_json,
        "identity": {"train_molecules": len(split_ids["train"]), "holdout_molecules": len(split_ids["holdout"]),
                     "overlap": len(split_ids["train"] & split_ids["holdout"])},
        "gate": {"initial_unchanged": before == after, "identity_disjoint": not (split_ids["train"] & split_ids["holdout"]),
                 "raw_replay_complete": True, "admitted_pairs_ambiguity_zero": True,
                 "training_admissible": False, "next": "review raw reachability by cell; no training admission"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "totals": dict(total), "unknown": dict(unknown), "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
