#!/usr/bin/env python3
"""Read-only raw/selected endpoint and pair reachability audit for r17 m128.

The m128 graph sidecar cache records only the selected AtomObject set.  This
audit replays the *same* identity-disjoint panel through the frozen initial
checkpoint so that raw AtomObject proposals and post-set-selector proposals
can be compared directly.  It does not train a head, write a checkpoint, or
read any sealed evaluation data.

The report separates four causes for a target bond:

* ``raw_endpoint_gap``: at least one target endpoint was absent from raw
  proposals;
* ``endpoint_selector_drop``: both endpoints were raw-matched but the set
  selector removed at least one;
* ``raw_pair_gap``: raw endpoints exist but the frozen pair presence proposal
  did not pass the fixed threshold;
* ``selected_pair_gap``: selected endpoints exist but the selected pair did
  not pass the same threshold.

Ground-truth layers/graphs are joined only after the frozen forward pass for
matching and stratification.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

if str(ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets
from chemvector.components.graph.joint.direct_atom_pair_bond import enumerate_unordered_atom_pairs

CACHE = ROOT / "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m128-t0p72-c1-panel128/feature-cache.pt"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
PAIR_THRESHOLD = 0.72
DEFAULT_OUT = ROOT / "outputs/audits/chemvector-r17-m128-raw-selected-reachability-v1"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _empty() -> dict[str, int]:
    return {
        "images": 0,
        "target_atoms": 0,
        "raw_matched": 0,
        "selected_matched": 0,
        "raw_gap": 0,
        "selector_drop": 0,
        "target_bonds": 0,
        "raw_endpoint_reachable": 0,
        "selected_endpoint_reachable": 0,
        "raw_endpoint_gap": 0,
        "endpoint_selector_drop": 0,
        "raw_pair_reachable": 0,
        "selected_pair_reachable": 0,
        "raw_pair_gap": 0,
        "selected_pair_gap": 0,
    }


def _add_atom(bucket: dict[str, int], raw_ok: bool, selected_ok: bool) -> None:
    bucket["target_atoms"] += 1
    bucket["raw_matched"] += int(raw_ok)
    bucket["selected_matched"] += int(selected_ok)
    bucket["raw_gap"] += int(not raw_ok)
    bucket["selector_drop"] += int(raw_ok and not selected_ok)


def _add_bond(bucket: dict[str, int], *, raw_endpoint: bool, selected_endpoint: bool,
              raw_pair: bool, selected_pair: bool) -> None:
    bucket["target_bonds"] += 1
    bucket["raw_endpoint_reachable"] += int(raw_endpoint)
    bucket["selected_endpoint_reachable"] += int(selected_endpoint)
    bucket["raw_endpoint_gap"] += int(not raw_endpoint)
    bucket["endpoint_selector_drop"] += int(raw_endpoint and not selected_endpoint)
    bucket["raw_pair_reachable"] += int(raw_pair)
    bucket["selected_pair_reachable"] += int(selected_pair)
    bucket["raw_pair_gap"] += int(raw_endpoint and not raw_pair)
    bucket["selected_pair_gap"] += int(selected_endpoint and not selected_pair)


def _rates(value: dict[str, int]) -> dict[str, float]:
    out = dict(value)
    atoms = max(value["target_atoms"], 1)
    bonds = max(value["target_bonds"], 1)
    for key in ("raw_matched", "selected_matched", "raw_gap", "selector_drop"):
        out[f"{key}_rate"] = value[key] / atoms
    for key in (
        "raw_endpoint_reachable", "selected_endpoint_reachable", "raw_endpoint_gap",
        "endpoint_selector_drop", "raw_pair_reachable", "selected_pair_reachable",
        "raw_pair_gap", "selected_pair_gap",
    ):
        out[f"{key}_rate"] = value[key] / bonds
    return out


def _category(atom: dict) -> str:
    element = str(atom.get("element", "UNKNOWN"))
    visible = bool(atom.get("visible_text", False))
    if visible and element in {"O", "N"}:
        return f"visible:{element}"
    if not visible and element == "C":
        return "implicit:C"
    return f"{('visible' if visible else 'implicit')}:{element}"


def _target_graph(sample: dict) -> tuple[dict[int, dict], list[tuple[int, int, str]]]:
    atoms = {int(a["atom_id"]): a for a in sample["layers"]["atoms"]}
    bonds = []
    for bond in sample["layers"]["bonds"]:
        drawing = bond["drawing"]
        left, right = sorted((int(drawing["begin_atom"]), int(drawing["end_atom"])))
        kind = str(bond.get("chemical_type", drawing.get("bond_type", "UNKNOWN"))).upper()
        bonds.append((left, right, kind))
    return atoms, bonds


def _pair_targets(output0: torch.Tensor, atom_output: dict, model, threshold: float):
    """Return endpoint-target maps and target pair set for one atom output."""
    pairs, sample_indices = enumerate_unordered_atom_pairs(atom_output["atom_mask"])
    if not len(pairs):
        return {}, set(), 0
    direct = model.pair(
        output0,
        atom_output["atom_features"],
        atom_output["atom_geometry"][..., :2],
        atom_output["atom_scores"],
        pairs,
        sample_indices,
    )
    chosen = torch.where(direct["presence_logits"].float().sigmoid() >= PAIR_THRESHOLD)[0].tolist()
    return pairs, set(chosen), len(pairs)


def _map_target_pairs(mapping: dict[int, int], pairs: torch.Tensor, chosen: set[int], target_ids: list[int]) -> set[tuple[int, int]]:
    slot_to_target = {}
    for slot, target_index in mapping.items():
        if 0 <= target_index < len(target_ids):
            slot_to_target[int(slot)] = int(target_ids[target_index])
    result = set()
    for index in chosen:
        left, right = (int(value) for value in pairs[index].tolist())
        if left in slot_to_target and right in slot_to_target and slot_to_target[left] != slot_to_target[right]:
            result.add(tuple(sorted((slot_to_target[left], slot_to_target[right]))))
    return result


def run(*, output: Path = DEFAULT_OUT, limit: int | None = None, num_threads: int = 1) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if _sha(INITIAL) != INITIAL_SHA:
        raise RuntimeError("frozen initial checkpoint SHA changed")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    records = cache.get("records", {})
    expected = {"train": 512, "dev": 256, "fresh_holdout": 512}
    if {k: len(v) for k, v in records.items()} != expected:
        raise RuntimeError(f"unexpected m128 panel counts: { {k: len(v) for k, v in records.items()} }")
    ordered = [(split, record) for split in ("train", "dev", "fresh_holdout") for record in records[split]]
    if limit is not None:
        if not 1 <= limit <= len(ordered):
            raise ValueError("limit must be within panel size")
        ordered = ordered[:limit]

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.eval().requires_grad_(False)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="diagnostic",
    )
    atom_threshold = float(payload["provenance"]["atom_threshold"])
    state_before = state_digest(model)
    totals = _empty()
    by_split: dict[str, dict] = defaultdict(_empty)
    by_profile: dict[str, dict] = defaultdict(_empty)
    by_split_profile: dict[str, dict] = defaultdict(_empty)
    by_category: dict[str, dict] = defaultdict(_empty)
    by_bond_type: dict[str, dict] = defaultdict(_empty)
    rows = []
    with torch.inference_mode():
        for index, (split, record) in enumerate(ordered, 1):
            key = str(record["sample_key"])
            if "sealed_test" in key:
                raise RuntimeError("sealed_test key is forbidden")
            sample = load_sample(SOURCE, key, purpose="diagnostic")
            meta = sample["meta"]
            if meta.get("split") != "quarantine" or meta.get("training_authorized") is not False:
                raise RuntimeError(f"panel source is not quarantine-only: {key}")
            profile = str(meta.get("profile_id", key.rsplit("__", 1)[-1]))
            atoms, bonds = _target_graph(sample)
            batch = collator([sample])
            tensors = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = structure_forward(model, tensors["pixel_values"], config, atom_threshold)
            output0, raw, selected = outputs[0], outputs[3], outputs[6]
            raw_mapping, _ = match_atom_targets(raw, tensors, config)
            selected_mapping, _ = match_atom_targets(selected, tensors, config)
            target_ids = [int(v) for v in batch["atom_ids"][0]]
            raw_map = {int(slot): int(target_ids[int(target_index)]) for slot, target_index in enumerate(raw_mapping[0].tolist())
                       if int(target_index) >= 0 and int(target_index) < len(target_ids)}
            selected_map = {int(slot): int(target_ids[int(target_index)]) for slot, target_index in enumerate(selected_mapping[0].tolist())
                            if int(target_index) >= 0 and int(target_index) < len(target_ids)}
            raw_targets, selected_targets = set(raw_map.values()), set(selected_map.values())
            raw_pairs, raw_chosen, _ = _pair_targets(output0, raw, model, PAIR_THRESHOLD)
            selected_pairs, selected_chosen, _ = _pair_targets(output0, selected, model, PAIR_THRESHOLD)
            raw_pair_targets = _map_target_pairs(raw_map, raw_pairs, raw_chosen, target_ids)
            selected_pair_targets = _map_target_pairs(selected_map, selected_pairs, selected_chosen, target_ids)
            image_bucket = _empty()
            image_bucket["images"] = 1
            for atom_id, atom in atoms.items():
                raw_ok, selected_ok = atom_id in raw_targets, atom_id in selected_targets
                for bucket in (totals, by_split[split], by_profile[profile], by_split_profile[f"{split}:{profile}"], by_category[_category(atom)], image_bucket):
                    _add_atom(bucket, raw_ok, selected_ok)
            for left, right, kind in bonds:
                raw_endpoint = left in raw_targets and right in raw_targets
                selected_endpoint = left in selected_targets and right in selected_targets
                raw_pair = (left, right) in raw_pair_targets
                selected_pair = (left, right) in selected_pair_targets
                for bucket in (totals, by_split[split], by_profile[profile], by_split_profile[f"{split}:{profile}"], by_bond_type[kind], image_bucket):
                    _add_bond(bucket, raw_endpoint=raw_endpoint, selected_endpoint=selected_endpoint,
                              raw_pair=raw_pair, selected_pair=selected_pair)
            for bucket in (totals, by_split[split], by_profile[profile], by_split_profile[f"{split}:{profile}"], image_bucket):
                bucket["images"] += 1
            rows.append({
                "sample_key": key, "split": split, "profile": profile,
                "target_atoms": len(atoms), "raw_atoms": len(raw_targets), "selected_atoms": len(selected_targets),
                "raw_gap_atoms": len(set(atoms) - raw_targets), "selector_drop_atoms": len(raw_targets - selected_targets),
                "target_bonds": len(bonds),
                "raw_endpoint_reachable": sum(int(a in raw_targets and b in raw_targets) for a, b, _ in bonds),
                "selected_endpoint_reachable": sum(int(a in selected_targets and b in selected_targets) for a, b, _ in bonds),
                "raw_pair_reachable": sum(int((a, b) in raw_pair_targets) for a, b, _ in bonds),
                "selected_pair_reachable": sum(int((a, b) in selected_pair_targets) for a, b, _ in bonds),
                "raw_proposal_slots": int(raw["atom_mask"][0].sum()), "selected_slots": int(selected["atom_mask"][0].sum()),
            })
            if index % 100 == 0:
                print(json.dumps({"phase": "replay", "images": index, "total": len(ordered),
                                  "raw_atom_gap": totals["raw_gap"], "selector_atom_drop": totals["selector_drop"]}), flush=True)
    state_after = state_digest(model)
    if state_before != state_after or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("frozen replay changed model state or produced gradients")
    result = {
        "schema_version": "chemvector-r17-m128-raw-selected-reachability-0.1",
        "complete": limit is None, "read_only": True, "training_performed": False,
        "optimizer_steps": 0, "gpu_used": False, "device": "cpu", "sealed_test_read": False,
        "production_checkpoint_written": False, "training_admissible": False,
        "initial_sha256": INITIAL_SHA, "initial_state_before_sha256": state_before,
        "initial_state_after_sha256": state_after, "cache": str(CACHE.relative_to(ROOT)),
        "cache_sha256": _sha(CACHE), "panel_counts": expected, "images": len(ordered),
        "pair_threshold": PAIR_THRESHOLD, "atom_threshold": atom_threshold,
        "definitions": {"raw_gap": "target atom absent after raw proposal matching",
                        "selector_drop": "raw-matched target absent after set selector",
                        "raw_endpoint_gap": "at least one target bond endpoint raw-unmatched",
                        "endpoint_selector_drop": "both endpoints raw-matched but selected misses one",
                        "raw_pair_gap": "raw endpoints present but raw pair presence below threshold",
                        "selected_pair_gap": "selected endpoints present but selected pair below threshold"},
        "totals": _rates(totals),
        "by_split": {k: _rates(v) for k, v in sorted(by_split.items())},
        "by_profile": {k: _rates(v) for k, v in sorted(by_profile.items())},
        "by_split_profile": {k: _rates(v) for k, v in sorted(by_split_profile.items())},
        "by_category": {k: _rates(v) for k, v in sorted(by_category.items())},
        "by_bond_type": {k: _rates(v) for k, v in sorted(by_bond_type.items())},
        "rows": rows,
        "gate": {"identity_disjoint_panel": True, "panel_complete": limit is None,
                 "checkpoint_unchanged": state_before == state_after and _sha(INITIAL) == INITIAL_SHA,
                 "sealed_test_read": False, "production_go": False,
                 "training_go": False},
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output / 'report.json'), "images": len(ordered), "totals": result["totals"], "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args(argv)
    run(output=args.output, limit=args.limit, num_threads=args.num_threads)


if __name__ == "__main__":
    main()
