"""Bounded frozen-initial replay on the 10k release plan.

The source remains quarantine-only and is opened with the diagnostic loader.
This measures raw/selected atom reachability by profile before any admission or
training; no labels are fed back to the model and no checkpoint is written.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/manifest.json"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_10K_REPLAY_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-structure-expansion-10k-frozen-replay-20261016-r1/report.json"),
))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
MOLECULES_PER_SPLIT = int(os.environ.get("CHEMVECTOR_10K_REPLAY_MOLECULES_PER_SPLIT", "40"))
TRAIN_SLICE = int(os.environ.get("CHEMVECTOR_10K_REPLAY_TRAIN_MOLECULES", str(MOLECULES_PER_SPLIT)))
HOLDOUT_SLICE = int(os.environ.get("CHEMVECTOR_10K_REPLAY_HOLDOUT_MOLECULES", str(MOLECULES_PER_SPLIT)))
BATCH_SIZE = 4


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    rows = plan["rows"]
    by_molecule: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_molecule[row["molecule_id"]].append(row)
    selected_ids = []
    for split in ("train", "holdout"):
        ids = sorted({mid for mid, group in by_molecule.items() if group[0]["split"] == split})
        selected_ids.extend(ids[:(TRAIN_SLICE if split == "train" else HOLDOUT_SLICE)])
    selected = [row for row in rows if row["molecule_id"] in set(selected_ids)]
    if len(selected) != (TRAIN_SLICE + HOLDOUT_SLICE) * 8:
        raise ValueError("fixed replay slice does not contain exactly 8 profiles per molecule")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    if digest(INITIAL) != INITIAL_SHA:
        raise ValueError("initial checkpoint binding failed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this bounded replay")
    device = torch.device("cuda:0")
    model.to(device).eval().requires_grad_(False)
    before = state_digest(model)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                                                       bond_queries=config.maximum_bonds), purpose="diagnostic")
    totals = Counter(); by_profile = defaultdict(Counter); by_split = defaultdict(Counter)
    pending = []

    def flush(items: list[dict]) -> None:
        if not items:
            return
        samples = [load_sample(SOURCE, row["sample_key"], purpose="diagnostic") for row in items]
        batch = collator(samples)
        tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        with torch.inference_mode():
            _, _, _, raw, _, _, selected_out = structure_forward(model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"])
        raw_map, _ = match_atom_targets(raw, tensors, config)
        selected_map, _ = match_atom_targets(selected_out, tensors, config)
        for i, row in enumerate(items):
            graph = samples[i]["graph"]
            target_ids = {int(atom["atom_id"]) for atom in graph["atoms"]}
            # Collator target order is stable; matched indices are mapped to
            # semantic IDs solely for this diagnostic count.
            atom_ids = [int(value) for value in batch["atom_ids"][i]]
            raw_ids = {atom_ids[int(v)] for v in raw_map[i].tolist() if 0 <= int(v) < len(atom_ids)}
            selected_ids_local = {atom_ids[int(v)] for v in selected_map[i].tolist() if 0 <= int(v) < len(atom_ids)}
            block = by_profile[row["profile_id"]]; split_block = by_split[row["split"]]
            values = {"target_atoms": len(target_ids), "raw_matched": len(raw_ids & target_ids),
                      "selected_matched": len(selected_ids_local & target_ids)}
            for key, value in values.items():
                totals[key] += value; block[key] += value; split_block[key] += value
            totals["images"] += 1; by_profile[row["profile_id"]]["images"] += 1; by_split[row["split"]]["images"] += 1
            totals["raw_gap"] += len(target_ids - raw_ids); totals["selector_drop"] += len(raw_ids - selected_ids_local)
            block["raw_gap"] += len(target_ids - raw_ids); block["selector_drop"] += len(raw_ids - selected_ids_local)
            split_block["raw_gap"] += len(target_ids - raw_ids); split_block["selector_drop"] += len(raw_ids - selected_ids_local)

    for row in selected:
        pending.append(row)
        if len(pending) == BATCH_SIZE:
            flush(pending); pending = []
    flush(pending)
    after = state_digest(model)
    if before != after or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen initial state changed")
    def ratios(value: Counter) -> dict:
        result = dict(value)
        result["raw_reachability"] = value["raw_matched"] / max(value["target_atoms"], 1)
        result["selected_reachability"] = value["selected_matched"] / max(value["target_atoms"], 1)
        result["selector_drop_rate_of_raw"] = value["selector_drop"] / max(value["raw_matched"], 1)
        return result
    profile_metrics = {k: ratios(v) for k, v in sorted(by_profile.items())}
    report = {"schema_version": "chemvector-structure-expansion-10k-frozen-replay-0.1",
              "complete": True, "read_only": True, "training_performed": False,
              "optimizer_steps": 0, "gpu_used": True, "device": str(device),
              "sealed_test_read": False, "deployment_authorized": False,
              "initial_sha256": INITIAL_SHA, "initial_sha256_after": digest(INITIAL),
              "plan_sha256": digest(PLAN), "source_summary_sha256": digest(SOURCE / "summary.json"),
              "slice": {"train_molecules": TRAIN_SLICE, "holdout_molecules": HOLDOUT_SLICE, "images": len(selected),
                        "profiles": sorted({row["profile_id"] for row in selected})},
              "totals": ratios(totals), "by_profile": profile_metrics,
              "by_split": {k: ratios(v) for k, v in sorted(by_split.items())},
              "state_before": before, "state_after": after,
              "gate": {"initial_unchanged": before == after, "sealed_false": True,
                       "aggregate_raw_reachability_ge_098": (totals["raw_matched"] / max(totals["target_atoms"], 1)) >= .98,
                       "each_profile_raw_reachability_ge_098": all(value["raw_reachability"] >= .98 for value in profile_metrics.values()),
                       "training_admissible": False,
                       "next": "complete release/admission and full replay only if this slice has raw reachability >= 0.98"}}
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "totals": report["totals"], "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
