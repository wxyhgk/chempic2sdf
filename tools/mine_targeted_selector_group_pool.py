"""Frozen-initial mining of the serif/large targeted pool.

This is a read-only data diagnostic.  It scans only quarantine images from the
already identity-screened structure-expansion batch, records raw/selected
reachability and strict selector-drop counts, and writes no labels back to the
source.  It does not read sealed data or train a model.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
PUBLIC = ROOT / "outputs/audits/structure-expansion-public-identity-20260923/remaining-candidates.json"
OLD_PLAN = ROOT / "outputs/audits/chemvector-structure-expansion-targeted-split-plan-20261006-r5/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-group-pool-mine-20261006-r1/report.json"
PROFILES = {"oa11", "oa13"}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or immutable initial checkpoint unavailable")
    molecules = {r["molecule_id"]: r for r in json.loads((DATA / "molecules.json").read_text(encoding="utf-8"))}
    samples = json.loads((DATA / "samples.json").read_text(encoding="utf-8"))
    cleared = {key for row in json.loads(PUBLIC.read_text(encoding="utf-8")) for key in row["sample_keys"]}
    old_ids = {row["molecule_id"] for row in json.loads(OLD_PLAN.read_text(encoding="utf-8"))["rows"]}
    rows = [row for row in samples if row["profile_id"] in PROFILES and row["sample_key"] in cleared
            and row["molecule_id"] not in old_ids]
    if not rows:
        raise ValueError("No identity-cleared targeted rows")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval().requires_grad_(False)
    before = state_digest(model)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                                                       bond_queries=config.maximum_bonds), purpose="diagnostic")
    totals = Counter()
    strata = Counter()
    records = []
    with torch.inference_mode():
        for index, row in enumerate(sorted(rows, key=lambda r: r["sample_key"]), 1):
            sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
            if sample["meta"].get("split") != "quarantine" or sample["meta"].get("training_authorized") is not False:
                raise ValueError("Mining source is not quarantine-only")
            batch = collator([sample])
            tensors = {key: value.cuda() for key, value in batch.items() if isinstance(value, torch.Tensor)}
            out = structure_forward(model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"])
            raw, selected = out[3], out[6]
            raw_map, _ = match_atom_targets(raw, tensors, config)
            selected_map, _ = match_atom_targets(selected, tensors, config)
            target_ids = [int(v) for v in batch["atom_ids"][0]]
            raw_ids = {target_ids[int(v)] for v in raw_map[0].tolist() if 0 <= int(v) < len(target_ids)}
            selected_ids = {target_ids[int(v)] for v in selected_map[0].tolist() if 0 <= int(v) < len(target_ids)}
            strict = len(raw_ids - selected_ids)
            target_count = len(target_ids)
            profile = row["profile_id"]
            records.append({"sample_key": row["sample_key"], "molecule_id": row["molecule_id"],
                            "profile_id": profile, "target_atoms": target_count,
                            "raw_matched": len(raw_ids), "selected_matched": len(selected_ids),
                            "raw_gap": target_count - len(raw_ids), "strict_selector_drop": strict})
            totals["images"] += 1
            totals["target_atoms"] += target_count
            totals["raw_matched"] += len(raw_ids)
            totals["selected_matched"] += len(selected_ids)
            totals["raw_gap"] += target_count - len(raw_ids)
            totals["strict_selector_drop"] += strict
            strata[(profile, "strict_positive_image" if strict else "no_strict_positive")] += 1
            if index % 200 == 0:
                print(json.dumps({"images": index, "strict_selector_drop": totals["strict_selector_drop"]}), flush=True)
    after = state_digest(model)
    if before != after or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Frozen model state changed")
    result = {
        "schema_version": "chemvector-targeted-selector-group-pool-mine-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "optimizer_steps": 0, "gpu_used": True, "sealed_test_read": False,
        "initial_sha256": INITIAL_SHA, "state_before": before, "state_after": after,
        "source": {"dataset": str(DATA.relative_to(ROOT)), "public_remaining_sha256": file_digest(PUBLIC),
                    "old_targeted_plan_sha256": file_digest(OLD_PLAN), "profiles": sorted(PROFILES),
                    "rows_after_identity_clear_and_old_slice_exclusion": len(rows)},
        "totals": dict(totals), "strata": {"%s:%s" % key: value for key, value in sorted(strata.items())},
        "records": records,
        "gate": {"quarantine_only": True, "identity_cleared_subset": True,
                 "strict_positive_count": int(totals["strict_selector_drop"]),
                 "training_ready": False,
                 "next": "plan a new identity-disjoint split only if strict positives support a fixed abstention holdout"},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "totals": dict(totals), "strata": result["strata"]}, indent=2))
    return result


if __name__ == "__main__":
    run()
