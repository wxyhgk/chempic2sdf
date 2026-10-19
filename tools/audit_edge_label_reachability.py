"""Read-only audit of edge-label reachability on the fixed chemical-head train plan.

This audit performs no optimizer step and never reads the sealed test set.  It
replays the initial model's selected atoms/matching on the exact 1,200 training
rows bound to the chemical-head preparation, then reports which pair labels are
actually available to the current supervision contract and sampler.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum.chemical_head_probe import (
    INITIAL,
    PREPARATION,
    SAMPLES,
    _load_preparation,
)
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.interface_diagnostic import write_json
from chemvector.training.curriculum.pair_scale_plan import sample_id
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.supervision_contract import pair_targets_with_unknown, select_known_pair_rows
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005"


def _inc(counter: Counter, key: str, value: int = 1) -> None:
    counter[key] += int(value)


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output}")
    report, plan, protocol = _load_preparation()
    sources = source_checks()
    if file_digest(INITIAL) != report["initial_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint changed")
    if len(plan["samples"]) != 1200 or plan.get("training_only") is not True:
        raise ValueError("The fixed 1,200-image training-only plan is required")

    loader = RecordLoader(sources["manifest"])
    rows_by_key = {sample_id(row): row for row in sources["manifest"]["samples"]}
    sample_rows = [rows_by_key[sample_id(row)] for row in plan["samples"]]
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=128, bond_queries=256), purpose="training")
    model, config, payload = load_bundle_for_probe(INITIAL, report["initial_checkpoint_sha256"])
    if payload["progress"]["optimizer_steps"] != 0:
        raise ValueError("Audit requires the zero-step initial checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fixed replay audit")
    torch.set_num_threads(2)
    torch.manual_seed(protocol["seed"])
    torch.cuda.manual_seed_all(protocol["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    model.to(device).eval().requires_grad_(False)

    totals = Counter()
    domains = Counter()
    bond_types = Counter()
    sampled_bond_types = Counter()
    per_sample = []
    started = time.monotonic()
    with torch.no_grad():
        for attempt, row in enumerate(sample_rows, 1):
            sample = loader(sources["manifest"], row)
            batch = collator([sample])
            tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            forward = structure_forward(model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"])
            features, _, _, _, _, _, selected = forward
            mapping, _ = match_atom_targets(selected, tensors, config)
            contract = pair_targets_with_unknown(selected, mapping, tensors)
            pairs, rows_idx = contract["pair_indices"], contract["sample_indices"]
            targets = contract["legacy_targets"]
            known = contract["known_mask"]
            positive = targets["presence"]
            chemical = targets["chemical"]
            sample_stats = Counter()
            sample_stats["candidate_pairs"] = len(pairs)
            sample_stats["known_pairs"] = int(known.sum())
            sample_stats["unknown_pairs"] = int((~known).sum())
            sample_stats["confirmed_positive_pairs"] = int((known & positive).sum())
            sample_stats["confirmed_nonedge_pairs"] = int((known & ~positive).sum())
            sample_stats["unreachable_target_bonds"] = int(contract["coverage"]["unreachable_bonds"])
            # Chemical labels are meaningful only on confirmed positive rows.
            for type_id, name in enumerate(BOND_TYPES):
                count = int((known & positive & (chemical == type_id)).sum())
                _inc(bond_types, name, count)
            chosen = select_known_pair_rows(contract, selected, negative_ratio=6, minimum_negatives=48)
            sample_stats["sampled_rows"] = len(chosen)
            sample_stats["sampled_positive_pairs"] = int(positive[chosen].sum()) if len(chosen) else 0
            sample_stats["sampled_confirmed_nonedges"] = int((known[chosen] & ~positive[chosen]).sum()) if len(chosen) else 0
            for type_id, name in enumerate(BOND_TYPES):
                count = int((positive[chosen] & (chemical[chosen] == type_id)).sum()) if len(chosen) else 0
                _inc(sampled_bond_types, name, count)
            for key, value in sample_stats.items():
                _inc(totals, key, value)
            domains[row["domain"]] += 1
            per_sample.append({"sample_key": row["sample_key"], "domain": row["domain"], **dict(sample_stats)})
            if attempt % 100 == 0:
                print(json.dumps({"attempt": attempt, "known": totals["known_pairs"], "positive": totals["confirmed_positive_pairs"]}), flush=True)

    if len(per_sample) != 1200:
        raise ValueError("Fixed 1,200-image coverage incomplete")
    result = {
        "schema_version": "chemvector-edge-label-reachability-0.1",
        "complete": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "gpu_used": True,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "fixed_training_images": len(per_sample),
        "domains": dict(sorted(domains.items())),
        "totals": dict(sorted(totals.items())),
        "confirmed_positive_bond_types": dict(sorted(bond_types.items())),
        "sampled_positive_bond_types": dict(sorted(sampled_bond_types.items())),
        "bond_type_vocabulary": list(BOND_TYPES),
        "supervision_contract": {
            "known_pair_definition": "both selected endpoints matched to distinct target atoms",
            "confirmed_nonedge_definition": "known pair with no target bond",
            "unknown_definition": "at least one selected endpoint unmatched",
            "chemical_loss_rows": "confirmed positive rows only",
            "negative_sampler": "known pairs, max(48, 6*positive_count) negatives per image",
        },
        "source_sha256": {
            "preparation_report.json": file_digest(PREPARATION / "report.json"),
            "training-samples.json": file_digest(SAMPLES),
            "initial.pt": file_digest(INITIAL),
        },
        "elapsed_seconds": time.monotonic() - started,
        "per_sample": per_sample,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", result)
    return result


if __name__ == "__main__":
    value = run()
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "totals": value["totals"]}, indent=2))
