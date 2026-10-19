"""Read-only preflight for the aromatic-ring-safe chemical-head probe."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.aromatic_ring_objective import (
    AromaticRingObjectiveConfig,
    _aromatic_cycles,
)
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
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.supervision_contract import pair_targets_with_unknown
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.curriculum.chemical_head_probe import _pair_output


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/audits/chemvector-aromatic-ring-safe-preflight-20261001"
CONFIG = AromaticRingObjectiveConfig()


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preflight output: {output}")
    report, plan, protocol = _load_preparation()
    sources = source_checks()
    if file_digest(INITIAL) != report["initial_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint changed")
    if len(plan["samples"]) != 1200:
        raise ValueError("The fixed 1,200-image plan is required")
    loader = RecordLoader(sources["manifest"])
    rows_by_key = {sample_id(row): row for row in sources["manifest"]["samples"]}
    sample_rows = [rows_by_key[sample_id(row)] for row in plan["samples"]]
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=128, bond_queries=256), purpose="training"
    )
    model, config, payload = load_bundle_for_probe(INITIAL, report["initial_checkpoint_sha256"])
    if payload["progress"]["optimizer_steps"] != 0:
        raise ValueError("Preflight requires the zero-step initial checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the fixed preflight")
    torch.set_num_threads(2)
    torch.manual_seed(protocol["seed"])
    torch.cuda.manual_seed_all(protocol["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    model.to(device).eval().requires_grad_(False)

    started = time.monotonic()
    counters = Counter(images=0, no_known_pairs=0, rings=0, below_mean=0,
                       below_minimum=0, below_both=0)
    losses = []
    domains = Counter()
    with torch.no_grad():
        for attempt, row in enumerate(sample_rows, 1):
            sample = loader(sources["manifest"], row)
            batch = collator([sample])
            tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            forward = structure_forward(
                model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
            )
            features, _, _, _, _, _, selected = forward
            from chemvector.train_direct_atom_pair_bond import match_atom_targets

            mapping, _ = match_atom_targets(selected, tensors, config)
            contract = pair_targets_with_unknown(selected, mapping, tensors)
            if not len(contract["pair_indices"]):
                counters["no_known_pairs"] += 1
                counters["images"] += 1
                domains[row["domain"]] += 1
                continue
            output_pair = _pair_output(
                model.pair,
                features,
                selected,
                contract["pair_indices"],
                contract["sample_indices"],
            )
            probabilities = output_pair["chemical_logits"].float().softmax(-1)[:, 3]
            cycles = _aromatic_cycles(
                contract["pair_indices"],
                contract["sample_indices"],
                contract["known_mask"],
                contract["legacy_targets"]["presence"],
                contract["legacy_targets"]["chemical"],
            )
            for cycle in cycles:
                values = probabilities[torch.tensor(cycle, device=probabilities.device)]
                mean = float(values.mean())
                minimum = float(values.min())
                counters["rings"] += 1
                counters["below_mean"] += mean < CONFIG.mean_floor
                counters["below_minimum"] += minimum < CONFIG.minimum_floor
                counters["below_both"] += mean < CONFIG.mean_floor and minimum < CONFIG.minimum_floor
            loss_terms = {
                "chemical_logits": output_pair["chemical_logits"],
            }
            from chemvector.training.curriculum.aromatic_ring_objective import aromatic_ring_safety_loss

            loss, _ = aromatic_ring_safety_loss(
                output_pair["chemical_logits"],
                contract["pair_indices"],
                contract["sample_indices"],
                contract["known_mask"],
                contract["legacy_targets"]["presence"],
                contract["legacy_targets"]["chemical"],
                config=CONFIG,
            )
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"Non-finite aromatic safety loss at attempt {attempt}")
            losses.append(float(loss))
            counters["images"] += 1
            domains[row["domain"]] += 1
            if attempt % 100 == 0:
                print(json.dumps({"attempt": attempt, "rings": counters["rings"]}), flush=True)
    if counters["images"] != 1200:
        raise ValueError("Fixed preflight coverage incomplete")
    report_out = {
        "schema_version": "chemvector-aromatic-ring-safe-preflight-0.1",
        "complete": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "backward_passes": 0,
        "gpu_used": True,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "production_modified": False,
        "objective": {
            "mean_floor": CONFIG.mean_floor,
            "minimum_floor": CONFIG.minimum_floor,
            "mean_weight": CONFIG.mean_weight,
            "minimum_weight": CONFIG.minimum_weight,
        },
        "fixed_training_images": counters["images"],
        "domains": dict(sorted(domains.items())),
        "counters": dict(counters),
        "loss": {
            "images_with_reachable_rings": len(losses),
            "mean": sum(losses) / len(losses) if losses else 0.0,
            "max": max(losses, default=0.0),
        },
        "source_sha256": {
            "preparation_report.json": file_digest(PREPARATION / "report.json"),
            "training_samples.json": file_digest(SAMPLES),
            "initial.pt": file_digest(INITIAL),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report_out)
    return report_out


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "counters": result["counters"]}))
