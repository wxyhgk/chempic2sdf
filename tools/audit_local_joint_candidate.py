"""Forward-only audit of local predicted atom-pair/bond candidates.

The audit keeps the deployed ``initial.pt`` structure path frozen.  It does
not add target atoms, oracle pairs, or optimizer updates.  Pairs are proposed
from predicted atom geometry only; labels are attached afterwards through the
existing strict selected-atom assignment.  A pair with either unmatched
endpoint is reported as unknown and is never counted as a negative.

The output is an independent audit artifact.  It is deliberately not wired
into the production selector, set head, or inference profile.
"""

from __future__ import annotations

from collections import Counter
import json
import os
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
from chemvector.training.curriculum.interface_diagnostic import tensor_tree_digest, write_json
from chemvector.training.curriculum.pair_scale_plan import sample_id
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.supervision_contract import pair_targets_with_unknown
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import (
    match_atom_targets,
)
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from tools.current_initial_cohort import split_group_disjoint


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_LOCAL_JOINT_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-local-joint-candidate-20261009"),
))

IMAGE_SIZE = 768.0
# These are inference-only proposal constants.  They are intentionally fixed
# before the audit and never derived from target bond coordinates.
LOCAL_MIN_DISTANCE_PX = 8.0
LOCAL_MAX_DISTANCE_PX = float(os.environ.get("CHEMVECTOR_LOCAL_MAX_DISTANCE_PX", "192"))
TOP_KS = (1, 2, 4, 8, 16)
PAIR_THRESHOLD = 0.85


def local_pair_mask(
    atom_geometry: torch.Tensor,
    atom_mask: torch.Tensor,
    pair_indices: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    min_distance_px: float = LOCAL_MIN_DISTANCE_PX,
    max_distance_px: float = LOCAL_MAX_DISTANCE_PX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a geometry-only local mask and the corresponding distances.

    ``pair_indices`` and ``sample_indices`` are the exact output of the
    existing unordered-pair enumerator.  No target tensor is accepted by this
    function, which makes it suitable for a candidate-generation unit test.
    """
    if (atom_geometry.ndim != 3 or atom_geometry.size(-1) < 2
            or atom_mask.ndim != 2 or atom_mask.shape[:2] != atom_geometry.shape[:2]
            or pair_indices.ndim != 2 or pair_indices.shape[-1] != 2
            or sample_indices.shape != (len(pair_indices),)):
        raise ValueError("Malformed predicted atom-pair inputs")
    if (not isinstance(min_distance_px, (int, float))
            or not isinstance(max_distance_px, (int, float))
            or min_distance_px < 0.0 or max_distance_px <= min_distance_px):
        raise ValueError("Invalid local distance bounds")
    if pair_indices.device != atom_geometry.device or sample_indices.device != atom_geometry.device:
        raise ValueError("Pair rows must share the predicted-atom device")
    if len(pair_indices) == 0:
        return (torch.zeros(0, dtype=torch.bool, device=atom_geometry.device),
                torch.zeros(0, dtype=atom_geometry.dtype, device=atom_geometry.device))
    if (sample_indices < 0).any() or (sample_indices >= atom_geometry.size(0)).any():
        raise ValueError("Pair sample index out of range")
    if (pair_indices < 0).any() or (pair_indices >= atom_geometry.size(1)).any():
        raise ValueError("Pair atom index out of range")
    if (pair_indices[:, 0] >= pair_indices[:, 1]).any():
        raise ValueError("Pairs must be sorted unordered pairs")
    valid = atom_mask[sample_indices[:, None], pair_indices].all(dim=1)
    points = atom_geometry[sample_indices[:, None], pair_indices, :2]
    distances = torch.linalg.vector_norm(points[:, 0] - points[:, 1], dim=-1) * IMAGE_SIZE
    mask = valid & torch.isfinite(distances) & (distances >= min_distance_px) & (distances <= max_distance_px)
    return mask, distances


def _top_k_metrics(
    scores: torch.Tensor,
    known: torch.Tensor,
    positive: torch.Tensor,
    chemical: torch.Tensor,
    chemical_targets: torch.Tensor,
    *,
    top_ks: tuple[int, ...] = TOP_KS,
) -> dict[str, dict]:
    """Rank only confirmed rows and report unknown contamination separately."""
    if any(t.ndim != 1 or t.shape != scores.shape for t in (known, positive, chemical, chemical_targets)):
        raise ValueError("Aligned pair labels are required")
    if not bool(((positive & ~known) == 0).all()):
        raise ValueError("Unknown rows cannot be positive")
    known_rows = torch.where(known)[0]
    order = torch.argsort(scores[known_rows], descending=True, stable=True)
    ranked = known_rows[order]
    result = {}
    for value in top_ks:
        if type(value) is not int or value < 1:
            raise ValueError("Top-K values must be positive integers")
        chosen = ranked[: min(value, len(ranked))]
        hits = int(positive[chosen].sum())
        typed_hits = int((positive[chosen] & (chemical[chosen] == chemical_targets[chosen])).sum())
        result[str(value)] = {
            "known_candidates": int(len(ranked)),
            "selected": int(len(chosen)),
            "true_positive": hits,
            "precision": hits / max(len(chosen), 1),
            "positive_total": int(positive.sum()),
            "recall": hits / max(int(positive.sum()), 1),
            "typed_true_positive": typed_hits,
            "typed_precision": typed_hits / max(len(chosen), 1),
            "chemical_type_counts": {
                name: int((positive[chosen] & (chemical[chosen] == index)).sum())
                for index, name in enumerate(BOND_TYPES)
            },
            "chemical_type_correct_counts": {
                name: int(
                    (positive[chosen]
                     & (chemical[chosen] == chemical_targets[chosen])
                     & (chemical_targets[chosen] == index)).sum()
                )
                for index, name in enumerate(BOND_TYPES)
            },
        }
    return result


def _counter_add(counter: Counter, values: dict[str, int]) -> None:
    for key, value in values.items():
        if key == "chemical_type_counts":
            for name, count in value.items():
                counter[f"chemical_type_counts.{name}"] += int(count)
        elif key == "chemical_type_correct_counts":
            for name, count in value.items():
                counter[f"chemical_type_correct_counts.{name}"] += int(count)
        elif key != "precision":
            counter[key] += int(value)


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite local-joint audit output: {output}")
    started = time.monotonic()
    preparation, plan, protocol = _load_preparation()
    sources = source_checks()
    if file_digest(INITIAL) != preparation["initial_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint changed")
    if len(plan["samples"]) != 1200 or plan.get("training_only") is not True:
        raise ValueError("The fixed 1,200-image training-only plan is required")
    rows_by_key = {sample_id(row): row for row in sources["manifest"]["samples"]}
    sample_rows = [rows_by_key[sample_id(row)] for row in plan["samples"]]
    if any(row["split"] != "train" for row in sample_rows):
        raise ValueError("Local candidate audit accepts released train rows only")
    train_rows, holdout_rows, split_meta = split_group_disjoint(sample_rows, 480)
    if len(train_rows) != 480 or len(holdout_rows) != 720:
        raise ValueError(f"Expected deterministic 480/720 split, got {split_meta}")
    if any(split_meta[key] != 0 for key in ("group_overlap", "molecule_overlap", "canonical_smiles_overlap")):
        raise ValueError(f"Identity overlap in split: {split_meta}")
    split_by_key = {sample_id(row): "train" for row in train_rows}
    split_by_key.update({sample_id(row): "holdout" for row in holdout_rows})
    if len(split_by_key) != len(sample_rows):
        raise ValueError("Partition does not cover each fixed sample exactly once")
    loader = RecordLoader(sources["manifest"])
    model, config, payload = load_bundle_for_probe(
        INITIAL, preparation["initial_checkpoint_sha256"]
    )
    if payload["progress"]["optimizer_steps"] != 0:
        raise ValueError("Audit requires the zero-step initial checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fixed local-pair replay audit")
    torch.set_num_threads(2)
    torch.manual_seed(protocol["seed"])
    torch.cuda.manual_seed_all(protocol["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    model.to(device).eval().requires_grad_(False)
    state_before = tensor_tree_digest(model.state_dict())
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="training",
    )
    totals = Counter()
    bond_types = Counter()
    per_sample = []
    top_k_totals = {str(k): Counter() for k in TOP_KS}
    partition_totals = {name: Counter() for name in ("train", "holdout")}
    partition_bond_types = {name: Counter() for name in ("train", "holdout")}
    partition_top_k_totals = {
        name: {str(k): Counter() for k in TOP_KS} for name in ("train", "holdout")
    }
    prefix_failures = []

    with torch.no_grad():
        for attempt, row in enumerate(sample_rows, 1):
            partition = split_by_key[sample_id(row)]
            sample = loader(sources["manifest"], row)
            batch = collator([sample])
            tensors = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            features, _, _, _, _, _, selected = structure_forward(
                model, tensors["pixel_values"], config,
                payload["provenance"]["atom_threshold"],
            )
            selected_digest_before = tensor_tree_digest(selected)
            mapping, _ = match_atom_targets(selected, tensors, config)
            contract = pair_targets_with_unknown(selected, mapping, tensors)
            pairs, sample_indices = contract["pair_indices"], contract["sample_indices"]
            local, distances = local_pair_mask(
                selected["atom_geometry"], selected["atom_mask"], pairs, sample_indices,
            )
            local_rows = torch.where(local)[0]
            if len(local_rows):
                output_pair = model.pair(
                    features,
                    selected["atom_features"],
                    selected["atom_geometry"][..., :2],
                    selected["atom_scores"],
                    pairs[local_rows], sample_indices[local_rows],
                )
                logits = output_pair["presence_logits"].float()
                chemical_logits = output_pair["chemical_logits"].float()
                chemical = chemical_logits.argmax(dim=-1)
            else:
                logits = torch.empty(0, device=device)
                chemical_logits = torch.empty((0, len(BOND_TYPES)), device=device)
                chemical = torch.empty(0, dtype=torch.long, device=device)
            known = contract["known_mask"][local_rows]
            positive = contract["legacy_targets"]["presence"][local_rows]
            chemical_targets = contract["legacy_targets"]["chemical"][local_rows]
            if bool((positive & ~known).any()):
                raise ValueError("Unknown local pair was labeled positive")
            selected_digest_after = tensor_tree_digest(selected)
            if selected_digest_before != selected_digest_after:
                prefix_failures.append(row["sample_key"])
            threshold_selected = (logits.sigmoid() >= PAIR_THRESHOLD)
            strict_threshold_selected = threshold_selected & known
            strict_threshold_positive = strict_threshold_selected & positive
            stats = Counter(
                candidate_pairs=len(pairs),
                local_candidates=len(local_rows),
                known_local_pairs=int(known.sum()),
                unknown_local_pairs=int((~known).sum()),
                confirmed_positive_local_pairs=int((known & positive).sum()),
                confirmed_nonedge_local_pairs=int((known & ~positive).sum()),
                target_bonds=int(tensors["bond_mask"].sum()),
                all_pair_reachable_target_bonds=int(contract["coverage"]["reachable_positive_pairs"]),
                local_reachable_target_bonds=int((known & positive).sum()),
                all_pair_unreachable_target_bonds=int(contract["coverage"]["unreachable_bonds"]),
                target_bonds_not_representable_by_local_pairs=int(tensors["bond_mask"].sum())
                - int((known & positive).sum()),
                baseline_selected_atoms=int(selected["atom_mask"].sum()),
                local_threshold_selected_pairs=int(threshold_selected.sum()),
                strict_threshold_selected_pairs=int(strict_threshold_selected.sum()),
                strict_threshold_true_positive=int(strict_threshold_positive.sum()),
                strict_threshold_false_positive=int(
                    (strict_threshold_selected & ~positive).sum()
                ),
            )
            for index, name in enumerate(BOND_TYPES):
                count = int((known & positive & (chemical_targets == index)).sum())
                bond_types[name] += count
                stats["positive_" + name.lower()] = count
            top_k = _top_k_metrics(logits, known, positive, chemical, chemical_targets)
            for key, values in top_k.items():
                _counter_add(top_k_totals[key], values)
                _counter_add(partition_top_k_totals[partition][key], values)
            totals.update(stats)
            partition_totals[partition].update(stats)
            for index, name in enumerate(BOND_TYPES):
                partition_bond_types[partition][name] += int(
                    (known & positive & (chemical_targets == index)).sum()
                )
            per_sample.append({
                "sample_key": row["sample_key"],
                "split": split_by_key[sample_id(row)],
                "domain": row["domain"],
                "partition": partition,
                **dict(stats),
                "distance_min_px": None if not len(local_rows) else float(distances[local_rows].min()),
                "distance_max_px": None if not len(local_rows) else float(distances[local_rows].max()),
                "top_k": top_k,
                "strict_threshold_precision": int(strict_threshold_positive.sum())
                / max(int(strict_threshold_selected.sum()), 1),
                "strict_threshold_recall": int(strict_threshold_positive.sum())
                / max(int(positive.sum()), 1),
                "selected_prefix_sha256": selected_digest_before,
            })
            if attempt % 100 == 0:
                print(json.dumps({"attempt": attempt, "local_candidates": totals["local_candidates"],
                                  "positive": totals["confirmed_positive_local_pairs"]}), flush=True)

    state_after = tensor_tree_digest(model.state_dict())
    if state_before != state_after:
        raise ValueError("Frozen initial model state changed during forward-only audit")
    if prefix_failures:
        raise ValueError(f"Baseline selected atom prefix changed: {prefix_failures[:3]}")
    if len(per_sample) != 1200:
        raise ValueError("Fixed 1,200-image coverage incomplete")
    # Aggregate top-K precision from sums, while retaining per-image records
    # above.  Unknown rows are intentionally omitted from each denominator.
    top_k_summary = {}
    for key, values in top_k_totals.items():
        top_k_summary[key] = {
            "images": len(per_sample),
            "known_candidates": values["known_candidates"],
            "selected": values["selected"],
            "true_positive": values["true_positive"],
            "positive_total": values["positive_total"],
            "precision": values["true_positive"] / max(values["selected"], 1),
            "recall": values["true_positive"] / max(values["positive_total"], 1),
            "typed_true_positive": values["typed_true_positive"],
            "typed_precision": values["typed_true_positive"] / max(values["selected"], 1),
            "chemical_type_counts": {
                name: values[f"chemical_type_counts.{name}"] for name in BOND_TYPES
            },
            "chemical_type_correct_counts": {
                name: values[f"chemical_type_correct_counts.{name}"] for name in BOND_TYPES
            },
        }
    partition_summary = {}
    for partition in ("train", "holdout"):
        current = partition_totals[partition]
        partition_summary[partition] = {
            "images": len(train_rows) if partition == "train" else len(holdout_rows),
            "totals": dict(sorted(current.items())),
            "confirmed_positive_bond_types": dict(sorted(partition_bond_types[partition].items())),
            "top_k_summary": {
                key: {
                    "images": len(train_rows) if partition == "train" else len(holdout_rows),
                    "known_candidates": values["known_candidates"],
                    "selected": values["selected"],
                    "true_positive": values["true_positive"],
                    "positive_total": values["positive_total"],
                    "precision": values["true_positive"] / max(values["selected"], 1),
                    "recall": values["true_positive"] / max(values["positive_total"], 1),
                    "typed_true_positive": values["typed_true_positive"],
                    "typed_precision": values["typed_true_positive"] / max(values["selected"], 1),
                    "chemical_type_counts": {
                        name: values[f"chemical_type_counts.{name}"] for name in BOND_TYPES
                    },
                    "chemical_type_correct_counts": {
                        name: values[f"chemical_type_correct_counts.{name}"] for name in BOND_TYPES
                    },
                }
                for key, values in partition_top_k_totals[partition].items()
            },
        }
    result = {
        "schema_version": "chemvector-local-joint-candidate-audit-0.1",
        "complete": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "gpu_used": True,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "fixed_training_images": len(per_sample),
        "split": split_meta,
        "split_summary": {
            name: {"images": len(train_rows) if name == "train" else len(holdout_rows),
                   "totals": dict(sorted(values["totals"].items()))}
            for name, values in partition_summary.items()
        },
        "candidate_policy": {
            "source": "predicted_selected_atom_geometry_only",
            "min_distance_px": LOCAL_MIN_DISTANCE_PX,
            "max_distance_px": LOCAL_MAX_DISTANCE_PX,
            "unknown_policy": "exclude_from_positive_and_negative_metrics",
            "strict_positive": "both selected endpoints map one-to-one and target bond_atom_refs identity matches",
            "oracle_candidates_injected": False,
        },
        "totals": dict(sorted(totals.items())),
        "confirmed_positive_bond_types": dict(sorted(bond_types.items())),
        "top_k_summary": top_k_summary,
        "partition_summary": partition_summary,
        "gate": {
            "prefix_immutable": len(prefix_failures) == 0,
            "split_disjoint": all(split_meta[key] == 0 for key in (
                "group_overlap", "molecule_overlap", "canonical_smiles_overlap")),
            "candidate_target_independent": True,
            "reachable_strict_coverage": totals["local_reachable_target_bonds"] / max(
                totals["all_pair_reachable_target_bonds"], 1),
            "reachable_coverage_sufficient": totals["local_reachable_target_bonds"] / max(
                totals["all_pair_reachable_target_bonds"], 1) >= 0.99,
            "train_positive_count": partition_totals["train"]["confirmed_positive_local_pairs"],
            "holdout_positive_count": partition_totals["holdout"]["confirmed_positive_local_pairs"],
            "positive_count_sufficient": (
                partition_totals["train"]["confirmed_positive_local_pairs"] >= 500
                and partition_totals["holdout"]["confirmed_positive_local_pairs"] >= 200
            ),
            "training_admissible": bool(
                len(prefix_failures) == 0
                and all(split_meta[key] == 0 for key in (
                    "group_overlap", "molecule_overlap", "canonical_smiles_overlap"))
                and totals["local_reachable_target_bonds"] / max(
                    totals["all_pair_reachable_target_bonds"], 1) >= 0.99
                and partition_totals["train"]["confirmed_positive_local_pairs"] >= 500
                and partition_totals["holdout"]["confirmed_positive_local_pairs"] >= 200
            ),
        },
        "baseline_prefix": {
            "immutable": True,
            "selected_atom_prefix_failures": len(prefix_failures),
            "pair_threshold": PAIR_THRESHOLD,
            "production_checkpoint_changed": False,
            "state_before_sha256": state_before,
            "state_after_sha256": state_after,
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
