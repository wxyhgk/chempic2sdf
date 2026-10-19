"""Read-only bond-support attribution for selector-dropped implicit carbons."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets
from tools.current_initial_cohort import split_group_disjoint


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_BOND_SUPPORTED_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-bond-supported-selector-drops-20261006-r1"),
))
ENDPOINT_RADIUS_PX = 12.0


def _bond_match(pred_endpoints, target_endpoints, radius):
    direct = torch.linalg.vector_norm(pred_endpoints[0] - target_endpoints[0]) <= radius and torch.linalg.vector_norm(pred_endpoints[1] - target_endpoints[1]) <= radius
    reverse = torch.linalg.vector_norm(pred_endpoints[0] - target_endpoints[1]) <= radius and torch.linalg.vector_norm(pred_endpoints[1] - target_endpoints[0]) <= radius
    return bool(direct or reverse)


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    fixed = json.loads(COHORT.read_text(encoding="utf-8"))
    keys = {row["sample_key"] for row in fixed["per_sample"]}
    sources = source_checks(); manifest = sources["manifest"]
    rows = sorted((row for row in manifest["samples"] if row["sample_key"] in keys), key=lambda r: r["sample_key"])
    train_rows, holdout_rows, split = split_group_disjoint(rows, 480)
    split_by_key = {row["sample_key"]: "train" for row in train_rows}
    split_by_key.update({row["sample_key"]: "holdout" for row in holdout_rows})
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval().requires_grad_(False)
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds), purpose="training")
    records = []
    by_partition = {"train": Counter(), "holdout": Counter()}
    by_style = Counter()
    by_type = Counter()
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            sample = loader(manifest, row); batch = collator([sample])
            tensors = {key: value.cuda() for key, value in batch.items() if isinstance(value, torch.Tensor)}
            _, _, _, raw, bonds, set_output, selected = structure_forward(model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"])
            raw_map, _ = match_atom_targets(raw, tensors, config)
            selected_map, _ = match_atom_targets(selected, tensors, config)
            raw_values = raw_map[0].detach().cpu().tolist(); selected_values = selected_map[0].detach().cpu().tolist()
            raw_by_target = {int(target): slot for slot, target in enumerate(raw_values) if int(target) >= 0}
            selected_targets = {int(target) for target in selected_values if int(target) >= 0}
            raw_mask = raw["atom_mask"][0].detach().cpu().bool(); selected_mask = selected["atom_mask"][0].detach().cpu().bool()
            target_count = int(tensors["atom_mask"].sum())
            anchors = tensors["atom_anchors"][0, :target_count, :2].detach().cpu() * 768.0
            pred_endpoints = bonds["bond_endpoints"][0].detach().cpu() * 768.0
            pred_scores = bonds["bond_scores"][0].detach().cpu()
            pred_types = bonds["bond_types"][0].detach().cpu().argmax(-1)
            pred_mask = bonds["bond_mask"][0].detach().cpu().bool()
            gt_refs = tensors["bond_atom_refs"][0].detach().cpu()
            gt_ports = tensors["bond_ports"][0].detach().cpu() * 768.0
            gt_types = tensors["bond_chemical"][0].detach().cpu()
            partition = split_by_key[row["sample_key"]]
            atoms = {int(atom["atom_id"]): atom for atom in sample["layers"]["atoms"]}
            for target, slot in sorted(raw_by_target.items()):
                if target in selected_targets or not bool(raw_mask[slot]) or bool(selected_mask[slot]):
                    continue
                atom = atoms[target]
                if bool(atom["visible_text"]):
                    continue
                point = raw["atom_geometry"][0, slot, :2].detach().cpu() * 768.0
                endpoint_distances = torch.linalg.vector_norm(pred_endpoints - point[None, None, :], dim=-1)
                eligible = pred_mask[:, None].expand_as(endpoint_distances)
                if bool(eligible.any()):
                    flat = endpoint_distances.masked_fill(~eligible, float("inf")).reshape(-1)
                    best_flat = int(torch.argmin(flat)); best_bond = best_flat // 2
                    best_distance = float(flat[best_flat]); support_count = int((endpoint_distances[eligible] <= ENDPOINT_RADIUS_PX).sum())
                    best_score = float(pred_scores[best_bond]); best_type = int(pred_types[best_bond])
                else:
                    best_bond, best_distance, support_count, best_score, best_type = -1, float("inf"), 0, 0.0, -1
                incident = [int(i) for i, refs in enumerate(gt_refs.tolist()) if target in refs]
                strict = []
                for gt_index in incident:
                    for pred_index in torch.where(pred_mask)[0].tolist():
                        if _bond_match(pred_endpoints[pred_index], gt_ports[gt_index], ENDPOINT_RADIUS_PX) and int(pred_types[pred_index]) == int(gt_types[gt_index]):
                            strict.append((pred_index, gt_index, float(pred_scores[pred_index])))
                record = {
                    "sample_key": row["sample_key"], "split": partition, "style": str(sample["meta"]["profile_id"]),
                    "target_id": target, "element": str(atom["element"]), "role": "implicit", "raw_slot": slot,
                    "best_endpoint_distance_px": best_distance, "support_count_within12px": support_count,
                    "best_supported_bond_index": best_bond, "best_supported_bond_score": best_score,
                    "best_supported_bond_type": best_type, "incident_gt_bond_count": len(incident),
                    "strict_identity_supported": bool(strict),
                    "strict_identity_best_score": max((v[2] for v in strict), default=0.0),
                    "strict_identity_pairs": [{"predicted_bond": a, "gt_bond": b, "score": c} for a, b, c in strict],
                }
                records.append(record)
                stats = by_partition[partition]; stats["drops"] += 1; stats["endpoint_supported"] += int(support_count > 0); stats["strict_identity_supported"] += int(bool(strict))
                by_style[(partition, record["style"])] += 1
                for gt_index in incident: by_type[(partition, int(gt_types[gt_index]))] += 1
            if index % 200 == 0: print(json.dumps({"images": index, "implicit_drops": len(records)}), flush=True)
    result = {
        "schema_version": "chemvector-bond-supported-selector-drops-0.1", "complete": True,
        "read_only": True, "training_performed": False, "optimizer_steps": 0,
        "gpu_used": True, "sealed_test_read": False, "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA, "endpoint_radius_px": ENDPOINT_RADIUS_PX,
        "split": split, "by_partition": {k: dict(v) for k, v in by_partition.items()},
        "by_style": {f"{p}:{s}": n for (p, s), n in sorted(by_style.items())},
        "incident_bond_type_counts": {f"{p}:{t}": n for (p, t), n in sorted(by_type.items())},
        "records": records,
        "gate": {
            "identity_disjoint": all(split[k] == 0 for k in ("group_overlap", "molecule_overlap", "canonical_smiles_overlap")),
            "train_strict_positive_min50": by_partition["train"]["strict_identity_supported"] >= 50,
            "holdout_strict_positive_min14": by_partition["holdout"]["strict_identity_supported"] >= 14,
            "training_admissible": False,
            "reason": "read-only attribution; selector trial requires a separate preflight",
        },
    }
    OUTPUT.mkdir(parents=True); (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "by_partition": result["by_partition"], "gate": result["gate"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
