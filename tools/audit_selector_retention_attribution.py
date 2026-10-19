"""Read-only per-target attribution for raw-matched but selector-dropped atoms."""

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
    "CHEMVECTOR_SELECTOR_ATTRIBUTION_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-selector-retention-attribution-20261006-r3"),
))


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    fixed = json.loads(COHORT.read_text(encoding="utf-8"))
    keys = {row["sample_key"] for row in fixed["per_sample"]}
    sources = source_checks()
    manifest = sources["manifest"]
    rows = sorted((row for row in manifest["samples"] if row["sample_key"] in keys),
                  key=lambda row: row["sample_key"])
    if len(rows) != 1200:
        raise ValueError(f"Expected 1,200 rows, found {len(rows)}")
    train, holdout, split = split_group_disjoint(rows, 480)
    split_by_key = {row["sample_key"]: "train" for row in train}
    split_by_key.update({row["sample_key"]: "holdout" for row in holdout})
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval().requires_grad_(False)
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="training",
    )
    dropped = []
    totals = Counter()
    by_split = {name: Counter() for name in ("train", "holdout")}
    by_style = Counter()
    by_role = Counter()
    by_element = Counter()
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            sample = loader(manifest, row)
            batch = collator([sample])
            tensors = {key: value.cuda() for key, value in batch.items() if isinstance(value, torch.Tensor)}
            _, _, _, raw, _, set_output, selected = structure_forward(
                model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
            )
            raw_mapping, _ = match_atom_targets(raw, tensors, config)
            selected_mapping, _ = match_atom_targets(selected, tensors, config)
            raw_values = raw_mapping[0].detach().cpu().tolist()
            selected_values = selected_mapping[0].detach().cpu().tolist()
            raw_by_target = {int(target): slot for slot, target in enumerate(raw_values) if int(target) >= 0}
            selected_targets = {int(target) for target in selected_values if int(target) >= 0}
            selected_slot_targets = {slot: int(target) for slot, target in enumerate(selected_values) if int(target) >= 0}
            atoms = {int(atom["atom_id"]): atom for atom in sample["layers"]["atoms"]}
            partition = split_by_key[row["sample_key"]]
            totals["images"] += 1
            totals["raw_matched_targets"] += len(raw_by_target)
            totals["selected_matched_targets"] += len(selected_targets)
            for target, slot in sorted(raw_by_target.items()):
                if target in selected_targets:
                    continue
                atom = atoms[target]
                element = str(atom["element"])
                role = "text" if bool(atom["visible_text"]) else "implicit"
                point = raw["atom_geometry"][0, slot, :2].detach().cpu() * 768.0
                anchor = tensors["atom_anchors"][0, target, :2].detach().cpu() * 768.0
                distance = float(torch.linalg.vector_norm(point - anchor))
                score = float(raw["atom_scores"][0, slot].detach().cpu())
                selected_score = float(selected["atom_scores"][0, slot].detach().cpu())
                record = {
                    "sample_key": row["sample_key"],
                    "split": partition,
                    "style": str(sample["meta"]["profile_id"]),
                    "group_key": row["group_key"],
                    "molecule_id": row["molecule_id"],
                    "canonical_smiles": row["canonical_smiles"],
                    "target_id": target,
                    "raw_slot": slot,
                    "element": element,
                    "role": role,
                    "visible_text": bool(atom["visible_text"]),
                    "label_text": atom.get("label_text"),
                    "raw_score": score,
                    "selected_score": selected_score,
                    "score_delta": selected_score - score,
                    "selector_presence_logit": float(set_output["atom_presence_logits"][0, slot].detach().cpu()),
                    "selector_pre_presence_logit": float(set_output["atom_pre_presence_logits"][0, slot].detach().cpu()),
                    "selector_threshold": float(payload["provenance"]["atom_threshold"]),
                    "selector_margin": float(set_output["atom_presence_logits"][0, slot].detach().cpu()) - float(payload["provenance"]["atom_threshold"]),
                    "raw_slot_active": bool(raw["atom_mask"][0, slot].detach().cpu()),
                    "selected_slot_active": bool(selected["atom_mask"][0, slot].detach().cpu()),
                    "selected_slot_target": selected_slot_targets.get(slot),
                    "raw_target_distance_px": distance,
                    "raw_point_px": point.tolist(),
                    "target_anchor_px": anchor.tolist(),
                }
                dropped.append(record)
                totals["selector_drops"] += 1
                by_split[partition]["selector_drops"] += 1
                by_split[partition][role] += 1
                by_style[(partition, record["style"])] += 1
                by_role[(partition, role)] += 1
                by_element[(partition, element)] += 1
            if index % 200 == 0:
                print(json.dumps({"images": index, "selector_drops": len(dropped)}), flush=True)
    result = {
        "schema_version": "chemvector-selector-retention-attribution-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "optimizer_steps": 0, "gpu_used": True, "sealed_test_read": False,
        "deployment_authorized": False, "initial_sha256": INITIAL_SHA,
        "cohort_report": str(COHORT.relative_to(ROOT)), "split": split,
        "totals": dict(totals),
        "by_split": {k: dict(v) for k, v in by_split.items()},
        "by_style": {f"{p}:{s}": n for (p, s), n in sorted(by_style.items())},
        "by_role": {f"{p}:{s}": n for (p, s), n in sorted(by_role.items())},
        "by_element": {f"{p}:{s}": n for (p, s), n in sorted(by_element.items())},
        "records": dropped,
        "gate": {
            "identity_disjoint": all(split[k] == 0 for k in ("group_overlap", "molecule_overlap", "canonical_smiles_overlap")),
            "training_admissible": False,
            "reason": "attribution_only; no selector trial authorized",
        },
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "totals": result["totals"], "by_split": result["by_split"],
                      "by_role": result["by_role"], "by_element": result["by_element"]}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
