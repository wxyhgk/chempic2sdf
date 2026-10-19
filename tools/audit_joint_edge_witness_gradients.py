"""Read-only gradient-direction audit for the proposed masked joint objective."""

from __future__ import annotations

from pathlib import Path
import json
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL, _load_preparation
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.joint_edge_objective import joint_edge_loss
from chemvector.training.curriculum.pair_scale_plan import sample_id
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.supervision_contract import pair_targets_with_unknown
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/audits/chemvector-joint-edge-witness-gradients-20261005-r2"
TRAIN_WITNESSES = (
    "full_molecule_expansion:curr_9b54d301940543f8__font_large",
    "full_molecule_expansion:curr_75deb03a71d9a1d3__font_small",
    "full_molecule_expansion:curr_e6354f1e02efa8bd__reference",
)
FALLBACK_WITNESSES = (
    "fresh_dev:fresh_dev_001401_coordgen",
    "fresh_dev:fresh_dev_001833_original",
    "fresh_dev:fresh_dev_001866_stress",
)


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output}")
    report, plan, _ = _load_preparation()
    sources = source_checks()
    if file_digest(INITIAL) != report["initial_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint changed")
    manifest = sources["manifest"]
    rows = {f"{row['source']}:{row['sample_key']}": row for row in manifest["samples"]}
    planned = {f"{row['source']}:{row['sample_key']}": row for row in plan["samples"]}
    selected_rows = []
    for key in TRAIN_WITNESSES:
        if key not in rows or key not in planned or rows[key]["split"] != "train":
            raise ValueError(f"Witness is not a fixed released training row: {key}")
        selected_rows.append(rows[key])
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=128, bond_queries=256), purpose="training")
    model, config, payload = load_bundle_for_probe(INITIAL, report["initial_checkpoint_sha256"])
    if payload["progress"]["optimizer_steps"] != 0 or not torch.cuda.is_available():
        raise ValueError("Witness audit requires the zero-step initial checkpoint and CUDA")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    model.to(device).eval().requires_grad_(False)
    records = []
    started = time.monotonic()
    for row in selected_rows:
        sample = loader(manifest, row)
        batch = collator([sample])
        tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            forward = structure_forward(model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"])
            features, _, _, _, _, _, selected = forward
            mapping, _ = match_atom_targets(selected, tensors, config)
            contract = pair_targets_with_unknown(selected, mapping, tensors)
            pairs, sample_indices = contract["pair_indices"], contract["sample_indices"]
            raw = model.pair(
                features, selected["atom_features"], selected["atom_geometry"][..., :2],
                selected["atom_scores"], pairs, sample_indices,
            )
        output_logits = {
            "presence_logits": raw["presence_logits"].detach().requires_grad_(True),
            "chemical_logits": raw["chemical_logits"].detach().requires_grad_(True),
        }
        loss, details = joint_edge_loss(
            output_logits, contract["known_mask"], contract["legacy_targets"]["presence"],
            contract["legacy_targets"]["chemical"], pair_indices=pairs,
            sample_indices=sample_indices, aromatic_safety_weight=1.0,
        )
        if loss is None:
            raise ValueError(f"Witness has no known rows: {sample_id(row)}")
        loss.backward()
        pgrad = output_logits["presence_logits"].grad
        cgrad = output_logits["chemical_logits"].grad
        known = contract["known_mask"]
        positive = contract["legacy_targets"]["presence"].bool()
        negative = known & ~positive
        aromatic = positive & (contract["legacy_targets"]["chemical"] == 3)
        records.append({
            "key": f"{row['source']}:{row['sample_key']}",
            "known_rows": int(known.sum()),
            "unknown_rows": int((~known).sum()),
            "positive_rows": int((known & positive).sum()),
            "confirmed_nonedge_rows": int(negative.sum()),
            "aromatic_positive_rows": int(aromatic.sum()),
            "loss": float(loss.detach()),
            "details": details,
            "unknown_presence_grad_max": float(pgrad[~known].abs().max()) if bool((~known).any()) else 0.0,
            "unknown_chemical_grad_max": float(cgrad[~known].abs().max()) if bool((~known).any()) else 0.0,
            "negative_presence_grad_mean": float(pgrad[negative].mean()) if bool(negative.any()) else None,
            "positive_presence_grad_mean": float(pgrad[known & positive].mean()) if bool((known & positive).any()) else None,
            "aromatic_target_grad_mean": float(cgrad[aromatic, 3].mean()) if bool(aromatic.any()) else None,
            "finite_gradients": bool(torch.isfinite(pgrad).all() and torch.isfinite(cgrad).all()),
        })
    if len(records) != len(TRAIN_WITNESSES):
        raise ValueError("Witness coverage incomplete")
    result = {
        "schema_version": "chemvector-joint-edge-witness-gradients-0.1",
        "complete": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "gpu_used": True,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "objective": "masked_presence_plus_positive_chemical_plus_aromatic_safety",
        "labeled_training_witnesses": list(TRAIN_WITNESSES),
        "fallback_witnesses": list(FALLBACK_WITNESSES),
        "fallback_label_access": "not available; validation rows carry no training graph labels",
        "records": records,
        "all_unknown_gradients_zero": all(
            r["unknown_presence_grad_max"] == 0.0 and r["unknown_chemical_grad_max"] == 0.0
            for r in records
        ),
        "all_finite_gradients": all(r["finite_gradients"] for r in records),
        "source_sha256": {"initial.pt": file_digest(INITIAL)},
        "elapsed_seconds": time.monotonic() - started,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": "completed", "output": str(OUTPUT), "all_unknown_gradients_zero": result["all_unknown_gradients_zero"]}, indent=2))
