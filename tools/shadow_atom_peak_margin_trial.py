"""In-memory local-peak margin shadow for the atom heatmap."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.components.atoms.objects.spatial_atom_object import atom_local_peak_margin_loss
from chemvector.components.atoms.objects.spatial_atom_object import spatial_atom_object_loss
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.vertex import build_vertex_heatmap, focal_heatmap_loss
from tools.shadow_atom_offset_trial import _coverage


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_ATOM_PEAK_MARGIN_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-atom-peak-margin-shadow-20261006"),
))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
VIRTUAL_STEPS = int(os.environ.get("CHEMVECTOR_ATOM_PEAK_MARGIN_STEPS", "10"))
MARGIN = 0.75
MARGIN_WEIGHT = 0.5
STYLE_COUNTS = {"font_large": 4, "serif": 4, "line_thick": 4}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    sources = source_checks()
    manifest = sources["manifest"]
    fixed = json.loads((ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json").read_text())
    fixed_keys = {row["sample_key"] for row in fixed["per_sample"]}
    rows = [row for row in manifest["samples"] if row["sample_key"] in fixed_keys]
    if len(rows) != 1200:
        raise ValueError("Fixed 1,200-row cohort changed")
    loader = RecordLoader(manifest)
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = tuple(model.atom.heatmap.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    if sum(parameter.numel() for parameter in trainable) != 129:
        raise ValueError("Unexpected heatmap parameter count")
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="training",
    )
    all_baseline = [
        (row, _coverage(model, config, payload, loader, manifest, row, collator, "cuda:0"))
        for row in rows
    ]
    chosen = []
    for style, count in STYLE_COUNTS.items():
        candidates = [(row, cov) for row, cov in all_baseline
                      if cov["profile_id"] == style and cov["raw_matched"] < cov["targets"]]
        if len(candidates) < count:
            candidates.extend((row, cov) for row, cov in all_baseline
                              if cov["profile_id"] == style and row not in [x[0] for x in candidates])
        if len(candidates) < count:
            raise ValueError(f"Insufficient {style} rows")
        chosen.extend(row for row, _ in candidates[:count])
    baseline_by_key = {row["sample_key"]: cov for row, cov in all_baseline}
    baseline = [baseline_by_key[row["sample_key"]] for row in chosen]
    optimizer = torch.optim.SGD(trainable, lr=1e-3, momentum=0.0, weight_decay=0.0)
    events = []
    for step in range(1, VIRTUAL_STEPS + 1):
        row = chosen[(step - 1) % len(chosen)]
        sample = loader(manifest, row)
        batch = collator([sample])
        tensors = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            features = model.vision.forward_stroke_resolution(tensors["pixel_values"])
        output = model.atom(features)
        positive = build_vertex_heatmap(
            tensors["atom_anchors"], tensors["atom_mask"],
            output["heatmap_logits"].shape[-2], output["heatmap_logits"].shape[-1], radius_cells=0,
        ).bool()
        focal = focal_heatmap_loss(output["heatmap_logits"], positive.float())
        margin = atom_local_peak_margin_loss(output["heatmap_logits"], positive, margin=MARGIN, kernel_size=3)
        loss = focal + MARGIN_WEIGHT * margin
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite peak-margin loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in trainable):
            raise ValueError("Missing/nonfinite heatmap gradient")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "sample_key": row["sample_key"], "loss": float(loss.detach()),
                       "focal": float(focal.detach()), "margin": float(margin.detach()), "gradient_norm": float(norm)})
    after = [_coverage(model, config, payload, loader, manifest, row, collator, "cuda:0") for row in chosen]
    broken, recovered = [], []
    for old, new in zip(baseline, after):
        lost = sorted(set(old["raw_target_ids"]) - set(new["raw_target_ids"]))
        gained = sorted(set(new["raw_target_ids"]) - set(old["raw_target_ids"]))
        if lost:
            broken.append({"sample_key": old["sample_key"], "lost_raw_target_ids": lost})
        if gained:
            recovered.append({"sample_key": old["sample_key"], "gained_raw_target_ids": gained})
    result = {
        "schema_version": "chemvector-atom-peak-margin-shadow-0.1",
        "complete": True, "training_performed": False, "optimizer_steps": 0,
        "virtual_steps": VIRTUAL_STEPS, "persisted_parameter_updates": 0,
        "gpu_used": True, "sealed_test_read": False, "deployment_authorized": False,
        "trainable_parameters": ["atom.heatmap.weight", "atom.heatmap.bias"],
        "trainable_parameter_count": 129, "margin": MARGIN, "margin_weight": MARGIN_WEIGHT,
        "baseline": baseline, "after": after, "events": events,
        "broken_raw_target_ids": broken, "recovered_raw_target_ids": recovered,
        "gate": {"no_baseline_raw_loss": not broken, "raw_recovery_count": len(recovered),
                 "pass": not broken and bool(recovered)}, "initial_sha256": INITIAL_SHA,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
