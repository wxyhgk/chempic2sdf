"""In-memory offset-only shadow on a tiny admitted training slice.

This is deliberately not a training launcher.  It performs virtual optimizer
steps on a copy of the initial model and persists only a read-only audit
report.  Heatmap, selector, vision, pair, role, bbox, and OCR parameters are
frozen; no checkpoint is written and no validation/sealed data is read.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_ATOM_OFFSET_SHADOW_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-atom-offset-shadow-20261006"),
))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
VIRTUAL_STEPS = int(os.environ.get("CHEMVECTOR_ATOM_OFFSET_STEPS", "10"))
if VIRTUAL_STEPS < 1 or VIRTUAL_STEPS > 200:
    raise ValueError("Offset shadow/trial steps must be in [1, 200]")
STYLE_COUNTS = {"font_large": 4, "serif": 4, "line_thick": 4}


def _offset_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    anchors = batch["atom_anchors"]
    mask = batch["atom_mask"]
    _, _, height, width = output["anchor_offset_logits"].shape
    cell_x = (anchors[..., 0] * width).clamp(0, width - 1e-4)
    cell_y = (anchors[..., 1] * height).clamp(0, height - 1e-4)
    index_x, index_y = cell_x.long(), cell_y.long()
    batch_index = torch.arange(anchors.size(0), device=anchors.device)[:, None]
    predicted = output["anchor_offset_logits"].sigmoid().permute(0, 2, 3, 1)[
        batch_index, index_y, index_x
    ]
    expected = torch.stack((cell_x - index_x, cell_y - index_y), dim=-1)
    if not bool(mask.any()):
        raise ValueError("offset shadow received an empty target image")
    return F.smooth_l1_loss(predicted[mask], expected[mask])


@torch.no_grad()
def _coverage(model, config, payload, loader, manifest, row, collator, device):
    sample = loader(manifest, row)
    batch = collator([sample])
    tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    _, _, _, raw, _, _, selected = structure_forward(
        model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"]
    )
    raw_mapping, _ = match_atom_targets(raw, tensors, config)
    selected_mapping, _ = match_atom_targets(selected, tensors, config)
    return {
        "sample_key": row["sample_key"],
        "profile_id": sample["meta"]["profile_id"],
        "targets": int(tensors["atom_mask"].sum()),
        "raw_candidates": int(raw["atom_mask"].sum()),
        "raw_matched": int((raw_mapping >= 0).sum()),
        "selected_candidates": int(selected["atom_mask"].sum()),
        "selected_matched": int((selected_mapping >= 0).sum()),
        "raw_target_ids": sorted(int(x) for x in raw_mapping[0].tolist() if int(x) >= 0),
        "selected_target_ids": sorted(int(x) for x in selected_mapping[0].tolist() if int(x) >= 0),
    }


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the bounded shadow")
    if file_digest(INITIAL) != INITIAL_SHA:
        raise ValueError("Initial checkpoint changed")
    sources = source_checks()
    fixed = json.loads((ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json").read_text())
    fixed_keys = {row["sample_key"] for row in fixed["per_sample"]}
    rows = [row for row in sources["manifest"]["samples"] if row["sample_key"] in fixed_keys]
    if len(rows) != 1200:
        raise ValueError(f"Expected fixed 1,200 training rows, found {len(rows)}")
    torch.manual_seed(20261006)
    torch.cuda.manual_seed_all(20261006)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = tuple(model.atom.anchor_offset.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    if sum(parameter.numel() for parameter in trainable) != 258:
        raise ValueError("Unexpected anchor_offset parameter count")
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    loader = RecordLoader(sources["manifest"])
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="training",
    )
    # First perform a read-only coverage pass over the fixed cohort, then select
    # actual raw-gap images.  A style-only slice can otherwise contain no gap
    # and make a short shadow appear uninformative.
    all_baseline = [
        (row, _coverage(model, config, payload, loader, sources["manifest"], row, collator, device))
        for row in rows
    ]
    chosen = []
    for style, count in STYLE_COUNTS.items():
        candidates = [
            (row, coverage) for row, coverage in all_baseline
            if coverage["profile_id"] == style and coverage["raw_matched"] < coverage["targets"]
        ]
        if len(candidates) < count:
            candidates.extend(
                (row, coverage) for row, coverage in all_baseline
                if coverage["profile_id"] == style and (row, coverage) not in candidates
            )
        if len(candidates) < count:
            raise ValueError(f"Insufficient {style} rows")
        chosen.extend(row for row, _ in candidates[:count])
    if len(chosen) != sum(STYLE_COUNTS.values()) or len({row["sample_key"] for row in chosen}) != len(chosen):
        raise ValueError("Shadow slice size/identity mismatch")
    baseline_by_key = {row["sample_key"]: coverage for row, coverage in all_baseline}
    baseline = [baseline_by_key[row["sample_key"]] for row in chosen]
    optimizer = torch.optim.SGD(trainable, lr=1e-3, momentum=0.0, weight_decay=0.0)
    events = []
    for step in range(1, VIRTUAL_STEPS + 1):
        row = chosen[(step - 1) % len(chosen)]
        sample = loader(sources["manifest"], row)
        batch = collator([sample])
        pixels = batch["pixel_values"].to(device)
        tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            features = model.vision.forward_stroke_resolution(pixels)
        output = model.atom(features)
        loss = _offset_loss(output, tensors)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite offset loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in trainable):
            raise ValueError("Missing/nonfinite offset gradient")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "sample_key": row["sample_key"], "loss": float(loss.detach()), "gradient_norm": float(norm)})

    after = [_coverage(model, config, payload, loader, sources["manifest"], row, collator, device) for row in chosen]
    broken = []
    recovered = []
    for old, new in zip(baseline, after):
        lost = sorted(set(old["raw_target_ids"]) - set(new["raw_target_ids"]))
        gained = sorted(set(new["raw_target_ids"]) - set(old["raw_target_ids"]))
        if lost:
            broken.append({"sample_key": old["sample_key"], "lost_raw_target_ids": lost})
        if gained:
            recovered.append({"sample_key": old["sample_key"], "gained_raw_target_ids": gained})
    changed_frozen = []
    for name, value in before.items():
        if name.startswith("atom.anchor_offset."):
            continue
        if not torch.equal(value, model.state_dict()[name].detach().cpu()):
            changed_frozen.append(name)
    if changed_frozen:
        raise ValueError(f"Frozen parameters changed: {changed_frozen[:5]}")
    result = {
        "schema_version": "chemvector-atom-offset-shadow-0.1",
        "complete": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "virtual_steps": VIRTUAL_STEPS,
        "persisted_parameter_updates": 0,
        "in_memory_virtual_parameter_updates": VIRTUAL_STEPS,
        "gpu_used": True,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "trainable_parameters": ["atom.anchor_offset.weight", "atom.anchor_offset.bias"],
        "trainable_parameter_count": 258,
        "slice": STYLE_COUNTS,
        "selection": "first actual raw-gap rows per style, then deterministic style fill",
        "baseline": baseline,
        "after": after,
        "broken_raw_target_ids": broken,
        "recovered_raw_target_ids": recovered,
        "events": events,
        "gate": {"no_baseline_raw_loss": not broken, "raw_recovery_count": len(recovered), "pass": not broken and bool(recovered)},
        "initial_sha256": INITIAL_SHA,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
