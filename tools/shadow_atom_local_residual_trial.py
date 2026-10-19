"""In-memory zero-start local heatmap residual shadow.

The residual is initialized to exactly zero, so step zero is byte-equivalent to
the frozen production atom head.  Only the new residual parameters are updated;
no checkpoint is written and no sealed/validation data is read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch import nn

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.components.atoms.objects.spatial_atom_object import atom_local_peak_margin_loss
from chemvector.components.graph.joint.predicted_incidence import (
    PredictedProposalConfig,
    decode_predicted_atom_proposals,
)
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets
from chemvector.vertex import build_vertex_heatmap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_ATOM_LOCAL_RESIDUAL_OUTPUT",
    str(ROOT / "outputs/audits/chemvector-atom-local-residual-shadow-20261006"),
))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
STEPS = int(os.environ.get("CHEMVECTOR_ATOM_LOCAL_RESIDUAL_STEPS", "40"))
STYLE_COUNTS = {"font_large": 4, "serif": 4, "line_thick": 4}


class ZeroStartLocalResidual(nn.Module):
    """Small local adapter; the final projection starts at exact zero."""

    def __init__(self, channels: int = 96, hidden: int = 32) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(channels, hidden, 1)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.output_projection = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.input_projection(features))
        hidden = torch.nn.functional.gelu(self.local(hidden))
        return self.output_projection(hidden)


def _coverage(model, residual, config, payload, loader, manifest, row, collator, device):
    sample = loader(manifest, row)
    batch = collator([sample])
    tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        features = model.vision.forward_stroke_resolution(tensors["pixel_values"])
        output = model.atom(features)
        output["heatmap_logits"] = output["heatmap_logits"] + residual(features)
        proposals = decode_predicted_atom_proposals(output, config)
    mapping, _ = match_atom_targets(proposals, tensors, config)
    target_count = int(tensors["atom_mask"].sum())
    heat = output["heatmap_logits"].sigmoid()[0, 0]
    peaks = heat == torch.nn.functional.max_pool2d(heat[None, None], 3, 1, 1)[0, 0]
    anchors = tensors["atom_anchors"][0]
    height, width = heat.shape
    local_peak_count = 0
    for atom in range(target_count):
        cx = min(width - 1, int(float(anchors[atom, 0]) * width))
        cy = min(height - 1, int(float(anchors[atom, 1]) * height))
        local_peak_count += int(bool(peaks[cy, cx]))
    return {
        "sample_key": row["sample_key"],
        "profile_id": sample["meta"]["profile_id"],
        "targets": target_count,
        "raw_candidates": int(proposals["atom_mask"].sum()),
        "raw_matched": int((mapping >= 0).sum()),
        "raw_target_ids": sorted(int(x) for x in mapping[0].tolist() if int(x) >= 0),
        "target_local_peaks": local_peak_count,
    }


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available() or file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("CUDA or initial checkpoint contract unavailable")
    if not 1 <= STEPS <= 200:
        raise ValueError("steps must be in [1, 200]")
    torch.manual_seed(20261006)
    torch.cuda.manual_seed_all(20261006)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    sources = source_checks()
    fixed = json.loads((ROOT / "outputs/audits/chemvector-edge-label-reachability-20261005/report.json").read_text())
    fixed_keys = {row["sample_key"] for row in fixed["per_sample"]}
    rows = [row for row in sources["manifest"]["samples"] if row["sample_key"] in fixed_keys]
    if len(rows) != 1200:
        raise ValueError(f"Expected fixed 1,200 rows, found {len(rows)}")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    residual = ZeroStartLocalResidual().to(device).train()
    if sum(parameter.numel() for parameter in residual.parameters()) != 3457:
        raise ValueError("Unexpected residual parameter count")
    loader = RecordLoader(sources["manifest"])
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds),
        purpose="training",
    )
    config = PredictedProposalConfig(
        maximum_atoms=128,
        maximum_bonds=config.maximum_bonds,
        atom_threshold=0.01,
        bond_threshold=0.01,
        atom_match_radius_px=6.0,
        bond_match_radius_px=12.0,
        image_size=768.0,
    )
    all_baseline = [
        (row, _coverage(model, residual, config, payload, loader, sources["manifest"], row, collator, device))
        for row in rows
    ]
    chosen = []
    for style, count in STYLE_COUNTS.items():
        candidates = [(row, cov) for row, cov in all_baseline if cov["profile_id"] == style and cov["raw_matched"] < cov["targets"]]
        if len(candidates) < count:
            raise ValueError(f"Insufficient raw-gap rows for {style}")
        chosen.extend(row for row, _ in candidates[:count])
    baseline = {row["sample_key"]: cov for row, cov in all_baseline if row["sample_key"] in {x["sample_key"] for x in chosen}}
    optimizer = torch.optim.AdamW(residual.parameters(), lr=2e-4, weight_decay=0.0)
    events = []
    for step in range(1, STEPS + 1):
        row = chosen[(step - 1) % len(chosen)]
        sample = loader(sources["manifest"], row)
        batch = collator([sample])
        tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            features = model.vision.forward_stroke_resolution(tensors["pixel_values"])
            base = model.atom(features)
        logits = base["heatmap_logits"] + residual(features)
        positive = build_vertex_heatmap(
            tensors["atom_anchors"], tensors["atom_mask"], logits.shape[-2], logits.shape[-1], radius_cells=0,
        ).bool()
        loss = atom_local_peak_margin_loss(logits, positive, margin=0.75, kernel_size=3)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite residual loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(residual.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        events.append({"step": step, "sample_key": row["sample_key"], "loss": float(loss.detach())})
    after = [_coverage(model, residual, config, payload, loader, sources["manifest"], row, collator, device) for row in chosen]
    before_rows = [baseline[row["sample_key"]] for row in chosen]
    broken = []
    recovered = []
    for old, new in zip(before_rows, after):
        lost = sorted(set(old["raw_target_ids"]) - set(new["raw_target_ids"]))
        gained = sorted(set(new["raw_target_ids"]) - set(old["raw_target_ids"]))
        if lost:
            broken.append({"sample_key": old["sample_key"], "lost_raw_target_ids": lost})
        if gained:
            recovered.append({"sample_key": old["sample_key"], "gained_raw_target_ids": gained})
    result = {
        "schema_version": "chemvector-atom-local-residual-shadow-0.1",
        "complete": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "virtual_steps": STEPS,
        "persisted_parameter_updates": 0,
        "gpu_used": True,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "initial_sha256": INITIAL_SHA,
        "trainable_parameter_count": sum(p.numel() for p in residual.parameters()),
        "slice": STYLE_COUNTS,
        "selection": "first raw-gap row per style, deterministic",
        "baseline": before_rows,
        "after": after,
        "events": events,
        "broken_raw_target_ids": broken,
        "recovered_raw_target_ids": recovered,
        "gate": {
            "no_baseline_raw_loss": not broken,
            "raw_recovery_count": len(recovered),
            "target_local_peak_non_decrease": all(n["target_local_peaks"] >= o["target_local_peaks"] for o, n in zip(before_rows, after)),
            "pass": not broken and bool(recovered),
        },
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
