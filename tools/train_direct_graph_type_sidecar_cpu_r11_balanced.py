#!/usr/bin/env python3
"""Isolated r11 CPU trial for profile x bond-type balanced type residual.

The r10 type sidecar was trained to classify ``typer(features)`` and then
added to frozen chemical logits.  That objective is not a residual objective:
it can inject a large absolute class prior (notably AROMATIC) into images that
contain no aromatic bonds.  This audit-local trial instead optimizes
``base_chemical_logits + residual`` with deterministic profile/type sample
weights.  The frozen production checkpoint, proposal, presence sidecar, and
sealed test remain untouched.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import (
    DATA, SHA, DirectPairResidual, _extract, rows,
)
from tools.train_direct_graph_pair_scorer_cpu_r10_coverage import choose_train
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import one, finish

PRESENCE_TRIAL = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage"
PRESENCE_STATE = PRESENCE_TRIAL / "scorer-state.pt"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-type-sidecar-cpu-r11-balanced"
REPORT = OUT / "report.json"
TYPE_STATE = OUT / "type-state.pt"
STEPS = 300
THRESHOLD = 0.72
RESIDUAL_SCALE = 0.5


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class TypeResidual(nn.Module):
    def __init__(self, dim: int = 262) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, len(BOND_TYPES))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(x))


def positive_rows(items: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    feats, base, labels, profile_ids = [], [], [], []
    cells = Counter()
    for item in items:
        pairs, slots, mapping = item["pairs"], item["slots"], item["mapping"]
        types = item["target_type_by_edge"]
        profile = item["row"]["profile_id"]
        for i in torch.where(item["labels"] > .5)[0].tolist():
            left, right = (int(v) for v in pairs[i].tolist())
            li = int((slots == left).nonzero(as_tuple=False)[0])
            ri = int((slots == right).nonzero(as_tuple=False)[0])
            if li not in mapping or ri not in mapping:
                continue
            edge = tuple(sorted((mapping[li], mapping[ri])))
            name = types.get(edge)
            if name not in BOND_TYPES:
                continue
            j = BOND_TYPES.index(name)
            feats.append(item["features"][i]); base.append(item["chemical_logits"][i])
            labels.append(j); profile_ids.append(profile); cells[(profile, name)] += 1
    if not feats:
        raise RuntimeError("no positive type rows")
    # Inverse cell-frequency balancing, softened to sqrt so a tiny triple cell
    # cannot overpower the frozen logits.  Mean weight is exactly one.
    mean_count = sum(cells.values()) / max(len(cells), 1)
    raw = torch.tensor([(mean_count / cells[(p, BOND_TYPES[y])]) ** 0.5 for p, y in zip(profile_ids, labels)], dtype=torch.float32)
    raw *= len(raw) / raw.sum().clamp_min(1e-9)
    meta = {"cell_counts": {f"{p}|{t}": int(v) for (p, t), v in sorted(cells.items())}, "weight_min": float(raw.min()), "weight_max": float(raw.max()), "weight_mean": float(raw.mean())}
    return torch.stack(feats), torch.stack(base), torch.tensor(labels, dtype=torch.long), raw, meta


def apply_type(item: dict, presence: DirectPairResidual, typer: TypeResidual) -> dict:
    with torch.inference_mode():
        out = dict(item)
        out["chemical_logits"] = item["chemical_logits"] + RESIDUAL_SCALE * typer(item["features"])
        return out


def run() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed")
    if not PRESENCE_STATE.exists():
        raise RuntimeError("r10 presence sidecar missing")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(20261010)
    state = torch.load(PRESENCE_STATE, map_location="cpu", weights_only=False)
    presence = DirectPairResidual(262); presence.load_state_dict(state["state_dict"]); presence.eval()
    model, config, _ = load_bundle_for_probe(INITIAL, SHA); model.eval().requires_grad_(False)
    all_rows = rows(); train_rows = choose_train(all_rows)
    holdout_rows = sorted((r for r in all_rows if r.get("split") == "fresh_holdout"), key=lambda r: r["sample_key"])
    if {r["molecule_id"] for r in train_rows} & {r["molecule_id"] for r in holdout_rows}:
        raise RuntimeError("train/holdout molecule identity overlap")
    train = [_extract(r, model, config, need_labels=True) for r in train_rows]
    hold = [_extract(r, model, config, need_labels=True) for r in holdout_rows]
    x, base, y, sample_weights, sampling = positive_rows(train)
    typer = TypeResidual(x.size(1)); opt = torch.optim.AdamW(typer.parameters(), lr=2e-3, weight_decay=1e-4)
    events = []
    for step in range(1, STEPS + 1):
        typer.train(); opt.zero_grad(set_to_none=True)
        logits = base + typer(x)
        loss_each = F.cross_entropy(logits, y, reduction="none")
        loss = (loss_each * sample_weights).mean()
        # A small residual penalty discourages a class-prior shift.
        loss = loss + 1e-4 * typer.out.weight.square().mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(typer.parameters(), 1.0); opt.step()
        if step in (1, 100, 200, 300): events.append({"step": step, "loss": float(loss.detach())})
    typer.eval()
    base_total = Counter(); typed_total = Counter(); base_profiles = {}; typed_profiles = {}
    for item in hold:
        base_total.update(one(item, presence, THRESHOLD)); typed_total.update(one(apply_type(item, presence, typer), presence, THRESHOLD))
    for profile in sorted({i["row"]["profile_id"] for i in hold}):
        b = Counter(); t = Counter()
        for item in hold:
            if item["row"]["profile_id"] == profile:
                b.update(one(item, presence, THRESHOLD)); t.update(one(apply_type(item, presence, typer), presence, THRESHOLD))
        base_profiles[profile] = finish(b); typed_profiles[profile] = finish(t)
    after = digest(INITIAL)
    base_f, typed_f = finish(base_total), finish(typed_total)
    result = {
        "schema_version": "chemvector-identity-aligned-direct-graph-type-sidecar-cpu-r11-balanced-0.1",
        "complete": True, "read_only": False, "training_performed": True, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "dataset": str(DATA), "train_images": len(train_rows), "holdout_images": len(hold),
        "train_type_rows": len(y), "sampling": sampling,
        "type_sidecar": {"class": "TypeResidual", "parameters": sum(p.numel() for p in typer.parameters()), "steps": STEPS, "residual_scale": RESIDUAL_SCALE, "loss_events": events},
        "baseline_holdout": base_f, "typed_holdout": typed_f, "baseline_profiles": base_profiles, "typed_profiles": typed_profiles,
        "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": after,
        "gate": {"checkpoint_unchanged": after == SHA, "sealed_test_read": False, "strict_sanitize_improvement": typed_f["oracle_sanitize_rate"] > base_f["oracle_sanitize_rate"], "oracle_sanitize_ge_0.90": typed_f["oracle_sanitize_rate"] >= .90, "all_profile_sanitize_ge_0.85": all(v["oracle_sanitize_rate"] >= .85 for v in typed_profiles.values()), "production_go": False, "next": "diagnostic CPU result; no production merge"},
    }
    OUT.mkdir(parents=True); torch.save({"schema_version": result["schema_version"], "state_dict": typer.state_dict(), "initial_checkpoint_sha256": SHA}, TYPE_STATE); REPORT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(REPORT), "baseline": base_f, "typed": typed_f, "typed_profiles": typed_profiles, "gate": result["gate"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
