#!/usr/bin/env python3
"""Profile-specific CPU A/B for frozen direct AtomPair bond types.

The r10 presence scorer and all AtomPair candidates are frozen.  This audit
trains one tiny type residual head per reviewed rendering profile, using only
positive bond rows from the deterministic 384-image r10 coverage subset.  It
does not update, save, or import a production checkpoint and never uses GPU.
"""
from __future__ import annotations

from collections import Counter, defaultdict
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
    DATA, SHA, DirectPairResidual, _extract, _finish, _score_rows, rows,
)
from tools.train_direct_graph_pair_scorer_cpu_r10_coverage import choose_train
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import one, finish

PRESENCE_STATE = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage/scorer-state.pt"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-type-profile-sidecar-cpu-r10"
TYPE_STATE = OUT / "type-state.pt"
REPORT = OUT / "report.json"
STEPS = 300
THRESHOLD = 0.72  # fixed from r10 train-only calibration; never tuned on holdout


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ProfileTypeResidual(nn.Module):
    """Independent 262->4 zero-initialized type residual per profile."""

    def __init__(self, profiles: list[str], dim: int = 262) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({
            profile: nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, len(BOND_TYPES)),
            )
            for profile in profiles
        })
        for head in self.heads.values():
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, profile: str, features: torch.Tensor) -> torch.Tensor:
        return self.heads[profile](features)


def collect_type_rows(items: list[dict]) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, dict[str, int]]]:
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    labels: dict[str, list[int]] = defaultdict(list)
    counts: dict[str, Counter] = defaultdict(Counter)
    for item in items:
        profile = str(item["row"]["profile_id"])
        pairs, slots, mapping = item["pairs"], item["slots"], item["mapping"]
        target_types = item["target_type_by_edge"]
        for index in torch.where(item["labels"] > 0.5)[0].tolist():
            left, right = (int(v) for v in pairs[index].tolist())
            li = int((slots == left).nonzero(as_tuple=False)[0])
            ri = int((slots == right).nonzero(as_tuple=False)[0])
            if li not in mapping or ri not in mapping:
                continue
            edge = tuple(sorted((mapping[li], mapping[ri])))
            name = target_types.get(edge)
            if name not in BOND_TYPES:
                continue
            grouped[profile].append(item["features"][index])
            labels[profile].append(BOND_TYPES.index(name))
            counts[profile][name] += 1
    if not grouped:
        raise RuntimeError("no positive type rows")
    tensors = {
        profile: (torch.stack(grouped[profile]), torch.tensor(labels[profile], dtype=torch.long))
        for profile in sorted(grouped)
    }
    return tensors, {profile: dict(counts[profile]) for profile in sorted(counts)}


def apply_type(item: dict, presence: DirectPairResidual, typer: ProfileTypeResidual) -> dict:
    # This function intentionally changes only chemical logits.  Candidate
    # pair indices, presence logits, and the fixed r10 presence residual stay
    # byte-for-byte identical between baseline and typed evaluation.
    updated = dict(item)
    profile = str(item["row"]["profile_id"])
    with torch.inference_mode():
        updated["chemical_logits"] = item["chemical_logits"] + typer(profile, item["features"].clone())
    return updated


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed")
    if not PRESENCE_STATE.exists():
        raise RuntimeError("reviewed r10 presence sidecar is missing")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261010)

    all_rows = rows()
    train_rows = choose_train(all_rows)
    holdout_rows = sorted((r for r in all_rows if r.get("split") == "fresh_holdout"), key=lambda r: r["sample_key"])
    if {r["molecule_id"] for r in train_rows} & {r["molecule_id"] for r in holdout_rows}:
        raise RuntimeError("train/holdout molecule identities overlap")
    model, config, _ = load_bundle_for_probe(INITIAL, SHA)
    model.eval().requires_grad_(False)
    payload = torch.load(PRESENCE_STATE, map_location="cpu", weights_only=False)
    presence = DirectPairResidual(input_dim=262)
    presence.load_state_dict(payload["state_dict"])
    presence.eval().requires_grad_(False)

    train = [_extract(row, model, config, need_labels=True) for row in train_rows]
    holdout = [_extract(row, model, config, need_labels=True) for row in holdout_rows]
    grouped, counts = collect_type_rows(train)
    profiles = sorted(grouped)
    typer = ProfileTypeResidual(profiles)
    optimizer = torch.optim.AdamW(typer.parameters(), lr=2e-3, weight_decay=1e-4)
    weights = {}
    for profile, (_, y) in grouped.items():
        tally = torch.bincount(y, minlength=len(BOND_TYPES)).float()
        # Mild inverse-frequency compensation, matching the prior type trial.
        weights[profile] = (tally.sum() / tally.clamp_min(1.0)).sqrt()
    events = []
    for step in range(1, STEPS + 1):
        typer.train()
        optimizer.zero_grad(set_to_none=True)
        losses = [F.cross_entropy(typer(profile, x), y, weight=weights[profile])
                  for profile, (x, y) in grouped.items()]
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite type loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(typer.parameters(), 1.0)
        optimizer.step()
        if step in (1, 100, 200, 300):
            events.append({"step": step, "loss": float(loss.detach())})
    typer.eval().requires_grad_(False)

    baseline_graph = Counter(); typed_graph = Counter()
    baseline_chem = Counter(); typed_chem = Counter()
    profiles_graph: dict[str, dict[str, object]] = {}
    profiles_chem: dict[str, dict[str, object]] = {}
    for item in holdout:
        baseline_graph.update(_score_rows([item], presence, THRESHOLD))
        typed_graph.update(_score_rows([apply_type(item, presence, typer)], presence, THRESHOLD))
        baseline_chem.update(one(item, presence, THRESHOLD))
        typed_chem.update(one(apply_type(item, presence, typer), presence, THRESHOLD))
    for profile in profiles:
        subset = [item for item in holdout if item["row"]["profile_id"] == profile]
        bgraph = Counter(); tgraph = Counter(); bchem = Counter(); tchem = Counter()
        for item in subset:
            bgraph.update(_score_rows([item], presence, THRESHOLD))
            tgraph.update(_score_rows([apply_type(item, presence, typer)], presence, THRESHOLD))
            bchem.update(one(item, presence, THRESHOLD))
            tchem.update(one(apply_type(item, presence, typer), presence, THRESHOLD))
        profiles_graph[profile] = {"baseline": _finish(bgraph), "typed": _finish(tgraph)}
        profiles_chem[profile] = {"baseline": finish(bchem), "typed": finish(tchem)}

    baseline_metrics = _finish(baseline_graph)
    typed_metrics = _finish(typed_graph)
    baseline_sanitize = finish(baseline_chem)
    typed_sanitize = finish(typed_chem)
    after = digest(INITIAL)
    typed_profile_sanitize = {k: v["typed"] for k, v in profiles_chem.items()}
    gate = {
        "checkpoint_unchanged": after == SHA,
        "sealed_test_read": False,
        "presence_candidate_set_unchanged": baseline_metrics["predicted_edges"] == typed_metrics["predicted_edges"] and baseline_metrics["mapped_edges"] == typed_metrics["mapped_edges"],
        "edge_precision_ge_0.99": typed_metrics["edge_precision"] >= 0.99,
        "edge_recall_gt_0.95": typed_metrics["edge_recall"] > 0.95,
        "dangling_le_0.01": typed_metrics["dangling_rate"] <= 0.01,
        "type_accuracy_ge_0.95": typed_metrics["type_accuracy"] >= 0.95,
        "oracle_sanitize_ge_0.90": typed_sanitize["oracle_sanitize_rate"] >= 0.90,
        "all_profile_sanitize_ge_0.90": all(v["oracle_sanitize_rate"] >= 0.90 for v in typed_profile_sanitize.values()),
        "strict_sanitize_improvement": typed_sanitize["oracle_sanitize_rate"] > baseline_sanitize["oracle_sanitize_rate"],
        "production_go": False,
        "gpu_go": False,
    }
    result = {
        "schema_version": "chemvector-identity-aligned-direct-graph-type-profile-sidecar-cpu-r10-0.1",
        "complete": True, "read_only": False, "training_performed": True, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "train_images": len(train_rows), "holdout_images": len(holdout_rows),
        "train_profiles": dict(Counter(r["profile_id"] for r in train_rows)),
        "train_type_rows": int(sum(len(x[1]) for x in grouped.values())),
        "train_type_counts": counts,
        "type_sidecar": {"class": "ProfileTypeResidual", "profiles": profiles,
                          "parameters": sum(p.numel() for p in typer.parameters()),
                          "steps": STEPS, "loss_events": events,
                          "features": 262, "threshold_fixed": THRESHOLD},
        "baseline_holdout": {"graph": baseline_metrics, "chemistry": baseline_sanitize},
        "typed_holdout": {"graph": typed_metrics, "chemistry": typed_sanitize},
        "holdout_profiles": profiles_graph,
        "holdout_chemistry_profiles": profiles_chem,
        "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": after,
        "presence_state_sha256": digest(PRESENCE_STATE),
        "gate": gate,
        "next": "CPU-only diagnostic; no GPU or production integration",
    }
    OUT.mkdir(parents=True)
    torch.save({"schema_version": result["schema_version"], "state_dict": typer.state_dict(),
                "initial_checkpoint_sha256": SHA, "production_checkpoint_written": False}, TYPE_STATE)
    REPORT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(REPORT), "baseline": result["baseline_holdout"],
                      "typed": result["typed_holdout"], "gate": gate}, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()

