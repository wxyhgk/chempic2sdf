#!/usr/bin/env python3
"""CPU shadow: learn a type residual on the t3f frozen-pair route.

Presence selection remains the already-trained frozen-pair residual and the
candidate edge set stays unchanged.  A small four-class residual is fitted
only on mapped true train edges, then evaluated on the disjoint holdout.  This
is an attribution probe, never a production checkpoint update.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.train_r17_pair_incidence_known_clean_cpu as base
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA
from chemvector.bond_object import BOND_TYPES

COHORT = ROOT / os.environ.get("R17_T3F_TYPE_COHORT", "outputs/datasets/chemvector-r17-residual-density-source-20261020-t3f/manifest.json")
SOURCE = ROOT / os.environ.get("R17_T3F_TYPE_SOURCE", "outputs/datasets/chemvector-r17-residual-density-source-20261020-t3f/samples")
PRESENCE = ROOT / os.environ.get("R17_T3F_TYPE_PRESENCE", "outputs/audits/chemvector-r17-residual-density-pair-incidence-cpu-20261020-t3f/sidecar-state.pt")
OUT = ROOT / os.environ.get("R17_T3F_TYPE_OUT", "outputs/audits/chemvector-r17-t3f-type-correction-cpu")
STEPS = int(os.environ.get("R17_T3F_TYPE_STEPS", "300"))
THRESHOLD = float(os.environ.get("R17_T3F_TYPE_THRESHOLD", "0.95"))
SEED = int(os.environ.get("R17_T3F_TYPE_SEED", "20261020"))
CACHE_INPUT = os.environ.get("R17_T3F_TYPE_CACHE", "")
HARD_WEIGHT = float(os.environ.get("R17_T3F_TYPE_HARD_WEIGHT", "1"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class TypeResidual(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 48), nn.GELU(), nn.Linear(48, 4))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def type_id(name: str) -> int:
    name = str(name).upper()
    if name not in BOND_TYPES:
        raise ValueError(f"unsupported true bond type: {name}")
    return BOND_TYPES.index(name)


def residual_logits(head, features, frozen_logits):
    """The identical composition must be used for fitting and replay."""
    return frozen_logits + head(features)


def cache_binding(train_rows, hold_rows):
    files = {}
    for row in train_rows + hold_rows:
        for suffix in (".png", ".graph.json", ".layers.json"):
            path = SOURCE / (row["sample_key"] + suffix)
            files[path.name] = digest(path)
    return {
        "initial_checkpoint_sha256": SHA,
        "manifest_sha256": digest(COHORT),
        "source_files_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "extractor_sha256": digest(Path(base.__file__)),
        "train_keys": [r["sample_key"] for r in train_rows],
        "holdout_keys": [r["sample_key"] for r in hold_rows],
    }


def validate_cache(cache, binding):
    if cache.get("binding") != binding:
        raise RuntimeError("feature cache input binding mismatch")
    for split, key in (("train", "train_keys"), ("holdout", "holdout_keys")):
        if [r["key"] for r in cache[split]] != binding[key]:
            raise RuntimeError("feature cache split/key mismatch")


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(base.INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed")
    base.SOURCE = SOURCE
    base.PAIR_THRESHOLD = THRESHOLD
    manifest = json.loads(COHORT.read_text())
    train_rows, hold_rows = base.choose(manifest)
    if {r["molecule_id"] for r in train_rows} & {r["molecule_id"] for r in hold_rows}:
        raise RuntimeError("identity overlap")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    random.seed(SEED)
    if HARD_WEIGHT < 1:
        raise ValueError("hard-example weight must be >= 1")
    state = torch.load(PRESENCE, map_location="cpu", weights_only=False)
    if state.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("presence checkpoint binding mismatch")
    presence = base.Residual(262)
    presence.load_state_dict(state["state_dict"])
    presence.eval()
    binding = cache_binding(train_rows, hold_rows)
    if CACHE_INPUT:
        cache = torch.load(ROOT / CACHE_INPUT, map_location="cpu", weights_only=False)
        validate_cache(cache, binding)
        train, hold = cache["train"], cache["holdout"]
    else:
        model, config, _ = base.load_bundle_for_probe(base.INITIAL, SHA)
        model.eval().requires_grad_(False)
        train, hold = [], []
        for name, rows, items in (("train", train_rows, train), ("holdout", hold_rows, hold)):
            for i, row in enumerate(rows):
                items.append(base.extract(row, model, config))
                if (i + 1) % 24 == 0:
                    print(json.dumps({"phase": "extract", "split": name, "images": i + 1}), flush=True)
        cache = {"binding": binding, "train": train, "holdout": hold}
    # Restore the same head initialization stream whether features are loaded
    # or freshly extracted; model loading must not change experiment seed.
    torch.manual_seed(SEED)

    def feature(item, indices):
        return torch.cat((item["features"][indices], item["chemical_logits"][indices]), dim=-1)

    # Fit only on true mapped bonds; false candidate rows have no type target.
    xs, ys = [], []
    with torch.inference_mode():
        for item in train:
            pos = torch.where(item["labels"] > 0.5)[0]
            if len(pos):
                xs.append(feature(item, pos))
                ys.append(torch.tensor([type_id(item["type_labels"][i]) for i in pos.tolist()], dtype=torch.long))
    x = torch.cat(xs); y = torch.cat(ys)
    head = TypeResidual(x.shape[-1])
    with torch.no_grad():
        if not torch.equal(residual_logits(head, x, x[:, -4:]), x[:, -4:]):
            raise RuntimeError("zero residual does not preserve baseline")
    counts = torch.bincount(y, minlength=4).float()
    weights = (len(y) / counts.clamp_min(1)).sqrt()
    row_weights = torch.where(x[:, -4:].argmax(-1) != y, HARD_WEIGHT, 1.0)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    events = []
    for step in range(1, STEPS + 1):
        losses = F.cross_entropy(residual_logits(head, x, x[:, -4:]), y, weight=weights, reduction="none")
        loss = (losses * row_weights).sum() / (weights[y] * row_weights).sum()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        if step in (1, 100, STEPS):
            events.append({"step": step, "loss": float(loss.detach())})
    head.eval()

    def score(items, corrected, use_presence=True):
        out = {}
        for profile in sorted({item["profile"] for item in items}):
            pred = true = typed_base = typed = target = reachable = 0
            for item in items:
                if item["profile"] != profile: continue
                with torch.inference_mode():
                    presence_logits = item["base_logits"]
                    if use_presence:
                        presence_logits = presence_logits + presence(item["features"])
                    selected = torch.where(presence_logits.sigmoid() >= THRESHOLD)[0]
                    if corrected:
                        logits = residual_logits(head, feature(item, selected), item["chemical_logits"][selected])
                    else:
                        logits = item["chemical_logits"][selected]
                pred += len(selected)
                target += len(item["target_edges"])
                reachable += int(item["labels"].sum())
                true += int(item["labels"][selected].sum())
                for j, idx in enumerate(selected.tolist()):
                    if item["labels"][idx] > 0.5:
                        truth = type_id(item["type_labels"][idx])
                        base_kind = int(item["chemical_logits"][idx].argmax())
                        typed_base += int(base_kind == truth)
                        typed += int(int(logits[j].argmax()) == truth)
            out[profile] = {"predicted_edges": pred, "true_edges": true, "target_edges": target,
                            "reachable_target_edges": reachable,
                            "edge_precision": true / max(pred, 1), "edge_recall": true / max(target, 1),
                            "typed_precision": typed / max(pred, 1), "typed_recall": typed / max(target, 1),
                            "baseline_typed_precision": typed_base / max(pred, 1), "baseline_typed_recall": typed_base / max(target, 1)}
        return out

    frozen = score(hold, False, use_presence=False)
    baseline = score(hold, False); corrected = score(hold, True)
    deltas = {p: {k: corrected[p][k] - baseline[p][k] for k in ("edge_precision", "edge_recall", "typed_precision", "typed_recall")} for p in baseline}
    frozen_deltas = {p: {k: corrected[p][k] - frozen[p][k] for k in ("edge_precision", "edge_recall", "typed_precision", "typed_recall")} for p in frozen}
    after = digest(base.INITIAL)
    gate = {"checkpoint_unchanged": after == SHA, "sealed_test_read": False,
            "edge_set_frozen": True, "oa13_typed_recall_gain": deltas["oa13"]["typed_recall"] > 0,
            "oa33_typed_recall_gain": deltas["oa33"]["typed_recall"] > 0,
            "each_profile_typed_precision_drop_le_0.01": all(v["typed_precision"] >= -0.01 for v in deltas.values()),
            "each_profile_edge_precision_drop_le_0.01": all(v["edge_precision"] >= -0.01 for v in deltas.values()),
            "versus_frozen_each_profile_typed_precision_drop_le_0.01": all(v["typed_precision"] >= -0.01 for v in frozen_deltas.values()),
            "versus_frozen_each_profile_edge_precision_drop_le_0.01": all(v["edge_precision"] >= -0.01 for v in frozen_deltas.values()),
            "sanitize_evaluated": False, "perturbation_evaluated": False,
            "production_go": False}
    report = {"schema_version": "chemvector-r17-t3f-type-correction-cpu-0.2", "complete": True,
              "training_performed": True, "gpu_used": False, "sealed_test_read": False,
              "production_checkpoint_written": False, "train_molecules": len({r["molecule_id"] for r in train_rows}),
              "holdout_molecules": len({r["molecule_id"] for r in hold_rows}), "train_rows": len(y),
              "steps": STEPS, "seed": SEED, "events": events, "baseline": baseline, "corrected": corrected,
              "hard_example_weight": HARD_WEIGHT, "training_wrong_type_rows": int((row_weights > 1).sum()),
              "cache_binding": binding, "feature_cache_reused": bool(CACHE_INPUT),
              "frozen_baseline": frozen, "deltas_versus_frozen": frozen_deltas,
              "loss_composition": "cross_entropy(frozen_chemical_logits + type_residual, train_labels)",
              "recall_denominator": "all ground-truth bonds; reachable bonds reported separately",
              "cohort": str(COHORT), "cohort_sha256": digest(COHORT),
              "presence_sidecar_sha256": digest(PRESENCE),
              "deltas": deltas, "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": after, "gate": gate}
    OUT.mkdir(parents=True)
    if not CACHE_INPUT:
        torch.save(cache, OUT / "feature-cache.pt")
    torch.save({"schema_version": report["schema_version"], "state_dict": head.state_dict(),
                "training_cohort": str(COHORT), "training_cohort_sha256": digest(COHORT),
                "train_molecule_ids": sorted({r["molecule_id"] for r in train_rows}),
                "initial_checkpoint_sha256": SHA}, OUT / "sidecar-state.pt")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "baseline": baseline, "corrected": corrected, "deltas": deltas, "gate": gate}, indent=2))
    return report


if __name__ == "__main__":
    run()
