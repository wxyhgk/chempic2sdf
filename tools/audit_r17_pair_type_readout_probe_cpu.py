#!/usr/bin/env python3
"""Read-only evaluation of the frozen-feature type readout shadow.

This separates type-readout capacity from pair reachability: all mapped true
edges in the t2 cache are evaluated, regardless of whether the presence route
would admit them.  It never changes a checkpoint or reads sealed data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chemvector.bond_object import BOND_TYPES
from tools.train_r17_pair_type_residual_cpu import TypeResidual, digest
from tools.train_r17_pair_incidence_known_clean_cpu import Residual
from tools.train_r17_pair_incidence_known_clean_cpu import SHA
from chemvector.training.curriculum.chemical_head_probe import INITIAL

CACHE = ROOT / os.environ.get("R17_TYPE_READOUT_CACHE", "outputs/audits/chemvector-r17-targeted-source-pair-calibration-cpu-20261018-t2/features-cache.pt")
STATE = ROOT / os.environ.get("R17_TYPE_READOUT_STATE", "outputs/audits/chemvector-r17-targeted-source-pair-type-residual-cpu-20261018-t2-r2/type-residual-state.pt")
PRESENCE_STATE = ROOT / os.environ.get("R17_TYPE_READOUT_PRESENCE_STATE", "outputs/audits/chemvector-r17-targeted-source-pair-incidence-cpu-20261018-t2/sidecar-state.pt")
OUT = ROOT / os.environ.get("R17_TYPE_READOUT_OUT", "outputs/audits/chemvector-r17-targeted-source-pair-type-readout-probe-cpu-20261018-t2")


def score(items, head, presence_head=None):
    confusion_base = [[0] * 4 for _ in range(4)]
    confusion_refined = [[0] * 4 for _ in range(4)]
    added_base = [[0] * 4 for _ in range(4)]
    added_refined = [[0] * 4 for _ in range(4)]
    base_logit = torch.tensor(0.72 / (1.0 - 0.72)).log()
    for item in items:
        for i, label in enumerate(item["type_labels"]):
            if label is None or item["labels"][i] <= 0.5:
                continue
            y = BOND_TYPES.index(label)
            base = int(item["chemical_logits"][i].argmax())
            feat = torch.cat((item["features"][i].float(), item["chemical_logits"][i].float()))
            refined = int((item["chemical_logits"][i].float() + head(feat)).argmax())
            confusion_base[y][base] += 1
            confusion_refined[y][refined] += 1

            # A target-free raw sidecar-added subset, matching the bounded
            # presence trial's baseline threshold.  Targets are consulted
            # only after this mask is formed for the diagnostic counters.
            if presence_head is not None:
                residual = presence_head(item["features"].float()).detach()
                # Match the raw 600-step presence trial: its score uses the
                # residual at unit scale.  This is a diagnostic candidate
                # union, not the stricter train-only calibrated route.
                additions = (item["base_logits"] < base_logit) & (
                    item["base_logits"] + residual >= base_logit
                )
                if bool(additions[i]):
                    added_base[y][base] += 1
                    added_refined[y][refined] += 1

    def metrics(conf):
        total = sum(map(sum, conf)); correct = sum(conf[i][i] for i in range(4))
        f1 = []
        for i in range(4):
            tp = conf[i][i]; fn = sum(conf[i]) - tp; fp = sum(conf[j][i] for j in range(4)) - tp
            precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
            f1.append(2 * precision * recall / max(precision + recall, 1e-12))
        return {"rows": total, "accuracy": correct / max(total, 1), "macro_f1": sum(f1) / 4,
                "confusion": conf, "aromatic_to_single": conf[3][0]}
    result = {"frozen": metrics(confusion_base), "refined": metrics(confusion_refined)}
    if presence_head is not None:
        result["sidecar_added_frozen"] = metrics(added_base)
        result["sidecar_added_refined"] = metrics(added_refined)
    return result


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != SHA:
        raise RuntimeError("initial checkpoint changed")
    cache = torch.load(CACHE, map_location="cpu")
    state = torch.load(STATE, map_location="cpu")
    head = TypeResidual(266); head.load_state_dict(state["state_dict"]); head.eval()
    presence_state = torch.load(PRESENCE_STATE, map_location="cpu")
    presence = Residual(262); presence.load_state_dict(presence_state["state_dict"]); presence.eval()
    result = {"schema_version": "chemvector-r17-pair-type-readout-probe-cpu-0.1", "complete": True,
              "read_only": True, "training_performed": False, "gpu_used": False,
              "sealed_test_read": False, "production_checkpoint_written": False,
              "cache": str(CACHE.relative_to(ROOT)), "cache_sha256": digest(CACHE),
              "state": str(STATE.relative_to(ROOT)), "state_sha256": digest(STATE),
              "presence_state": str(PRESENCE_STATE.relative_to(ROOT)),
              "splits": {split: score(cache[split], head, presence) for split in ("train", "hold")},
              "gate": {"checkpoint_unchanged": digest(INITIAL) == SHA, "sealed_test_read": False,
                       "gpu_used": False, "production_go": False}}
    OUT.mkdir(parents=True)
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT / "report.json"), "splits": result["splits"], "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    run()
