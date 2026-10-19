#!/usr/bin/env python3
"""Paired CPU probe of the r17 element-head edge-context contract.

Only compiler element logits differ: the existing replay aggregates every
unordered candidate pair whereas training aggregates presence-selected pairs.
Targets are joined after both target-free compiler calls, for audit only.
No checkpoint is written and no threshold/head is fitted.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import audit_r17_selector_graph_replay as replay
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.components.graph.joint.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from tools.train_element_context_head_cpu_r17 import ElementContextHead, ELEMENTS
from tools.train_type_residual_cpu_r17 import TypeContextHead
from tools.train_graph_context_aromatic_cpu_r17 import AromaticContextHead

OUT = ROOT / "outputs/audits/chemvector-r17-element-context-scope-cpu-20261008-v1"
MOLECULES_PER_SPLIT = 4


class ScopedElementHead:
    """Replace pair-context features only, preserving atom order and head."""

    def __init__(self, head, nodes, present_features):
        self.head = head
        self.nodes = nodes
        self.present_features = present_features
        self.logits = None
        self.no_incident_nodes = 0

    def __call__(self, all_pair_features):
        rows = []
        for index, node in enumerate(self.nodes):
            context = self.present_features[node]
            if context:
                values = torch.stack(context)
                rows.append(torch.cat((all_pair_features[index, :128], values.mean(0), values.max(0).values, all_pair_features.new_tensor([len(context) / 8.0]))))
            else:
                # The trained head has no examples of degree-zero nodes.
                # Fail closed to the exact original logits for those nodes;
                # do not invent an untrained empty-context representation.
                rows.append(all_pair_features[index])
                self.no_incident_nodes += 1
        self.logits = self.head(torch.stack(rows))
        return self.logits


class CapturingHead:
    def __init__(self, head):
        self.head = head
        self.logits = None

    def __call__(self, features):
        self.logits = self.head(features)
        return self.logits


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if replay.digest(INITIAL) != replay.SHA:
        raise RuntimeError("initial checkpoint changed")
    cache = torch.load(replay.CACHE, map_location="cpu", weights_only=False)
    model, config, _ = load_bundle_for_probe(INITIAL, replay.SHA)
    model.cpu().eval().requires_grad_(False)
    element_state = torch.load(replay.ELEMENT_STATE, map_location="cpu", weights_only=False)
    type_state = torch.load(replay.TYPE_STATE, map_location="cpu", weights_only=False)
    aromatic_state = torch.load(replay.AROMATIC_STATE, map_location="cpu", weights_only=False)
    for state in (element_state, type_state, aromatic_state, cache):
        if state["initial_checkpoint_sha256"] != replay.SHA:
            raise RuntimeError("state/cache checkpoint binding mismatch")
    element = ElementContextHead().eval()
    element.load_state_dict(element_state["state_dict"])
    type_head = TypeContextHead(262).eval()
    type_head.load_state_dict(type_state["state_dict"])
    aromatic = AromaticContextHead(526).eval()
    aromatic.load_state_dict(aromatic_state["state_dict"])
    rows = []
    with torch.inference_mode():
        for split in ("train", "dev", "fresh_holdout"):
            records = cache["records"][split]
            ids = sorted({r["sample_key"].split("__")[0] for r in records})[:MOLECULES_PER_SPLIT]
            chosen = [r for r in records if r["sample_key"].split("__")[0] in ids]
            print(json.dumps({"phase": "frozen_forward", "split": split, "images": len(chosen)}), flush=True)
            for record in chosen:
                key = record["sample_key"]
                pixels = image_tensor(Image.open(replay.SOURCE / f"{key}.png").convert("RGB"))[None]
                outputs = structure_forward(model, pixels, config, -.25)
                vision, raw, selected = outputs[0], outputs[3], outputs[6]
                mask = selected["atom_mask"][0].bool()
                nodes = torch.where(mask)[0].tolist()
                pairs, sample_indices = enumerate_unordered_atom_pairs(mask[None])
                direct = model.pair(vision, raw["atom_features"], raw["atom_geometry"][..., :2], raw["atom_scores"], pairs, sample_indices)
                points = raw["atom_geometry"][0, :, :2][pairs]
                distances = torch.linalg.vector_norm((points[:, 1] - points[:, 0]) * 768., dim=-1)
                features = torch.cat((direct["hidden"].float(), direct["presence_logits"][:, None].float(), points.mean(1), distances[:, None], raw["atom_scores"][0, pairs].float()), -1)
                present = {node: [] for node in nodes}
                for index, (a, b) in enumerate(pairs.tolist()):
                    if float(direct["presence_logits"][index].sigmoid()) >= replay.PAIR_THRESHOLD:
                        present[a].append(features[index])
                        present[b].append(features[index])
                all_head = CapturingHead(element)
                present_head = ScopedElementHead(element, nodes, present)
                common = (raw, direct, mask, type_head)
                suffix = (aromatic, float(aromatic_state.get("threshold", .5)), nodes)
                all_compile = replay.compile_prediction(*common, all_head, *suffix)
                present_compile = replay.compile_prediction(*common, present_head, *suffix)
                # Every decision above has already completed without targets.
                atoms, centers, truth, types = replay.target(key)
                mapping = replay.match(raw["atom_geometry"][0, nodes, :2], centers)
                expected = [str(atoms[mapping[i]].get("element") or "C") for i in mapping]
                before = [ELEMENTS[int(all_head.logits[i].argmax())] for i in mapping]
                after = [ELEMENTS[int(present_head.logits[i].argmax())] for i in mapping]
                rows.append({"split": split, "sample_key": key, "profile": key.split("__")[-1], "atoms": len(nodes), "mapped_atoms": len(mapping), "all_pairs": len(pairs), "present_pairs": sum(len(v) for v in present.values()) // 2, "no_incident_nodes_baseline_fallback": present_head.no_incident_nodes, "all_pair_sanitize": bool(all_compile[0]), "present_pair_sanitize": bool(present_compile[0]), "all_pair_error": all_compile[1], "present_pair_error": present_compile[1], "all_pair_element_correct": sum(a == b for a, b in zip(expected, before)), "present_pair_element_correct": sum(a == b for a, b in zip(expected, after)), "element_changes": sum(a != b for a, b in zip(before, after)), "target_elements": expected, "all_pair_elements": before, "present_pair_elements": after})
    summary = {}
    for split in ("train", "dev", "fresh_holdout"):
        rs = [r for r in rows if r["split"] == split]
        n = len(rs)
        denominator = sum(r["mapped_atoms"] for r in rs)
        summary[split] = {"images": n, "mapped_atoms": denominator, "all_pair_sanitize": sum(r["all_pair_sanitize"] for r in rs) / n, "present_pair_sanitize": sum(r["present_pair_sanitize"] for r in rs) / n, "all_pair_element_accuracy": sum(r["all_pair_element_correct"] for r in rs) / denominator, "present_pair_element_accuracy": sum(r["present_pair_element_correct"] for r in rs) / denominator, "valid_to_invalid": sum(r["all_pair_sanitize"] and not r["present_pair_sanitize"] for r in rs), "invalid_to_valid": sum(not r["all_pair_sanitize"] and r["present_pair_sanitize"] for r in rs), "element_changes": sum(r["element_changes"] for r in rs), "no_incident_nodes_baseline_fallback": sum(r["no_incident_nodes_baseline_fallback"] for r in rs), "all_pair_errors": dict(Counter(r["all_pair_error"] for r in rs)), "present_pair_errors": dict(Counter(r["present_pair_error"] for r in rs))}
    after_sha = replay.digest(INITIAL)
    result = {"schema_version": "chemvector-r17-element-context-scope-cpu-0.1", "complete": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "initial_checkpoint_sha256": replay.SHA, "initial_checkpoint_sha256_after": after_sha, "fixed_molecules_per_split": MOLECULES_PER_SPLIT, "contract": {"training": "aggregate frozen presence>=.72 candidate pairs only", "existing_replay": "aggregate all unordered active pairs", "candidate_probe": "aggregate frozen presence>=.72 candidate pairs; preserve original context for degree-zero nodes", "targets_used_by_compiler": False, "pair_threshold_changed": False, "type_or_aromatic_predictions_changed": False}, "summary": summary, "rows": rows, "gate": {"checkpoint_unchanged": after_sha == replay.SHA, "each_split_element_accuracy_no_drop": all(s["present_pair_element_accuracy"] >= s["all_pair_element_accuracy"] for s in summary.values()), "each_split_sanitize_no_drop": all(s["present_pair_sanitize"] >= s["all_pair_sanitize"] for s in summary.values()), "no_valid_to_invalid": all(s["valid_to_invalid"] == 0 for s in summary.values()), "production_go": False, "next": "contract diagnostic only; full identity/type/compile audit required before integration"}}
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "summary": summary, "gate": result["gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    run()
