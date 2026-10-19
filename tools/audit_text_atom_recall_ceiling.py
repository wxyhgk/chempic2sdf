"""Frozen-initial, read-only audit of the visible-text atom recall ceiling.

The report separates visible text atoms from implicit carbon and records raw
and selected proposal reachability.  With no candidate generation or
production changes, selected text recall cannot exceed raw text recall.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator, load_sample
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/manifest.json"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
OUTPUT = ROOT / "outputs/audits/chemvector-text-atom-recall-ceiling-20261017/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
BATCH_SIZE = 8


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ratio(block: Counter, prefix: str) -> float:
    return block[f"raw_{prefix}"] / max(block[f"target_{prefix}"], 1)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    rows = plan["rows"]
    if len(rows) != 10080:
        raise ValueError("expected fixed 10,080-image replay manifest")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    if digest(INITIAL) != INITIAL_SHA or not torch.cuda.is_available():
        raise RuntimeError("initial checkpoint binding or CUDA unavailable")
    device = torch.device("cuda:0")
    model.to(device).eval().requires_grad_(False)
    before = state_digest(model)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
                                                       bond_queries=config.maximum_bonds), purpose="diagnostic")
    by_profile = defaultdict(Counter)
    by_split = defaultdict(Counter)
    by_element = defaultdict(Counter)
    pending = []

    def flush(items):
        samples = [load_sample(SOURCE, row["sample_key"], purpose="diagnostic") for row in items]
        batch = collator(samples)
        tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        with torch.inference_mode():
            _, _, _, raw, _, _, selected = structure_forward(
                model, tensors["pixel_values"], config, payload["provenance"]["atom_threshold"])
        raw_map, _ = match_atom_targets(raw, tensors, config)
        selected_map, _ = match_atom_targets(selected, tensors, config)
        for index, row in enumerate(items):
            sample = samples[index]
            layers = {int(atom["atom_id"]): atom for atom in sample["layers"]["atoms"]}
            graph_atoms = sample["graph"]["atoms"]
            atom_ids = [int(value) for value in batch["atom_ids"][index]]
            raw_ids = {atom_ids[int(value)] for value in raw_map[index].tolist()
                       if 0 <= int(value) < len(atom_ids)}
            selected_ids = {atom_ids[int(value)] for value in selected_map[index].tolist()
                            if 0 <= int(value) < len(atom_ids)}
            profile_block = by_profile[row["profile_id"]]
            split_block = by_split[row["split"]]
            for atom in graph_atoms:
                atom_id = int(atom["atom_id"])
                visible = bool(layers[atom_id]["visible_text"])
                kind = "visible_text" if visible else "implicit_C" if atom["element"] == "C" else "other_implicit"
                element = atom["element"]
                for block in (profile_block, split_block, by_element[(row["profile_id"], element, visible)]):
                    block[f"target_{kind}"] += 1
                    block[f"target_atoms"] += 1
                    block[f"raw_{kind}"] += int(atom_id in raw_ids)
                    block[f"selected_{kind}"] += int(atom_id in selected_ids)
                    block[f"raw_matched"] += int(atom_id in raw_ids)
                    block[f"selected_matched"] += int(atom_id in selected_ids)
                profile_block["images"] += 1
                split_block["images"] += 1

    for row in rows:
        pending.append(row)
        if len(pending) == BATCH_SIZE:
            flush(pending); pending = []
    if pending:
        flush(pending)
    after = state_digest(model)
    if before != after or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen initial state changed")

    def finish(block):
        result = dict(block)
        for kind in ("visible_text", "implicit_C", "other_implicit"):
            target = block[f"target_{kind}"]
            result[f"raw_{kind}_recall"] = block[f"raw_{kind}"] / max(target, 1)
            result[f"selected_{kind}_recall"] = block[f"selected_{kind}"] / max(target, 1)
        result["raw_atom_recall"] = block["raw_matched"] / max(block["target_atoms"], 1)
        result["selected_atom_recall"] = block["selected_matched"] / max(block["target_atoms"], 1)
        result["text_ceiling_minus_selected"] = result["raw_visible_text_recall"] - result["selected_visible_text_recall"]
        return result

    profile_result = {key: finish(value) for key, value in sorted(by_profile.items())}
    split_result = {key: finish(value) for key, value in sorted(by_split.items())}
    element_result = {"%s|%s|%s" % (key[0], key[1], "visible" if key[2] else "implicit"): finish(value)
                      for key, value in sorted(by_element.items())}
    text_targets = sum(value["target_visible_text"] for value in by_profile.values())
    report = {
        "schema_version": "chemvector-text-atom-recall-ceiling-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": True, "model_forwards": 1260, "sealed_test_read": False,
        "initial_sha256": INITIAL_SHA, "initial_sha256_after": digest(INITIAL),
        "plan_sha256": digest(PLAN), "source_summary_sha256": digest(SOURCE / "summary.json"),
        "replay_images": len(rows), "target_visible_text_atoms": text_targets,
        "by_profile": profile_result, "by_split": split_result,
        "by_element_and_visibility": element_result,
        "ceiling_definition": "without new candidate generation or production changes, selected visible-text atom recall <= raw visible-text atom recall",
        "data_reconstruction_gate": {
            "text_targets_have_same_source_layers": True,
            "all_rows_loaded_by_diagnostic_collator": True,
            "sealed_test_read": False,
            "training_authorized": False,
        },
        "next": "if visible-text raw recall is low while implicit-C recall is high, restructure text/glyph representation or candidate generation before selector training",
        "state_before": before, "state_after": after,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"by_profile": profile_result, "by_split": split_result}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
