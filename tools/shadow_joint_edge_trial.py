"""Ten-step in-memory shadow trial for the conservative joint edge heads.

No checkpoint or optimizer state is persisted.  This is the last preflight
before a separately authorized 400-attempt GPU trial.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.training.curriculum.chemical_head_probe import (
    INITIAL, _pair_output, _load_preparation,
)
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.joint_edge_objective import joint_edge_loss
from chemvector.training.curriculum.pair_scale_plan import sample_id
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.supervision_contract import pair_targets_with_unknown, select_known_pair_rows
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.graph.relations.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/audits/chemvector-joint-edge-shadow-20261005"
WITNESSES = (
    "full_molecule_expansion:curr_9b54d301940543f8__font_large",
    "full_molecule_expansion:curr_75deb03a71d9a1d3__font_small",
    "full_molecule_expansion:curr_e6354f1e02efa8bd__reference",
)
STEPS = 10
PAIR_THRESHOLD = 0.85


def _row_bundle(model, config, loader, manifest, row, collator, device, atom_threshold):
    sample = loader(manifest, row)
    batch = collator([sample])
    tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        features, _, _, _, _, _, selected = structure_forward(
            model, tensors["pixel_values"], config, atom_threshold,
        )
        mapping, _ = match_atom_targets(selected, tensors, config)
        contract = pair_targets_with_unknown(selected, mapping, tensors)
        pairs, sample_indices = contract["pair_indices"], contract["sample_indices"]
        output = _pair_output(
            model.pair, features, selected, pairs, sample_indices,
        )
    return tensors, selected, contract, features, pairs, sample_indices, output


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite shadow output: {output}")
    prep, plan, protocol = _load_preparation()
    sources = source_checks()
    if file_digest(INITIAL) != prep["initial_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint changed")
    manifest = sources["manifest"]
    rows = {sample_id(r): r for r in manifest["samples"]}
    plan_rows = {sample_id(r): r for r in plan["samples"]}
    selected_rows = []
    for key in WITNESSES:
        if key not in rows or key not in plan_rows or rows[key]["split"] != "train":
            raise ValueError(f"Shadow witness is not a released training row: {key}")
        selected_rows.append(rows[key])
    model, config, payload = load_bundle_for_probe(INITIAL, prep["initial_checkpoint_sha256"])
    if payload["progress"]["optimizer_steps"] != 0 or not torch.cuda.is_available():
        raise ValueError("Shadow requires zero-step initial checkpoint and CUDA")
    torch.set_num_threads(2)
    torch.manual_seed(protocol["seed"])
    torch.cuda.manual_seed_all(protocol["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in (*model.pair.presence.parameters(), *model.pair.chemical.parameters()):
        parameter.requires_grad_(True)
    trainable = [*model.pair.presence.parameters(), *model.pair.chemical.parameters()]
    frozen_digest = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                     if name not in {"pair.presence.weight", "pair.presence.bias",
                                     "pair.chemical.weight", "pair.chemical.bias"}}
    initial_heads = {name: value.detach().cpu().clone() for name, value in model.pair.state_dict().items()
                     if name in {"presence.weight", "presence.bias", "chemical.weight", "chemical.bias"}}
    loader = RecordLoader(manifest)
    collator = CurriculumCollator(AtomBondSceneConfig(atom_queries=128, bond_queries=256), purpose="training")
    baseline_masks = {}
    caches = {}
    for row in selected_rows:
        entry = _row_bundle(model, config, loader, manifest, row, collator, device,
                            payload["provenance"]["atom_threshold"])
        tensors, selected, contract, features, pairs, sample_indices, output_pair = entry
        baseline_masks[sample_id(row)] = (output_pair["presence_logits"].sigmoid() >= PAIR_THRESHOLD).detach().cpu()
        caches[sample_id(row)] = entry
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0.0)
    events = []
    started = time.monotonic()
    for step in range(1, STEPS + 1):
        row = selected_rows[(step - 1) % len(selected_rows)]
        key = sample_id(row)
        tensors, selected, contract, features, pairs, sample_indices, _ = caches[key]
        with torch.no_grad():
            initial_hidden = _pair_output(
                model.pair, features, selected, pairs, sample_indices,
            )["hidden"].detach()
            reference_presence = torch.nn.functional.linear(
                initial_hidden, initial_heads["presence.weight"].to(device),
                initial_heads["presence.bias"].to(device),
            ).squeeze(-1)
            reference_chemical = torch.nn.functional.linear(
                initial_hidden, initial_heads["chemical.weight"].to(device),
                initial_heads["chemical.bias"].to(device),
            )
        chosen = select_known_pair_rows(contract, selected, negative_ratio=6, minimum_negatives=48)
        if not len(chosen):
            raise ValueError("Shadow witness produced no known rows")
        output_pair = _pair_output(model.pair, features, selected, pairs, sample_indices)
        chosen_output = {k: v[chosen] for k, v in output_pair.items()}
        loss, details = joint_edge_loss(
            chosen_output,
            contract["known_mask"][chosen],
            contract["legacy_targets"]["presence"][chosen],
            contract["legacy_targets"]["chemical"][chosen],
            pair_indices=pairs[chosen], sample_indices=sample_indices[chosen],
            presence_weight=0.25, chemical_weight=1.0, replay_weight=1.0,
            reference_presence_logits=reference_presence[chosen],
            reference_chemical_logits=reference_chemical[chosen],
            aromatic_safety_weight=10.0, aromatic_anchor_weight=10.0,
        )
        if loss is None or not bool(torch.isfinite(loss)):
            raise ValueError("Shadow produced empty/non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in trainable):
            raise ValueError("Shadow produced missing/non-finite head gradients")
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        current = _row_bundle(model, config, loader, manifest, row, collator, device,
                              payload["provenance"]["atom_threshold"])[-1]
        current_mask = current["presence_logits"].sigmoid() >= PAIR_THRESHOLD
        if not torch.equal(current_mask.detach().cpu(), baseline_masks[key]):
            raise ValueError(f"Presence threshold mask changed in shadow: {key}")
        events.append({"step": step, "key": key, "loss": float(loss.detach()),
                       "gradient_norm": float(norm), "details": details,
                       "presence_mask_unchanged": True})
        optimizer.zero_grad(set_to_none=True)
    after_state = model.state_dict()
    changed_frozen = [name for name, before in frozen_digest.items()
                      if not torch.equal(after_state[name].detach().cpu(), before)]
    if changed_frozen:
        raise ValueError(f"Frozen parameters changed: {changed_frozen[:5]}")
    result = {
        "schema_version": "chemvector-joint-edge-shadow-0.1",
        "complete": True,
        "training_performed": False,
        "optimizer_steps": 0,
        "virtual_steps": STEPS,
        "persisted_parameter_updates": 0,
        "in_memory_virtual_parameter_updates": STEPS,
        "gpu_used": True,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "trainable_parameters": ["pair.presence.weight", "pair.presence.bias",
                                  "pair.chemical.weight", "pair.chemical.bias"],
        "frozen_parameter_changes": changed_frozen,
        "events": events,
        "presence_threshold_masks_unchanged": True,
        "source_sha256": {"initial.pt": file_digest(INITIAL)},
        "elapsed_seconds": time.monotonic() - started,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    value = run()
    print(json.dumps({"status": "completed", "output": str(OUTPUT),
                      "virtual_steps": value["virtual_steps"],
                      "presence_threshold_masks_unchanged": value["presence_threshold_masks_unchanged"]}, indent=2))
