"""Fixed CPU train/eval counterfactual; never changes the training recipe."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys

import torch

from .runtime import ResourceBudget, run_budgeted

SEEDS = (20260915, 20260916, 20260917, 20260918)
MODES = ("eval", "train", "train_vision_dropout_off", "eval_vision_dropout_on")
PRIOR = Path("outputs/audits/chemvector-freeze-regression-probe-20260915/worker.log")
PRIOR_SHA = "6fb9c41c9cf42f139d4a1706f5d76c0ee248dd720014f905e350b024c82a0b51"
OUTPUT = Path("outputs/audits/chemvector-train-eval-modes-20260915")
WITNESS = "curriculum:curr_8419bd7a8bf844e9__bond_short"


def emit(event, **record):
    print(json.dumps({"event": event, **record}, allow_nan=False), flush=True)


@contextmanager
def temporary_mode(model, mode):
    """Restore every mixed-mode flag even on failure; never edit dropout p."""
    if mode not in MODES:
        raise ValueError("Unknown diagnostic mode")
    previous = [(m, m.training) for m in model.modules()]
    try:
        model.train(mode.startswith("train"))
        if mode in ("train_vision_dropout_off", "eval_vision_dropout_on"):
            enabled = mode == "eval_vision_dropout_on"
            # Exactly the stages used by structure_forward, excluding frozen
            # stride-32 stages. Role freeze and gradient flags remain untouched.
            for stage in model.vision.stages[:5]:
                for module in stage.modules():
                    if isinstance(module, torch.nn.Dropout2d):
                        module.train(enabled)
        yield
    finally:
        for module, training in previous:
            module.training = training


def measure(model, batch, config, atom_threshold, mode, seed):
    """Read-only predictions, then target-based coverage (not SDF accuracy)."""
    from .continuation import structure_forward
    from .interface_diagnostic import tensor_tree_digest
    from chemvector.train_direct_atom_pair_bond import match_atom_targets, build_pair_targets
    from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs

    with temporary_mode(model, mode), torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        forward = structure_forward(model, batch["pixel_values"], config, atom_threshold)
        raw, bonds, sets, selected = forward[3:]
        _, raw_matched = match_atom_targets(raw, batch, config)
        mapping, selected_matched = match_atom_targets(selected, batch, config)
        pairs, indices = enumerate_unordered_atom_pairs(selected["atom_mask"])
        targets = build_pair_targets(pairs, indices, mapping, batch, selected)
        atom_count = int(batch["atom_mask"].sum())
        bond_count = int(batch["bond_mask"].sum())
        assert batch["atom_mask"].shape[0] == 1
        refs = batch["bond_atom_refs"][0, :bond_count]
        reachable = selected_matched[0, refs].all(-1)
        assert int(reachable.sum()) == int(targets["presence"].sum())
        mapped = mapping[mapping >= 0]
        distances = torch.cdist(raw["atom_geometry"][0, raw["atom_mask"][0], :2].float(),
                                batch["atom_anchors"][0, :atom_count].float()) * config.image_size
        margins = sets["atom_presence_logits"][0, raw["atom_mask"][0]] - atom_threshold
        target_rows = []
        for target in range(atom_count):
            near = distances[:, target] <= config.atom_match_radius_px
            target_rows.append({"target": target, "nearby_raw_candidates": int(near.sum()),
                "maximum_nearby_selection_margin": float(margins[near].max()) if bool(near.any()) else None,
                "nearest_raw_distance_px": float(distances[:, target].min()) if len(distances) else None,
                "matched_raw": bool(raw_matched[0, target]),
                "matched_selected": bool(selected_matched[0, target])})
        return {"mode": mode, "seed": seed, "target_atoms": atom_count, "target_bonds": bond_count,
            "raw_atoms": int(raw["atom_mask"].sum()), "raw_matched": int(raw_matched.sum()),
            "selected_atoms": int(selected["atom_mask"].sum()), "selected_matched": int(selected_matched.sum()),
            "unmatched_selected": int(selected["atom_mask"].sum()) - int(selected_matched.sum()),
            "pair_candidates": len(pairs), "reachable_positive_pairs": int(reachable.sum()),
            "bbox_rows": int(batch["atom_text_mask"][0, mapped].sum()),
            "target_text_atoms": int(batch["atom_text_mask"].sum()),
            "matched_target_ids": selected_matched[0].nonzero().flatten().tolist(),
            "reachable_target_bond_ids": reachable.nonzero().flatten().tolist(),
            "target_details": target_rows,
            "prediction_sha256": tensor_tree_digest({"features": forward[0], "raw": raw,
                                                     "bonds": bonds, "set": sets, "selected": selected})}


def summarize(records):
    result = {}
    for model in dict.fromkeys(r["model"] for r in records):
        result[model] = {}
        for mode in MODES:
            group = [r for r in records if r["model"] == model and r["mode"] == mode]
            totals = {k: sum(r[k] for r in group) for k in (
                "target_atoms", "target_bonds", "raw_matched", "selected_atoms", "selected_matched",
                "reachable_positive_pairs", "bbox_rows", "target_text_atoms", "unmatched_selected")}
            result[model][mode] = {"checks": len(group), **totals,
                "atom_match_recall": totals["selected_matched"] / totals["target_atoms"],
                "positive_bond_reachability": totals["reachable_positive_pairs"] / totals["target_bonds"],
                "no_pair_branch_checks": sum(r["pair_candidates"] == 0 for r in group),
                "no_positive_pair_checks": sum(r["reachable_positive_pairs"] == 0 for r in group),
                "all_target_atoms_matched_checks": sum(r["selected_matched"] == r["target_atoms"] for r in group)}
    return result


def audit(output):
    from . import continuation
    from .bundle import load_bundle
    from .checkpoints import file_digest, verify_sources
    from .data import CurriculumCollator, load_sample
    from .gradient_preflight import state_digest
    from chemvector.atom_bond_scene import AtomBondSceneConfig

    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    assert file_digest(PRIOR) == PRIOR_SHA
    prior = [json.loads(line) for line in PRIOR.read_text().splitlines()]
    old_plan = next(r for r in prior if r["event"] == "fixed_plan_before_models_or_pixels")
    prior_counts = {(r["key"], r["model"]): r for r in prior if r["event"] == "matching"}
    keys = old_plan["training_keys"]
    assert len(keys) == len(set(keys)) == 12 and WITNESS in keys
    bindings = dict(old_plan["input_sha256"])
    bindings[str(PRIOR)] = PRIOR_SHA
    for path in (Path(__file__), Path(continuation.__file__)):
        bindings[str(path)] = file_digest(path)
    for path, digest in bindings.items():
        assert file_digest(path) == digest
    run = Path("outputs/experiments/chemvector-expanded-freeze-role-20260915")
    report = json.loads((run / "report.json").read_text())
    provenance = json.loads((run / "run.json").read_text())["provenance"]
    manifest = json.loads(Path("outputs/datasets/chemvector-expanded-release-20260915/manifest.json").read_text())
    rows = {r["source"] + ":" + r["sample_key"]: r for r in manifest["samples"]}
    for key in keys:
        row = rows[key]
        assert row["split"] == "train"
        meta = Path(manifest["sources"][row["source"]]) / "samples" / (row["sample_key"] + ".meta.json")
        assert file_digest(meta) == row["meta_sha256"]
        bindings[str(meta)] = row["meta_sha256"]
    plan = {"training_keys": keys, "models": ["initial", "last"], "seeds": SEEDS, "modes": MODES,
        "deterministic_modes_evaluated_once_per_sample": ["eval", "train_vision_dropout_off"],
        "planned_forwards": 240, "backward": {"key": WITNESS, "model": "initial", "mode": "train_vision_dropout_off"},
        "input_sha256": bindings, "gpu": False, "optimizer_steps": 0,
        "scope": "fixed training-sample coverage; no full-molecule accuracy or training efficacy estimate"}
    emit("fixed_plan_before_models_or_pixels", **plan)
    records, comparisons, states = [], [], {}
    for name, binding in (("initial", report["selected"]), ("last", report["last"])):
        model, config, payload = load_bundle(binding["path"], expected_sha256=binding["sha256"])
        assert payload["provenance"] == provenance and model.update_recipe == "freeze-role-v1"
        before = state_digest(model)
        mode_flags = [m.training for m in model.modules()]
        grad_flags = [p.requires_grad for p in model.parameters()]
        dropouts = [{"name": n, "p": m.p} for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout2d)]
        emit("model_loaded", model=name, state_sha256=before, dropouts=dropouts)
        collator = CurriculumCollator(AtomBondSceneConfig(
            atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds), purpose="diagnostic")
        for key in keys:
            row = rows[key]
            sample = load_sample(manifest["sources"][row["source"]], row["sample_key"], purpose="diagnostic")
            batch = collator([sample])
            assert batch["purpose"] == "diagnostic" and not batch["training_authorized"]
            cases = {}
            for mode in MODES:
                seeds = SEEDS if mode in ("train", "eval_vision_dropout_on") else SEEDS[:1]
                for seed in seeds:
                    result = {"key": key, "model": name,
                              **measure(model, batch, config, provenance["atom_threshold"], mode, seed)}
                    cases[(mode, seed)] = result
                    records.append(result)
                    emit("measurement", **result)
            baseline = cases[("eval", SEEDS[0])]
            assert baseline["selected_matched"] == prior_counts[(key, name)]["old_matched"]
            assert baseline["reachable_positive_pairs"] == prior_counts[(key, name)]["old_positive_pairs"]
            off = cases[("train_vision_dropout_off", SEEDS[0])]
            paired = [{"seed": seed,
                "train_equals_eval_dropout_on": cases[("train", seed)]["prediction_sha256"] == cases[("eval_vision_dropout_on", seed)]["prediction_sha256"],
                "lost_vs_eval_target_atoms": sorted(set(baseline["matched_target_ids"]) - set(cases[("train", seed)]["matched_target_ids"])),
                "lost_vs_eval_target_bonds": sorted(set(baseline["reachable_target_bond_ids"]) - set(cases[("train", seed)]["reachable_target_bond_ids"]))}
                for seed in SEEDS]
            comparison = {"key": key, "model": name,
                "train_dropout_off_equals_eval": baseline["prediction_sha256"] == off["prediction_sha256"], "paired_seeds": paired}
            comparisons.append(comparison)
            emit("comparison", **comparison)
            if key == WITNESS and name == "initial":
                with temporary_mode(model, "train_vision_dropout_off"), torch.random.fork_rng(devices=[]):
                    torch.manual_seed(SEEDS[0])
                    model.zero_grad(set_to_none=True)
                    losses, coverage = continuation.continuation_losses(model, batch, config,
                        atom_threshold=provenance["atom_threshold"], bond_threshold=provenance["bond_threshold"])
                    losses["loss"].backward()
                    gradients = continuation.gradient_report(model)
                    assert all(r["finite"] for r in gradients.values())
                    assert all(p.grad is None for p in model.role.parameters()) and not model.role.training
                    emit("backward", key=key, model=name, mode="train_vision_dropout_off", coverage=coverage,
                         gradients=gradients, optimizer_steps=0, role_frozen=True)
                    model.zero_grad(set_to_none=True)
                    del losses
            del sample, batch, cases
        assert state_digest(model) == before
        assert mode_flags == [m.training for m in model.modules()]
        assert grad_flags == [p.requires_grad for p in model.parameters()]
        assert dropouts == [{"name": n, "p": m.p} for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout2d)]
        assert all(p.grad is None for p in model.parameters())
        states[name] = before
        del model
    assert len(records) == plan["planned_forwards"]
    verify_sources(provenance["checkpoints"])
    for binding in (report["selected"], report["last"]):
        assert file_digest(binding["path"]) == binding["sha256"]
    for path, digest in bindings.items():
        assert file_digest(path) == digest
    assert not torch.cuda.is_initialized()
    result = {"plan": plan, "summary": summarize(records), "comparisons": comparisons,
        "train_dropout_off_equals_eval_count": sum(r["train_dropout_off_equals_eval"] for r in comparisons),
        "train_equals_eval_dropout_on_count": sum(s["train_equals_eval_dropout_on"] for r in comparisons for s in r["paired_seeds"]),
        "records": records, "states_sha256": states, "states_and_modes_unchanged": True,
        "source_files_unchanged": True, "gpu_used": False, "optimizer_steps": 0, "accuracy_recovery_evaluated": False}
    with (Path(output) / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    emit("completed", **{k: v for k, v in result.items() if k not in ("records", "comparisons", "plan")})
    return result


def main():
    if sys.argv[1:] == ["--worker"]:
        audit(OUTPUT)
        return 0
    if sys.argv[1:]:
        raise ValueError("This fixed diagnostic accepts no runtime sample, seed or recipe overrides")
    receipt = run_budgeted([sys.executable, "-m", "chemvector.training.curriculum.mode_diagnostic", "--worker"],
                           output=OUTPUT, budget=ResourceBudget(300))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
