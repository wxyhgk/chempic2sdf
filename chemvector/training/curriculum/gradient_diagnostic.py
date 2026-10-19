"""Fixed CPU loss-gradient attribution; no optimizer, weight update or search."""

import inspect
import json
import math
from pathlib import Path
import sys

import torch

from .mode_diagnostic import PRIOR, PRIOR_SHA, WITNESS, emit, temporary_mode
from .runtime import ResourceBudget, run_budgeted

TASKS = ("atom", "bond", "set_head", "pair", "bbox")
SEED = 20260915
KEY_INDICES = (0, 1, 8, 11)
OUTPUT = Path("outputs/audits/chemvector-loss-gradients-20260915-v2")
GRADIENT_SUM_FP32_RTOL = 1e-3


def cosine(dot, norm_a, norm_b):
    # An absent/zero gradient has no direction; do not label it orthogonal.
    return max(-1., min(1., dot / norm_a / norm_b)) if norm_a > 0 and norm_b > 0 else None


def cold_adamw_direction(gradient, learning_rate, clip_coefficient):
    """Bias-corrected first AdamW step, empty moments and weight_decay=0.

    This returns a NEW tensor; it never writes to a parameter or creates an
    optimizer. It is not the direction of a historical warm-momentum step.
    """
    clipped = gradient * clip_coefficient
    return -learning_rate * clipped / (clipped.abs() + 1e-8)


def attribute_losses(losses, groups):
    """Differentiate all tasks on ONE graph; leaves parameter .grad untouched."""
    parameters = [p for group in groups for p in group["params"]]
    if not parameters or len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Gradient attribution needs unique nonempty parameter groups")
    if any(not p.requires_grad or p.grad is not None for p in parameters):
        raise ValueError("Gradient attribution requires trainable parameters with empty .grad")
    task_grads = [torch.autograd.grad(losses[task], parameters, retain_graph=True, allow_unused=True)
                  for task in TASKS]
    total_grads = torch.autograd.grad(losses["loss"], parameters, allow_unused=True)
    if any(not bool(torch.isfinite(g).all()) for gradients in (*task_grads, total_grads)
           for g in gradients if g is not None):
        raise ValueError("Non-finite diagnostic gradient")
    total_squared = sum(float(g.double().square().sum()) for g in total_grads if g is not None)
    total_norm = math.sqrt(total_squared)
    clip = min(1., 1. / (total_norm + 1e-6))

    def flat(gradients, start, count):
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1).double()
                          for p, g in zip(parameters[start:start + count], gradients[start:start + count])])

    result, start, sum_error_squared = {}, 0, 0.
    for group in groups:
        count = len(group["params"])
        matrix = torch.stack([flat(g, start, count) for g in task_grads])
        joint = flat(total_grads, start, count)
        gram = matrix @ matrix.T
        norms = gram.diag().clamp_min(0).sqrt()
        auxiliary = matrix[:3].sum(0)
        aux_norm = float(auxiliary.norm())
        pair_norm, joint_norm = float(norms[3]), float(joint.norm())
        error_squared = float((matrix.sum(0) - joint).square().sum())
        sum_error_squared += error_squared
        direction = cold_adamw_direction(joint, group["lr"], clip)
        linear = matrix @ direction
        pair_dot_joint = float(matrix[3] @ joint)
        pair_dot_aux = float(matrix[3] @ auxiliary)
        result[group["name"]] = {
            "parameters": joint.numel(), "learning_rate_unchanged": group["lr"],
            "task_order": TASKS, "task_gradient_l2": dict(zip(TASKS, norms.tolist())),
            "task_nonzero_tensors": {task: sum(g is not None and bool(torch.count_nonzero(g))
                for g in gradients[start:start + count]) for task, gradients in zip(TASKS, task_grads)},
            "gram_matrix": gram.tolist(), "joint_gradient_l2": joint_norm,
            "gradient_sum_error_l2": math.sqrt(error_squared),
            "gradient_sum_relative_error": math.sqrt(error_squared) / max(joint_norm, 1e-12),
            "auxiliary_gradient_l2": aux_norm, "auxiliary_to_pair_norm_ratio": aux_norm / pair_norm if pair_norm else None,
            "pair_vs_task_cosine": {task: cosine(float(gram[3, i]), pair_norm, float(norms[i])) for i, task in enumerate(TASKS)},
            "pair_vs_auxiliary_cosine": cosine(pair_dot_aux, pair_norm, aux_norm),
            "pair_vs_joint_cosine": cosine(pair_dot_joint, pair_norm, joint_norm),
            "pair_dot_joint": pair_dot_joint,
            "cold_adamw_direction_l2": float(direction.norm()),
            "cold_adamw_linearized_task_changes": dict(zip(TASKS, linear.tolist())),
        }
        start += count
        del matrix, joint, gram, norms, auxiliary, direction, linear
    error = math.sqrt(sum_error_squared) / max(total_norm, 1e-12)
    # FP32 backward kernels may accumulate shared paths in different orders for
    # joint vs separate backward passes. Record error, with an explicit 0.1%
    # numerical guard; do not mistake a tiny mismatch for missing loss wiring.
    if error > GRADIENT_SUM_FP32_RTOL:
        raise ValueError(f"Per-task gradients do not reconstruct total gradient: {error}")
    if any(p.grad is not None for p in parameters):
        raise ValueError("Diagnostic unexpectedly populated parameter .grad")
    return {"groups": result, "total_gradient_l2": total_norm, "global_clip_coefficient": clip,
        "gradient_sum_relative_error": error, "gradient_sum_fp32_rtol": GRADIENT_SUM_FP32_RTOL,
        "cold_adamw_total_linearized_task_changes": {task: sum(g["cold_adamw_linearized_task_changes"][task]
                                                             for g in result.values()) for task in TASKS},
        "direction_reference": "empty moments, bias-corrected AdamW, eps=1e-8, weight_decay=0, global norm clip=1; not applied"}


def measure(model, batch, config, provenance):
    from .continuation import continuation_losses
    if model.update_recipe != "freeze-role-v1":
        raise ValueError("Fixed audit requires the current frozen-role recipe")
    with temporary_mode(model, "train"), torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        losses, coverage = continuation_losses(model, batch, config,
            atom_threshold=provenance["atom_threshold"], bond_threshold=provenance["bond_threshold"])
        assert float(losses["role"].detach()) == 0
        values = {k: float(v.detach()) for k, v in losses.items()}
        attribution = attribute_losses(losses, model.parameter_groups())
        assert not model.role.training and all(p.grad is None and not p.requires_grad for p in model.role.parameters())
    return {"mode": "train", "seed": SEED, "losses": values, "coverage": coverage, **attribution}


def audit(output):
    from . import continuation, launcher, mode_diagnostic
    from .bundle import load_bundle
    from .checkpoints import file_digest, verify_sources
    from .data import CurriculumCollator, load_sample
    from .gradient_preflight import state_digest
    from chemvector.atom_bond_scene import AtomBondSceneConfig

    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    # Check the defaults used by launcher without constructing an optimizer.
    defaults = inspect.signature(torch.optim.AdamW).parameters
    assert defaults["eps"].default == 1e-8 and defaults["betas"].default == (.9, .999)
    assert file_digest(PRIOR) == PRIOR_SHA
    prior = [json.loads(line) for line in PRIOR.read_text().splitlines()]
    old_plan = next(r for r in prior if r["event"] == "fixed_plan_before_models_or_pixels")
    keys = [old_plan["training_keys"][i] for i in KEY_INDICES]
    assert len(keys) == len(set(keys)) == 4 and WITNESS in keys
    bindings = dict(old_plan["input_sha256"])
    bindings[str(PRIOR)] = PRIOR_SHA
    for path in (Path(__file__), Path(continuation.__file__), Path(launcher.__file__), Path(mode_diagnostic.__file__)):
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
    plan = {"training_keys": keys, "selection": "prior fixed keys at indices 0,1,8,11: first two full, first curriculum, C=N witness",
        "models": ["initial", "last"], "mode": "train", "seed": SEED, "tasks": TASKS,
        "forward_graphs": 8, "gradient_extractions_per_forward": 6, "input_sha256": bindings,
        "gradient_sum_fp32_rtol": GRADIENT_SUM_FP32_RTOL,
        "optimizer_created": False, "optimizer_steps": 0, "gpu": False,
        "cold_adamw_reference": "fresh empty optimizer moments at EACH input; not historical updates or cumulative steps",
        "scope": "local derivatives on four fixed training images, not accuracy, generalization or long-term causality"}
    emit("fixed_plan_before_models_or_pixels", **plan)
    records, states = [], {}
    for name, binding in (("initial", report["selected"]), ("last", report["last"])):
        model, config, payload = load_bundle(binding["path"], expected_sha256=binding["sha256"])
        assert payload["provenance"] == provenance
        before = state_digest(model)
        flags = ([m.training for m in model.modules()], [p.requires_grad for p in model.parameters()])
        collator = CurriculumCollator(AtomBondSceneConfig(
            atom_queries=config.maximum_atoms, bond_queries=config.maximum_bonds), purpose="diagnostic")
        for key in keys:
            row = rows[key]
            sample = load_sample(manifest["sources"][row["source"]], row["sample_key"], purpose="diagnostic")
            batch = collator([sample])
            assert batch["purpose"] == "diagnostic" and not batch["training_authorized"]
            emit("case_started", key=key, model=name)
            result = {"key": key, "model": name, **measure(model, batch, config, provenance)}
            records.append(result)
            emit("gradient_attribution", **result)
            del sample, batch
        assert state_digest(model) == before
        assert flags == ([m.training for m in model.modules()], [p.requires_grad for p in model.parameters()])
        assert all(p.grad is None for p in model.parameters())
        states[name] = before
        del model
    verify_sources(provenance["checkpoints"])
    for binding in (report["selected"], report["last"]):
        assert file_digest(binding["path"]) == binding["sha256"]
    for path, digest in bindings.items():
        assert file_digest(path) == digest
    assert not torch.cuda.is_initialized() and len(records) == plan["forward_graphs"]
    result = {"plan": plan, "records": records, "states_sha256": states,
        "states_modes_and_gradient_flags_unchanged": True, "source_files_unchanged": True,
        "optimizer_created": False, "optimizer_steps": 0, "gpu_used": False,
        "accuracy_recovery_evaluated": False,
        "maximum_gradient_sum_relative_error": max(r["gradient_sum_relative_error"] for r in records)}
    with (Path(output) / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    emit("completed", **{k: v for k, v in result.items() if k not in ("plan", "records")})
    return result


def main():
    if sys.argv[1:] == ["--worker"]:
        audit(OUTPUT)
        return 0
    if sys.argv[1:]:
        raise ValueError("This fixed audit accepts no runtime data, seed or optimizer overrides")
    receipt = run_budgeted([sys.executable, "-m", "chemvector.training.curriculum.gradient_diagnostic", "--worker"],
                           output=OUTPUT, budget=ResourceBudget(300))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
