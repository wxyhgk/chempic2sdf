"""CPU actual-checkpoint gradient audit: no optimizer, weight save or training."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from .runtime import ResourceBudget, run_budgeted


def state_digest(model):
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def audit(data_root, sample_keys, output, *, update_recipe="legacy-all-v1"):
    import torch
    from .checkpoints import load_current_components, verify_sources, file_digest
    from .continuation import continuation_losses, gradient_report
    from .data import CurriculumCollator, load_sample

    torch.set_num_threads(2)
    torch.manual_seed(20260914)
    if not 1 <= len(sample_keys) <= 4 or len(set(sample_keys)) != len(sample_keys):
        raise ValueError("CPU gradient diagnostic requires one to four distinct explicit sample keys")
    print("Loading existing self-developed checkpoints on CPU", flush=True)
    model, config, provenance = load_current_components()
    model.configure_update_recipe(update_recipe)
    before = state_digest(model)
    role_before = state_digest(model.role)
    rows = []
    for key in sample_keys:
        print(f"CPU forward/backward: {key}", flush=True)
        sample = load_sample(data_root, key, purpose="diagnostic")
        batch = CurriculumCollator(purpose="diagnostic")([sample])
        model.zero_grad(set_to_none=True)
        losses, coverage = continuation_losses(model, batch, config,
            atom_threshold=provenance["atom_threshold"], bond_threshold=provenance["bond_threshold"])
        losses["loss"].backward()
        gradients = gradient_report(model)
        rows.append({"sample_key": key, "losses": {k: float(v.detach()) for k, v in losses.items()},
                     "coverage": coverage, "gradients": gradients,
                     "sample_meta_sha256": file_digest(Path(data_root) / "samples" / (key + ".meta.json"))})
        print(json.dumps({"sample_key": key, "coverage": coverage,
                          "nonzero_groups": [k for k, v in gradients.items() if v["grad_l2"] > 0]}), flush=True)
        del losses, batch, sample
    after = state_digest(model)
    if before != after:
        raise ValueError("Diagnostic unexpectedly changed model parameters or buffers")
    if update_recipe == "freeze-role-v1" and (
            model.role.training or any(p.requires_grad or p.grad is not None for p in model.role.parameters())):
        raise ValueError("Frozen role diagnostic invariant failed")
    verify_sources(provenance["checkpoints"])
    groups = model.parameter_groups()
    missing = [g["name"] for g in groups if not any(r["gradients"][g["name"]]["grad_l2"] > 0 for r in rows)]
    report = {"schema_version": "chemvector-curriculum-gradient-preflight-0.1", "device": "cpu",
              "mode": "train_forward_backward_without_optimizer",
              "model_loaded": True, "optimizer_created": False, "optimizer_steps": 0,
              "training_performed": False, "training_authorized": False, "gpu_used": False,
              "update_recipe": update_recipe, "role_state_before_sha256": role_before,
              "role_state_after_sha256": state_digest(model.role),
              "model_architecture_changed": False, "state_before_sha256": before, "state_after_sha256": after,
              "checkpoint_provenance": provenance, "samples": rows, "missing_gradient_groups": missing,
              "group_gradient_coverage_passed": not missing,
              "parameter_groups": [{"name": g["name"], "learning_rate_proposal": g["lr"],
                                    "parameters": sum(p.numel() for p in g["params"])} for g in groups],
              "frozen_parameter_names": [n for n, p in model.named_parameters() if not p.requires_grad],
              "limitations": ["Diagnostic losses are not benchmark accuracy or a curriculum efficacy result.",
                 "Discrete top-k, thresholding and matching are not differentiable; reachable rows are reported.",
                 "No OCR transcription loss or new element/charge/stereo heads are added.",
                 "This audit does not release data; GPU numerics/throughput and training efficacy remain unverified."],
              "source_sha256": {str(p): file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    with (Path(output) / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-keys", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=180)
    parser.add_argument("--update-recipe", choices=("legacy-all-v1", "freeze-role-v1"), default="legacy-all-v1")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        report = audit(args.data_root, args.sample_keys, args.output, update_recipe=args.update_recipe)
        return 0 if report["group_gradient_coverage_passed"] else 2
    budget = ResourceBudget(args.wall_seconds)
    command = [sys.executable, "-m", "chemvector.training.curriculum.gradient_preflight", "--worker",
               "--data-root", str(args.data_root), "--sample-keys", *args.sample_keys, "--output", str(args.output),
               "--update-recipe", args.update_recipe]
    receipt = run_budgeted(command, output=args.output, budget=budget)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
