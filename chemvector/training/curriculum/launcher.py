"""One budgeted continuation run on an explicitly approved, hash-bound release.

No dataset discovery, implicit quarantine override, automatic resume or search.
The public CLI always runs its worker under the external wall-clock supervisor.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

from .release import verify_release, RecordLoader, training_order
from .runtime import ResourceBudget, StepLimit, run_budgeted
from .protocol import require_run_protocol, selection_records, update_recipe


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def preflight_records(manifest):
    loader = RecordLoader(manifest)
    for index, row in enumerate(manifest["samples"], 1):
        loader(manifest, row)
        if index % 500 == 0:
            print(f"Released input verification: {index}/{len(manifest['samples'])}", flush=True)
    return loader


def run_training(manifest_path, admission_path, output, *, max_steps, batch_size=1,
                 validate_every=100, patience=3, device="cpu", expected_bindings=None,
                 component_loader=None, predictor_factory=None):
    manifest, admission, bindings = verify_release(manifest_path, admission_path)
    require_run_protocol(manifest, max_steps=max_steps, batch_size=batch_size,
                         validate_every=validate_every, patience=patience)
    if expected_bindings is not None and expected_bindings != bindings:
        raise ValueError("Release changed between parent preflight and worker startup")
    if any(type(v) is not int or v < 1 for v in (max_steps, batch_size, validate_every, patience)):
        raise ValueError("Training step/batch/validation limits must be positive integers")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Unsupported training device")
    output = Path(output)
    # Output is created by the supervisor. Existing run artifacts are rejected.
    if not output.is_dir() or any((output / name).exists() for name in ("run.json", "initial.pt", "report.json")):
        raise ValueError("Worker requires a fresh supervised run directory")
    record_loader = preflight_records(manifest)  # Complete checks before model/CUDA setup.
    import torch
    from chemvector.atom_bond_scene import AtomBondSceneConfig
    from .bundle import save_bundle
    from .checkpoints import load_current_components, verify_sources
    from .continuation import train_step
    from .data import CurriculumCollator
    from .evaluation import FrozenOCRPredictor, evaluate, summarize_predictions

    torch.set_num_threads(2)
    torch.manual_seed(manifest["seed"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Explicit GPU request cannot run: CUDA is unavailable")
    device = torch.device(device)
    model, proposal_config, provenance = (component_loader or load_current_components)()
    if "initial_provenance" in manifest and manifest["initial_provenance"] != provenance:
        raise ValueError("Initial checkpoints or inference settings differ from the fixed release")
    model.configure_update_recipe(update_recipe(manifest))
    model.to(device)
    predictor = (predictor_factory or FrozenOCRPredictor)(provenance, device)
    optimizer = torch.optim.AdamW(model.parameter_groups(), weight_decay=0)
    from .gradient_preflight import state_digest
    frozen_states = {"role": state_digest(model.role)} if model.update_recipe == "freeze-role-v1" else {}
    def check_frozen_states():
        if any(state_digest(getattr(model, name)) != expected for name, expected in frozen_states.items()):
            raise ValueError("Frozen module state changed; no checkpoint promotion is allowed")
    scope = AtomBondSceneConfig(atom_queries=proposal_config.maximum_atoms, bond_queries=proposal_config.maximum_bonds)
    train_collator = CurriculumCollator(scope, purpose="training")
    recipe = {"max_steps": max_steps, "batch_size": batch_size, "validate_every": validate_every,
              "patience": patience, "device": str(device), "train_cycle": manifest["train_cycle"],
              "seed": manifest["seed"], "resume_supported": False,
              "update_recipe": model.update_recipe, "frozen_initial_states": frozen_states,
              "checkpoint_selection": "strict SDF identity exact count, then valid count; ties keep earlier weights",
              "selection_domains": (manifest.get("training_protocol") or {}).get("selection_domains", "all_validation")}
    write_json(output / "run.json", {"bindings": bindings, "recipe": recipe, "provenance": provenance})
    started = time.monotonic()
    baseline = evaluate(model, proposal_config, manifest, record_loader, predictor, device=device)
    baseline_metrics = summarize_predictions(baseline, baseline)
    baseline_selection = selection_records(manifest, baseline)
    baseline_selection_metrics = summarize_predictions(baseline_selection, baseline_selection)
    write_json(output / "validation-000000.json", {"metrics": baseline_metrics,
        "selection_metrics": baseline_selection_metrics, "predictions": baseline})
    selected = save_bundle(output / "initial.pt", model, proposal_config, provenance=provenance,
                           progress={"optimizer_steps": 0, "bindings": bindings, "recipe": recipe})
    best_score = (baseline_selection_metrics["exact"], baseline_selection_metrics["valid"])
    selected_metrics = baseline_metrics
    selected_selection_metrics = baseline_selection_metrics
    limit, stale = StepLimit(max_steps), 0
    order = training_order(manifest)
    domain_seen = Counter()
    reason = "max_steps"
    with (output / "steps.jsonl").open("x", encoding="utf-8") as log:
        while limit.consumed < max_steps:
            rows = [next(order) for _ in range(batch_size)]
            samples = [record_loader(manifest, row) for row in rows]
            batch = train_collator(samples)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            step = train_step(model, batch, proposal_config, optimizer, admission=admission, step_limit=limit,
                              atom_threshold=provenance["atom_threshold"], bond_threshold=provenance["bond_threshold"])
            domain_seen.update(r["domain"] for r in rows)
            step.update(keys=[r["source"] + ":" + r["sample_key"] for r in rows], domain_seen=dict(domain_seen))
            log.write(json.dumps(step, allow_nan=False) + "\n")
            log.flush()
            if limit.consumed % validate_every and limit.consumed != max_steps:
                continue
            current = evaluate(model, proposal_config, manifest, record_loader, predictor, device=device)
            check_frozen_states()
            metrics = summarize_predictions(current, baseline)
            selection_metrics = summarize_predictions(selection_records(manifest, current), baseline_selection)
            write_json(output / f"validation-{limit.consumed:06d}.json", {
                "metrics": metrics, "selection_metrics": selection_metrics, "predictions": current})
            score = (selection_metrics["exact"], selection_metrics["valid"])
            if score > best_score:
                selected = save_bundle(output / f"step-{limit.consumed:06d}.pt", model, proposal_config,
                    provenance=provenance, progress={"optimizer_steps": limit.consumed, "bindings": bindings, "recipe": recipe})
                selected_metrics, best_score, stale = metrics, score, 0
                selected_selection_metrics = selection_metrics
            else:
                stale += 1
            print(json.dumps({"step": limit.consumed, "validation": metrics,
                "selection_metrics": selection_metrics, "stale_evaluations": stale}), flush=True)
            if stale >= patience:
                reason = "validation_patience"
                break
    # Keep last separately for debugging; never silently deploy or replace best.
    check_frozen_states()
    last = save_bundle(output / "last.pt", model, proposal_config, provenance=provenance,
        progress={"optimizer_steps": limit.consumed, "bindings": bindings, "recipe": recipe})
    verify_sources(provenance["checkpoints"])
    report = {"schema_version": "chemvector-continuation-run-0.1", "stop_reason": reason,
              "optimizer_steps": limit.consumed, "domain_seen": dict(domain_seen),
              "baseline": baseline_metrics, "selected_metrics": selected_metrics,
              "baseline_selection": baseline_selection_metrics, "selected_selection_metrics": selected_selection_metrics,
              "selected": selected, "last": last, "bindings": bindings,
              "training_performed": True, "gpu_used": device.type == "cuda",
              "deployed": False, "elapsed_loop_seconds": time.monotonic() - started}
    report.update(update_recipe=model.update_recipe, frozen_initial_states=frozen_states,
                  frozen_final_states={name: state_digest(getattr(model, name)) for name in frozen_states})
    write_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--gpu-device", help="One physical GPU index; omitted means CPU")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--bindings", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    manifest, admission, bindings = verify_release(args.manifest, args.admission)
    if args.check_only:
        preflight_records(manifest)
        print(json.dumps({"release_checks_passed": True, "images": len(manifest["samples"]), **bindings}))
        return 0
    if args.output is None or args.max_steps is None or args.wall_seconds is None:
        parser.error("Training requires explicit --output, --max-steps and --wall-seconds")
    if any(value < 1 for value in (args.max_steps, args.batch_size, args.validate_every, args.patience)):
        parser.error("Training limits must be positive")
    budget = ResourceBudget(args.wall_seconds, () if args.gpu_device is None else (args.gpu_device,))
    require_run_protocol(manifest, max_steps=args.max_steps, batch_size=args.batch_size,
        validate_every=args.validate_every, patience=args.patience, wall_seconds=args.wall_seconds)
    if args.worker:
        run_training(args.manifest, args.admission, args.output, max_steps=args.max_steps,
            batch_size=args.batch_size, validate_every=args.validate_every, patience=args.patience,
            device="cpu" if args.gpu_device is None else "cuda", expected_bindings=json.loads(args.bindings))
        return 0
    command = [sys.executable, "-m", "chemvector.training.curriculum.launcher", "--worker",
        "--manifest", str(args.manifest), "--admission", str(args.admission), "--output", str(args.output),
        "--max-steps", str(args.max_steps), "--batch-size", str(args.batch_size),
        "--validate-every", str(args.validate_every), "--patience", str(args.patience),
        "--wall-seconds", str(args.wall_seconds), "--bindings", json.dumps(bindings)]
    if args.gpu_device is not None:
        command.extend(("--gpu-device", args.gpu_device))
    receipt = run_budgeted(command, output=args.output, budget=budget, training=True, admission=admission)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
