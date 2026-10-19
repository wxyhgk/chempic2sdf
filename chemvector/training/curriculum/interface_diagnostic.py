"""Bounded CPU role/bbox swap diagnostic. No training, threshold search or deployment."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from .interface_plan import ARMS, prediction_index, select_cases
from .runtime import ResourceBudget, run_budgeted


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def tensor_tree_digest(tree):
    """Include names, dtypes and shapes, not just tensor values."""
    digest = sha256()
    def visit(value):
        if isinstance(value, dict):
            for key, child in sorted(value.items()):
                digest.update(json.dumps(key).encode())
                visit(child)
        else:
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    visit(tree)
    return digest.hexdigest()


def structure_trace(structure):
    atoms = structure["atoms"]
    indices = atoms["atom_mask"][0].nonzero().flatten().tolist()
    return {"sha256": tensor_tree_digest(structure), "selected_atoms": [
        {"candidate": i, "geometry": atoms["atom_geometry"][0, i].cpu().tolist(),
         "score": float(atoms["atom_scores"][0, i])} for i in indices],
        "pairs": structure["pairs"].cpu().tolist(),
        "pair_probabilities": structure["output"]["presence_logits"].sigmoid().cpu().tolist(),
        "chemical_classes": structure["chemistry"].cpu().tolist()}


def interface_delta(before, after):
    old_roles = {r["candidate"]: r for r in before["roles"]}
    new_roles = {r["candidate"]: r for r in after["roles"]}
    if old_roles.keys() != new_roles.keys():
        raise ValueError("Text-interface comparison must retain the exact same candidates")
    old_ocr = {r["candidate"]: r for r in before["ocr"]}
    new_ocr = {r["candidate"]: r for r in after["ocr"]}
    common = sorted(old_ocr.keys() & new_ocr.keys())
    return {"opened_candidates": [i for i in old_roles if not old_roles[i]["text"] and new_roles[i]["text"]],
        "closed_candidates": [i for i in old_roles if old_roles[i]["text"] and not new_roles[i]["text"]],
        "common_ocr_candidates": len(common),
        "changed_ocr": [{"candidate": i, "before": old_ocr[i]["text"], "after": new_ocr[i]["text"]}
                        for i in common if old_ocr[i]["text"] != new_ocr[i]["text"]],
        "box_coordinate_absolute_delta_pixels": sum(abs(old_ocr[i][k][j] - new_ocr[i][k][j]) * 768
            for i in common for k in ("center", "half_extent") for j in (0, 1))}


def summarize(cases):
    result = {}
    for group in ("all_diagnostic_cases", *dict.fromkeys(r["stratum"] for r in cases)):
        rows = cases if group == "all_diagnostic_cases" else [r for r in cases if r["stratum"] == group]
        result[group] = {"images": len(rows), "arms": {}}
        for arm in ARMS:
            counts = {k: sum(bool(r["arms"][arm][k]) for r in rows) for k in ("exact", "valid", "fallback")}
            counts["rescued_vs_last"] = sum(r["arms"][arm]["exact"] and not r["arms"]["last"]["exact"] for r in rows)
            counts["broken_vs_last"] = sum(not r["arms"][arm]["exact"] and r["arms"]["last"]["exact"] for r in rows)
            result[group]["arms"][arm] = counts
    return result


def run_cases(plan, manifest, loader, predictor, models, config, saved, output):
    import torch
    from .pixels import image_tensor
    from .validation_truth import canonical_molblock, expected_identity

    rows = {r["source"] + ":" + r["sample_key"]: r for r in manifest["samples"]}
    results = []
    with torch.inference_mode():
        for selection in plan["samples"]:
            key = selection["key"]
            row = rows[key]
            if row["split"] != "validation" or row["domain"] != "full_molecule":
                raise ValueError("Interface audit accepts full-molecule validation only")
            sample = loader(manifest, row)
            if sample.get("purpose") != "diagnostic":
                raise ValueError("Interface audit sample must be diagnostic-only")
            pixels = image_tensor(sample["image"]).unsqueeze(0)
            # Neither sample truth nor annotations enter either inference API.
            structures = {name: predictor.predict_structure(model, config, pixels) for name, model in models.items()}
            structure_records = {name: structure_trace(value) for name, value in structures.items()}
            case = {**selection, "structures": structure_records, "arms": {}, "saved_replay": {}, "deltas_vs_last": {}}
            for arm in ARMS:
                source = "initial" if arm == "initial" else "last"
                role_source = "initial" if arm in ("initial", "last_initial_role", "last_initial_role_bbox") else "last"
                bbox_source = "initial" if arm in ("initial", "last_initial_bbox", "last_initial_role_bbox") else "last"
                trace = {}
                predicted = predictor.from_structure(structures[source], models[role_source].role,
                    models[bbox_source].bbox, pixels, sample["image"], trace=trace)
                if tensor_tree_digest(structures[source]) != structure_records[source]["sha256"]:
                    raise ValueError("Inference changed cached structure predictions")
                canonical = canonical_molblock(predicted["molblock"])
                expected = expected_identity(sample)
                case["arms"][arm] = {**predicted, "trace": trace, "structure_source": source,
                    "structure_sha256": structure_records[source]["sha256"], "role_source": role_source,
                    "bbox_source": bbox_source, "predicted_smiles": canonical, "expected_smiles": expected,
                    "valid": canonical is not None, "exact": canonical is not None and canonical == expected}
                if arm in ("initial", "last"):
                    fields = ("predicted_smiles", "expected_smiles", "valid", "exact", "fallback")
                    differences = {k: {"saved": saved[arm][key][k], "cpu": case["arms"][arm][k]}
                                   for k in fields if saved[arm][key][k] != case["arms"][arm][k]}
                    case["saved_replay"][arm] = {"matches": not differences, "differences": differences}
                elif arm != "last":
                    case["deltas_vs_last"][arm] = interface_delta(case["arms"]["last"]["trace"], trace)
            write_json(Path(output) / (row["sample_key"] + ".json"), case)
            results.append(case)
            print(json.dumps({"completed": len(results), "total": len(plan["samples"]), "key": key,
                "stratum": selection["stratum"], "exact": {k: v["exact"] for k, v in case["arms"].items()},
                "saved_replay_matches": all(v["matches"] for v in case["saved_replay"].values())}), flush=True)
    return results


def audit(run, manifest_path, admission_path, output):
    import torch
    from .bundle import load_bundle
    from .checkpoints import file_digest
    from .evaluation import FrozenOCRPredictor
    from .gradient_preflight import state_digest
    from .release import RecordLoader, verify_release

    torch.set_num_threads(2)
    torch.manual_seed(20260915)
    output, run = Path(output), Path(run).resolve()
    manifest, _, bindings = verify_release(manifest_path, admission_path)
    original_run = json.loads((run / "run.json").read_text())
    report = json.loads((run / "report.json").read_text())
    if bindings != original_run["bindings"] or bindings != report["bindings"]:
        raise ValueError("Run is not bound to the provided release")
    if Path(report["selected"]["path"]).resolve() != run / "initial.pt":
        raise ValueError("This diagnostic requires the failed run with initial selected")
    initial_path, last_path = run / "validation-000000.json", run / f"validation-{report['optimizer_steps']:06d}.json"
    payloads = {"initial": json.loads(initial_path.read_text()), "last": json.loads(last_path.read_text())}
    plan = select_cases(manifest, payloads["initial"], payloads["last"])
    files = (run / "run.json", run / "report.json", initial_path, last_path, Path(manifest_path), Path(admission_path))
    plan["input_sha256"] = {str(p.resolve()): file_digest(p) for p in files}
    plan["bundles"] = {"initial": report["selected"], "last": report["last"]}
    plan["thresholds"] = {k: v for k, v in original_run["provenance"].items() if k.endswith("_threshold")}
    # Commit selection, all arms and input bindings before loading models or pixels.
    write_json(output / "plan.json", plan)
    print("Fixed 24-case CPU plan saved; loading initial/last bundles", flush=True)
    models, configs, states = {}, {}, {}
    for name, binding in plan["bundles"].items():
        if Path(binding["path"]).resolve() != run / (name + ".pt"):
            raise ValueError("Unexpected checkpoint path")
        model, configs[name], payload = load_bundle(binding["path"], expected_sha256=binding["sha256"])
        if payload["provenance"] != original_run["provenance"]:
            raise ValueError("Bundle inference provenance changed")
        expected_steps = 0 if name == "initial" else report["optimizer_steps"]
        if payload["progress"]["optimizer_steps"] != expected_steps or payload["progress"]["bindings"] != bindings:
            raise ValueError("Bundle training progress/bindings changed")
        models[name] = model.eval().requires_grad_(False)
        states[name] = state_digest(model)
    if configs["initial"] != configs["last"]:
        raise ValueError("Proposal configurations changed")
    predictor = FrozenOCRPredictor(original_run["provenance"], "cpu")
    states["ocr"] = state_digest(predictor.ocr)
    cases = run_cases(plan, manifest, RecordLoader(manifest), predictor, models, configs["last"],
                      {name: prediction_index(p) for name, p in payloads.items()}, output)
    after = {name: state_digest(model) for name, model in {**models, "ocr": predictor.ocr}.items()}
    if states != after:
        raise ValueError("Diagnostic changed model parameters or buffers")
    for path, digest in plan["input_sha256"].items():
        if file_digest(path) != digest:
            raise ValueError("Diagnostic input artifact changed")
    for binding in (*plan["bundles"].values(), original_run["provenance"]["checkpoints"]["ocr_frozen_not_loaded"]):
        if file_digest(binding["path"]) != binding["sha256"]:
            raise ValueError("Diagnostic checkpoint file changed")
    replay = all(r["matches"] for c in cases for r in c["saved_replay"].values())
    result = {"schema_version": "chemvector-text-interface-diagnostic-0.1", "plan_sha256": file_digest(output / "plan.json"),
        "device": "cpu", "gpu_used": False, "optimizer_steps": 0, "training_performed": False,
        "checkpoint_selected": False, "deployed": False, "states_before": states, "states_after": after,
        "cached_structure_unchanged": True, "saved_baselines_reproduced": replay,
        "summary": summarize(cases), "case_files": {c["key"]: c["key"].split(":", 1)[1] + ".json" for c in cases},
        "source_sha256": {str(p): file_digest(p) for p in (Path(__file__), Path(__file__).with_name("interface_plan.py"),
            Path(__file__).with_name("evaluation.py"))}, "limitations": plan["limitations"]}
    write_json(output / "report.json", result)
    print(json.dumps({"saved_baselines_reproduced": replay, "summary": result["summary"]}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "manifest", "admission", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=600)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        report = audit(args.run, args.manifest, args.admission, args.output)
        return 0 if report["saved_baselines_reproduced"] else 2
    command = [sys.executable, "-m", "chemvector.training.curriculum.interface_diagnostic", "--worker"]
    for name in ("run", "manifest", "admission", "output"):
        command.extend(("--" + name, str(getattr(args, name).resolve())))
    receipt = run_budgeted(command, output=args.output, budget=ResourceBudget(args.wall_seconds))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
