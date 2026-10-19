"""Fixed 2,000-molecule / six-style CPU expansion into a fresh quarantine directory."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from .runtime import ResourceBudget, run_budgeted


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_public_prefilter(root, inputs):
    """Reuse public-side evidence only; retain its status as a partial prefilter."""
    from .full_molecule_data import file_hash
    from .public_review import KEY, SUBSETS

    root = Path(root)
    values = {}
    for name in ("report.json", "public-records.jsonl", "unresolved-records.json"):
        path = root / name
        data = path.read_bytes()
        inputs[str(path.resolve())] = sha256(data).hexdigest()
        values[name] = [json.loads(line) for line in data.splitlines()] if name.endswith(".jsonl") else json.loads(data)
    report, records, unresolved = (values[n] for n in ("report.json", "public-records.jsonl", "unresolved-records.json"))
    if (report.get("schema_version") != "chemvector-public-candidate-exclusion-0.1"
            or report.get("status") != "screened_with_conservative_exclusions"
            or len(records) != report["public_source_hashes_verified"]
            or len(unresolved) != report["unresolved_global_identity_records"]
            or len({r["sample_id"] for r in records}) != len(records)):
        raise ValueError("Incomplete historical public prefilter")
    for name, expected in report["implementation_sha256"].items():
        if file_hash(Path(__file__).with_name(name)) != expected:
            raise ValueError("Public prefilter implementation changed")
    if any(sum(r["subset"] == subset for r in records) != 10000 for subset in SUBSETS):
        raise ValueError("Public prefilter subset coverage changed")
    by_id = {r["sample_id"]: r for r in records}
    missing = {r["sample_id"] for r in records if not r["inchikey"]}
    if {r["sample_id"] for r in unresolved} != missing:
        raise ValueError("Unresolved public record coverage changed")
    if any(r["inchikey"] is not None and not KEY.fullmatch(r["inchikey"]) for r in records):
        raise ValueError("Invalid cached public key")
    for row in unresolved:
        if row["source_structure_sha256"] != by_id[row["sample_id"]]["source_structure_sha256"]:
            raise ValueError("Unresolved public record binding changed")
    return records, unresolved


def run(args):
    from Data.sdf.chemvector_curriculum_v0_1.build import build
    from .expansion_plan import chemistry_preflight, make_plan, selected_profiles
    from .full_molecule_data import file_hash, prepare_selected, read_train_source
    from .release import verify_release

    output = args.output
    inputs = {}
    for path in (args.manifest, args.admission, args.exclusion_index):
        inputs[str(path.resolve())] = file_hash(path)
    manifest, _, bindings = verify_release(args.manifest, args.admission)
    molecules, samples, commitment, provenance = read_train_source(args.source_root)
    public_records, unresolved = load_public_prefilter(args.public_review, inputs)
    index = json.loads(args.exclusion_index.read_text())
    plan = make_plan(molecules, manifest, index, public_records, unresolved)
    plan.update(input_sha256=inputs, previous_release_bindings=bindings, source=provenance,
        workers=args.workers, per_image_wall_seconds=60,
        source_sha256={str(p): file_hash(p) for p in (Path(__file__), Path(__file__).with_name("expansion_plan.py"),
            Path(__file__).with_name("full_molecule_data.py"))})
    write_json(output / "plan.json", plan)
    print(json.dumps({"fixed_molecules": len(plan["selected"]), "planned_images": plan["planned_images"],
        "prefilter_counts": plan["prefilter_counts"], "prior_train_molecules": plan["selected_already_in_previous_train"]}), flush=True)
    provenance.update(exclusion_index_sha256=file_hash(args.exclusion_index),
                      expansion_plan_sha256=file_hash(output / "plan.json"))
    entries, provenance = prepare_selected(args.source_root, plan["selected"], samples, commitment, provenance, plan["selection"])
    provenance["new_render_profiles"] = "six fixed profiles, two fonts, one layout engine"
    entries, chemistry_failures = chemistry_preflight(entries)
    chemistry_report = {"schema_version": "chemvector-expansion-chemistry-preflight-0.1",
        "plan_sha256": file_hash(output / "plan.json"), "checked_molecules": len(plan["selected"]),
        "supported_molecules": len(entries), "unsupported_molecules": len(chemistry_failures),
        "failures": chemistry_failures, "replacement_molecules_added": 0,
        "planned_images_after_chemistry_gate": len(entries) * len(plan["profiles"]), "training_authorized": False}
    write_json(output / "chemistry-preflight.json", chemistry_report)
    if chemistry_report["planned_images_after_chemistry_gate"] < plan["minimum_qualified_image_target"]:
        raise ValueError("Too few supported images; do not fill with replacement molecules or duplicates")
    provenance["chemistry_preflight_sha256"] = file_hash(output / "chemistry-preflight.json")
    print(json.dumps({"supported_molecules": len(entries), "chemistry_contract_failures": len(chemistry_failures),
        "render_jobs": chemistry_report["planned_images_after_chemistry_gate"]}), flush=True)
    summary = build(output / "dataset", workers=args.workers, entries=entries, selected_profiles=selected_profiles(),
        gallery=False, typography=plan["typography"], task_timeout=60, source_provenance=provenance)
    for path, expected in inputs.items():
        if file_hash(path) != expected:
            raise ValueError("Expansion input changed during generation")
    report = {"schema_version": "chemvector-expansion-result-0.1", "plan_sha256": file_hash(output / "plan.json"),
        "dataset": str((output / "dataset").resolve()), "planned_images": plan["planned_images"],
        "planned_render_images": summary["planned_images"], "chemistry_excluded_images": len(chemistry_failures) * len(plan["profiles"]),
        "render_accepted_images": summary["accepted_images"], "render_rejected_images": summary["rejected_images"],
        "molecules": summary["unique_molecules"], "training_authorized": False, "gpu_used": False,
        "training_performed": False, "qualified_train_images": 0,
        "target_compatibility": "pending_full_preflight", "admission": "pending_new_candidate_review",
        "summary_sha256": file_hash(output / "dataset/summary.json")}
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "manifest", "admission", "exclusion-index", "public-review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--wall-seconds", type=float, default=7200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        parser.error("CPU expansion supports one to sixteen workers")
    if args.worker:
        run(args)
        return 0
    command = [sys.executable, "-m", "chemvector.training.curriculum.expansion", "--worker"]
    for name in ("source-root", "manifest", "admission", "exclusion-index", "public-review", "output", "workers"):
        value = getattr(args, name.replace("-", "_"))
        command.extend(("--" + name, str(value.resolve() if isinstance(value, Path) else value)))
    receipt = run_budgeted(command, output=args.output, budget=ResourceBudget(args.wall_seconds))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
