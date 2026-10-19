"""Explicit recovery into a NEW quarantine directory; no training or auto-retry.

Reuse only verified complete bundles from one failed chemistry-gated expansion.
Check vector ownership before expensive rendering and journal each result.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

from .expansion import write_json
from .full_molecule_data import file_hash
from .runtime import ResourceBudget, run_budgeted


def recovery_inputs(previous, prior_runs, wall_seconds, total_wall_seconds):
    """Bind the failed run, original selection and cumulative generation budget."""
    previous = Path(previous).resolve()
    if not prior_runs or len({Path(p).resolve() for p in prior_runs}) != len(prior_runs):
        raise ValueError("Distinct prior runs are required for cumulative budget accounting")
    if previous not in {Path(p).resolve() for p in prior_runs}:
        raise ValueError("Recovery source must be included in prior runtime accounting")
    bindings = {}
    def read(path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or symlinked recovery input")
        data = path.read_bytes()
        bindings[str(path.resolve())] = sha256(data).hexdigest()
        return json.loads(data)
    plan = read(previous / "plan.json")
    chemistry = read(previous / "chemistry-preflight.json")
    entries = read(previous / "dataset/molecules.json")
    profiles = read(previous / "dataset/profiles.json")
    provenance = read(previous / "dataset/source-provenance.json")
    used = 0.0
    for prior in prior_runs:
        prior = Path(prior).resolve()
        receipt, original = read(prior / "runtime.json"), read(prior / "plan.json")
        if (receipt["gpu_devices"] or receipt["training"] or receipt["status"] != "worker_failed"
                or receipt["elapsed_seconds"] <= 0
                or original["selected"] != plan["selected"] or original["profiles"] != plan["profiles"]):
            raise ValueError("Prior run is not a failed CPU run of the same fixed selection")
        used += receipt["elapsed_seconds"]
    if wall_seconds <= 0 or not 0 < total_wall_seconds <= 3600 or used + wall_seconds > total_wall_seconds:
        raise ValueError("Recovery would reset or exceed aggregate generation budget")
    from .expansion_plan import selected_profiles
    if (plan.get("schema_version") != "chemvector-expansion-plan-0.1"
            or plan["training_authorized"] is not False or plan["typography"] != "stereo-aware-v1"
            or profiles != plan["profiles"] or profiles != [asdict(p) for p in selected_profiles()]
            or chemistry["plan_sha256"] != file_hash(previous / "plan.json")
            or provenance["expansion_plan_sha256"] != chemistry["plan_sha256"]
            or provenance["chemistry_preflight_sha256"] != file_hash(previous / "chemistry-preflight.json")
            or chemistry["replacement_molecules_added"] != 0
            or chemistry["supported_molecules"] != len(entries)
            or len(entries) + len(chemistry["failures"]) != len(plan["selected"])
            or plan["planned_images"] != len(plan["selected"]) * len(profiles)):
        raise ValueError("Recovery selection/profile/chemistry binding mismatch")
    selected = {r["molecule_id"]: r for r in plan["selected"]}
    from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
    expected_ids = {merged_records([dict(record_id=r["molecule_id"], name=r["molecule_id"],
                    stage="full_molecule", concepts=[], smiles=r["canonical_isomeric_smiles"])])[0]["molecule_id"]
                    for r in plan["selected"]}
    if (len({e["molecule_id"] for e in entries}) != len(entries)
            or {e["molecule_id"] for e in entries} & {f["molecule_id"] for f in chemistry["failures"]}
            or {e["molecule_id"] for e in entries} | {f["molecule_id"] for f in chemistry["failures"]} != expected_ids):
        raise ValueError("Recovery contains missing, duplicate or replaced molecule identities")
    for entry in entries:
        source = entry["source_provenance"]
        row = source["source_molecule"]
        if (selected.get(row["molecule_id"]) != row or row["split"] != "train"
                or entry["canonical_smiles"] != row["canonical_isomeric_smiles"]
                or source["commitment_sha256"] != provenance["commitment_sha256"]
                or entry["training_authorized"] is not False or entry["split"] != "quarantine"):
            raise ValueError("Recovery entry is not bound to original train-only selection")
    for filename, expected in plan["input_sha256"].items():
        if file_hash(Path(filename)) != expected:
            raise ValueError("Original expansion input changed")
        bindings[filename] = expected
    return plan, chemistry, entries, provenance, bindings, used


def journal_record(stream, row):
    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def run(args):
    from Data.sdf.chemvector_curriculum_v0_1.bounded import bounded_map
    from Data.sdf.chemvector_curriculum_v0_1.build import _render_job, _write, finish_build
    from Data.sdf.text_source_renderer.layers import json_bytes
    from .expansion_plan import selected_profiles
    from .expansion_recovery_checks import SUFFIXES, vector_batch, verified_bundle

    output, previous = args.output, args.previous.resolve()
    plan, chemistry, entries, provenance, bindings, used = recovery_inputs(
        previous, args.prior_runs, args.wall_seconds, args.total_wall_seconds)
    profiles = selected_profiles()
    for name in ("plan.json", "chemistry-preflight.json"):
        _write(output / name, (previous / name).read_bytes())
    recovery = dict(schema_version="chemvector-expansion-recovery-plan-0.1", previous=str(previous),
        input_sha256=bindings, original_plan_sha256=file_hash(output / "plan.json"),
        prior_elapsed_seconds=used, wall_budget_seconds=args.wall_seconds,
        aggregate_generation_budget_seconds=args.total_wall_seconds,
        replacement_molecules_added=0, training_authorized=False, gpu_used=False,
        policy="verify and copy completed bundles byte-for-byte; render only missing eligible jobs",
        implementation_sha256={str(p): file_hash(p) for directory in
            (Path(__file__).parent, Path("Data/sdf/chemvector_curriculum_v0_1"), Path("Data/sdf/text_source_renderer/v2"))
            for p in sorted(directory.glob("*.py"))})
    write_json(output / "recovery-plan.json", recovery)
    jobs = [(entries[i:i + 20], profiles, plan["typography"]) for i in range(0, len(entries), 20)]
    vector_rows = []
    results = bounded_map(vector_batch, jobs, workers=args.workers, timeout_seconds=60)
    try:
        with (output / "vector-journal.jsonl").open("x") as journal:
            for batch in results:
                for row in batch:
                    vector_rows.append(row)
                    journal_record(journal, row)
                print(f"Vector attribution checked: {len(vector_rows)}/{len(entries)}", flush=True)
    finally:
        results.close()
    excluded = {r["molecule_id"] for r in vector_rows if not r["supported"]}
    eligible = [e for e in entries if e["molecule_id"] not in excluded]
    gate = dict(schema_version="chemvector-expansion-vector-preflight-0.1", checked_molecules=len(entries),
        supported_molecules=len(eligible), excluded_molecules=len(excluded), excluded_images=len(excluded) * len(profiles),
        excluded_molecule_ids=sorted(excluded), rows=vector_rows, training_authorized=False,
        chemical_labels_modified=False, replacement_molecules_added=0,
        recovery_plan_sha256=file_hash(output / "recovery-plan.json"))
    write_json(output / "vector-preflight.json", gate)
    if len(eligible) * len(profiles) < plan["minimum_qualified_image_target"]:
        raise ValueError("Vector-supported image upper bound is below target; no silent candidate replacement")
    dataset = output / "dataset"
    dataset.mkdir()
    (dataset / "samples").mkdir()
    provenance = dict(provenance, recovery_plan_sha256=file_hash(output / "recovery-plan.json"),
                      vector_preflight_sha256=file_hash(output / "vector-preflight.json"))
    for name, value in (("molecules.json", eligible), ("profiles.json", [asdict(p) for p in profiles]),
                        ("source-provenance.json", provenance)):
        _write(dataset / name, json_bytes(value))
    old = previous / "dataset"
    if old.is_symlink() or (old / "samples").is_symlink():
        raise ValueError("Symlinked recovery dataset is forbidden")
    expected_keys = {e["molecule_id"] + "__" + p.profile_id for e in entries for p in profiles}
    completed = {p.name.removesuffix(".meta.json") for p in (old / "samples").glob("*.meta.json")}
    if not completed <= expected_keys:
        raise ValueError("Unplanned completed sample in recovery source")
    suffixes = sorted((*SUFFIXES, ".meta.json"), key=len, reverse=True)
    partial_keys = set()
    for path in (old / "samples").iterdir():
        suffix = next((s for s in suffixes if path.name.endswith(s)), None)
        key = path.name.removesuffix(suffix) if suffix else None
        if path.is_symlink() or not path.is_file() or key not in expected_keys:
            raise ValueError("Unexpected or symlinked artifact in recovery source")
        if key not in completed:
            partial_keys.add(key)
    samples, rejected, hashes, missing = [], [], {}, []
    quality = {(r["molecule_id"], q["profile_id"]): q for r in vector_rows if r["supported"]
               for q in r["quality_rejections"]}
    counts = Counter()
    with (output / "sample-journal.jsonl").open("x") as journal:
        def accept(files, meta, status):
            file_hashes = {}
            for suffix, payload in files.items():
                relative = "samples/" + meta["sample_key"] + suffix
                _write(dataset / relative, payload)
                hashes[relative] = sha256(payload).hexdigest()
                file_hashes[suffix] = hashes[relative]
            row = {k: meta[k] for k in ("sample_key", "molecule_id", "profile_id", "stages", "measured")}
            samples.append(row)
            counts[status] += 1
            journal_record(journal, dict(status=status, sample=row, sha256=file_hashes))

        for entry in entries:
            for profile in profiles:
                key = entry["molecule_id"] + "__" + profile.profile_id
                bundle = verified_bundle(old, entry, profile, plan["typography"]) if key in completed else None
                if bundle:
                    counts["verified_old_complete"] += 1
                if entry["molecule_id"] in excluded:
                    counts["vector_excluded"] += 1
                    journal_record(journal, dict(sample_key=key, status="vector_contract_excluded",
                                                had_verified_old_bundle=bundle is not None))
                elif (entry["molecule_id"], profile.profile_id) in quality:
                    if bundle:
                        raise ValueError("Previously accepted sample now fails vector quality gate")
                    failure = quality[(entry["molecule_id"], profile.profile_id)]
                    rejected.append(failure)
                    journal_record(journal, dict(status="vector_quality_rejected", failure=failure))
                elif bundle:
                    accept(*bundle, "reused")
                else:
                    missing.append((entry, profile, plan["typography"]))
            if counts["verified_old_complete"] and counts["verified_old_complete"] // 500 > counts["last_print"]:
                counts["last_print"] = counts["verified_old_complete"] // 500
                print(f"Verified old bundles: {counts['verified_old_complete']}/{len(completed)}", flush=True)
        print(json.dumps(dict(verified_complete=counts["verified_old_complete"], reused=counts["reused"],
            missing_jobs=len(missing), vector_excluded_images=gate["excluded_images"],
            partial_source_bundles_not_reused=len(partial_keys))), flush=True)
        results = bounded_map(_render_job, missing, workers=args.workers, timeout_seconds=60)
        try:
            for number, (files, meta, failure) in enumerate(results, 1):
                if failure:
                    rejected.append(failure)
                    journal_record(journal, dict(status="render_quality_rejected", failure=failure))
                else:
                    accept(files, meta, "newly_rendered")
                if number % 100 == 0 or number == len(missing):
                    print(f"Missing jobs: {number}/{len(missing)}; accepted total={len(samples)}, "
                          f"quality rejected={len(rejected)}", flush=True)
        finally:
            results.close()
    order = {e["molecule_id"] + "__" + p.profile_id: n
             for n, (e, p) in enumerate((e, p) for e in eligible for p in profiles)}
    samples.sort(key=lambda r: order[r["sample_key"]])
    rejected.sort(key=lambda r: order[r["molecule_id"] + "__" + r["profile_id"]])
    for path, expected in {**bindings, **recovery["implementation_sha256"]}.items():
        if file_hash(Path(path)) != expected:
            raise ValueError("Bound recovery input or implementation changed during generation")
    summary = finish_build(dataset, eligible, profiles, samples, rejected, hashes,
        typography=plan["typography"], task_timeout=60, source_provenance=provenance)
    report = dict(schema_version="chemvector-expansion-result-0.1", plan_sha256=file_hash(output / "plan.json"),
        dataset=str(dataset.resolve()), planned_images=plan["planned_images"],
        planned_render_images=summary["planned_images"], chemistry_excluded_images=len(chemistry["failures"]) * len(profiles),
        vector_excluded_images=gate["excluded_images"], render_accepted_images=len(samples),
        render_rejected_images=len(rejected), molecules=len(eligible), reused_images=counts["reused"],
        newly_rendered_images=counts["newly_rendered"], verified_old_complete_images=counts["verified_old_complete"],
        partial_source_bundles_not_reused=len(partial_keys), training_authorized=False, gpu_used=False,
        training_performed=False, qualified_train_images=0, target_compatibility="pending_full_preflight",
        admission="pending_new_candidate_review", summary_sha256=file_hash(dataset / "summary.json"),
        recovery_plan_sha256=file_hash(output / "recovery-plan.json"),
        vector_preflight_sha256=file_hash(output / "vector-preflight.json"),
        sample_journal_sha256=file_hash(output / "sample-journal.jsonl"))
    if report["planned_images"] != (len(samples) + len(rejected) + report["chemistry_excluded_images"] + report["vector_excluded_images"]):
        raise ValueError("Original planned image accounting mismatch")
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--prior-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--total-wall-seconds", type=float, default=3600)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        parser.error("CPU recovery supports one to sixteen workers")
    if args.worker:
        run(args)
        return 0
    recovery_inputs(args.previous, args.prior_runs, args.wall_seconds, args.total_wall_seconds)
    command = [sys.executable, "-m", "chemvector.training.curriculum.expansion_recovery", "--worker",
        "--previous", str(args.previous.resolve()), "--output", str(args.output.resolve()),
        "--workers", str(args.workers), "--wall-seconds", str(args.wall_seconds),
        "--total-wall-seconds", str(args.total_wall_seconds), "--prior-runs",
        *[str(p.resolve()) for p in args.prior_runs]]
    receipt = run_budgeted(command, output=args.output, budget=ResourceBudget(args.wall_seconds))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
