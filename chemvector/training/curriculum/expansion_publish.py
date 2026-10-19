"""Explicit exact-plan publication of new bundles; never launches training."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from .expansion_release_plan import build_plan, evidence
from .publish import transition, write
from .release import CHECKS, verify_release
from .release_evidence import digest, encoded, read


def publish(spec_path, plan_path, review_path, output):
    spec, plan, review = (json.loads(read(p)) for p in (spec_path, plan_path, review_path))
    plan_hash = digest(plan_path)
    if (review.get("schema_version") != "chemvector-expanded-promotion-review-0.1"
            or review.get("decision") != "approve_local_research_data_release"
            or review.get("approved_plan_sha256") != plan_hash
            or not review.get("reviewer") or not review.get("rationale")
            or review.get("release_checks") != {name: "checked" for name in CHECKS}
            or review.get("source_use_review_sha256") != spec["inputs"]["source_rights"]["sha256"]):
        raise ValueError("Expanded publication requires exact-plan evidence review")
    if plan != build_plan(spec):
        raise ValueError("Expanded plan or evidence changed before publication")
    output = Path(output)
    if not output.is_absolute() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Expanded publication requires an absolute non-symlinked destination")
    for root in plan["manifest_template"]["sources"].values():
        if output.is_relative_to(root) or Path(root).is_relative_to(output):
            raise ValueError("Expanded release must be independent of original sources")
    output.mkdir(parents=True, exist_ok=False)
    for name, path in (("promotion-plan.json", plan_path), ("promotion-review.json", review_path),
                       ("source-use-review.json", spec["inputs"]["source_rights"]["path"]), ("spec.json", spec_path)):
        write(output / name, read(path))
    original = Path(spec["data_root"])
    destination = output / "sources" / spec["new_source"]
    (destination / "samples").mkdir(parents=True)
    snapshot = output / "source-evidence"
    snapshot.mkdir()
    for name, expected in plan["source_manifest_sha256"].items():
        payload = read(original / name)
        if sha256(payload).hexdigest() != expected:
            raise ValueError("Original expanded manifest changed")
        write(snapshot / name, payload)
    manifest = deepcopy(plan["manifest_template"])
    manifest["sources"][spec["new_source"]] = str(destination)
    from .data import CurriculumCollator, load_sample
    from .validation_truth import expected_identity
    collator = CurriculumCollator(purpose="diagnostic")
    copied = 0
    for row in manifest["samples"]:
        if row["source"] != spec["new_source"]:
            continue
        key = row["sample_key"]
        payloads = {s: read(original / "samples" / (key + s)) for s in plan["source_artifacts"][key]}
        if any(sha256(v).hexdigest() != plan["source_artifacts"][key][s] for s, v in payloads.items()):
            raise ValueError("Expanded artifact changed during publication")
        sample = load_sample(original, key, purpose="diagnostic")
        batch = collator([sample])
        if (batch["spatial_cell_collision_pairs"][0] or expected_identity({"target_molblock": payloads[".sdf"].decode()}) != row["canonical_smiles"]):
            raise ValueError("Expanded published sample fails target/identity checks")
        converted = transition(payloads, row, plan_sha256=plan_hash, original_root=str(original))
        for suffix, payload in converted.items():
            write(destination / "samples" / (key + suffix), payload)
        row["meta_sha256"] = sha256(converted[".meta.json"]).hexdigest()
        copied += 1
        if copied % 500 == 0:
            print(f"Published expanded train bundles: {copied}/{plan['added_train_images']}", flush=True)
    from .launcher import preflight_records
    preflight_records(manifest)
    evidence(spec)  # Recheck exact prior release, fixed receipts and source-use binding.
    for name, expected in plan["source_manifest_sha256"].items():
        if digest(original / name) != expected:
            raise ValueError("Expanded source manifests changed during publication")
    write(output / "manifest.json", manifest)
    report = dict(schema_version="chemvector-expanded-publication-0.1", counts=plan["counts"],
        added_train_images=copied, unique_train_images=plan["unique_train_images"],
        inherited_release_bindings=plan["inherited_release_bindings"], promotion_plan_sha256=plan_hash,
        validation_changed=False, source_files_modified=False, training_performed=False, gpu_used=False,
        training_launch_authorized=False, limitations=plan["limitations"])
    write(output / "publication-report.json", report)
    admission = dict(schema_version="chemvector-curriculum-admission-0.1", training_authorized=True,
        external_overlap_status="checked", coverage_gaps=[], checked_images=len(manifest["samples"]),
        approved_manifest_sha256=digest(output / "manifest.json"), release_checks=review["release_checks"],
        use="local_research_only", approved_plan_sha256=plan_hash, review_sha256=digest(review_path),
        source_evidence=spec["inputs"], inherited_release_bindings=plan["inherited_release_bindings"],
        limitations=plan["limitations"], training_performed=False)
    write(output / "admission.json", admission)
    try:
        verify_release(output / "manifest.json", output / "admission.json")
    except Exception:
        (output / "admission.json").rename(output / "admission.failed.json")
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args(argv)
    if bool(args.plan) != bool(args.review):
        parser.error("Publication requires both an exact plan and its review")
    if args.plan:
        result = publish(args.spec.resolve(), args.plan.resolve(), args.review.resolve(), args.output.resolve())
    else:
        result = build_plan(json.loads(read(args.spec.resolve())))
        write(args.output.resolve(), result)
    print(json.dumps({k: result[k] for k in ("counts", "added_train_images", "unique_train_images")}))


if __name__ == "__main__":
    main()
