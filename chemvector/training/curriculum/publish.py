"""Two-phase local publication: fixed plan, then explicit exact-plan review.

Never edits quarantine originals or evaluation payloads. New independent copies
change only lifecycle flags and metadata hashes, not chemistry, pixels or labels.
No model loading, training, deployment, downloads or private queries.
"""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re

from .release_evidence import build_plan, digest, encoded, read
from .release import CHECKS, validate_manifest, verify_release

SUFFIXES = {".meta.json", ".graph.json", ".layers.json", ".text-records.json", ".png", ".svg", ".sdf",
            ".text.png", ".bond.png", ".semantic.png", ".auxiliary.png"}
FLAGS = {"split", "training_authorized", "external_overlap_status"}


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else encoded(value))


def transition(payloads, row, *, plan_sha256, original_root):
    """Pure transformation; return independently owned bytes for publication."""
    if set(payloads) != SUFFIXES or row["split"] not in {"train", "validation"}:
        raise ValueError("Incomplete same-render bundle or invalid planned split")
    result = dict(payloads)
    meta = json.loads(payloads[".meta.json"])
    if (sha256(payloads[".meta.json"]).hexdigest() != row["meta_sha256"]
            or meta["sample_key"] != row["sample_key"] or meta["molecule_id"] != row["molecule_id"]):
        raise ValueError("Planned metadata/identity mismatch")
    for suffix, value in payloads.items():
        if suffix != ".meta.json" and sha256(value).hexdigest() != meta["sha256"].get(suffix):
            raise ValueError("Original artifact metadata hash mismatch")
    for suffix in (".graph.json", ".layers.json", ".meta.json"):
        value = json.loads(payloads[suffix])
        # Legacy chemical graphs have no split/status fields; the two owning
        # records must still explicitly say quarantine. Never allow a missing
        # training_authorized flag, or an explicitly different graph split.
        split_ok = value.get("split") == "quarantine" or (suffix == ".graph.json" and "split" not in value)
        if not split_ok or value.get("training_authorized") is not False:
            raise ValueError("Only original quarantined data may be published by this path")
        if suffix == ".graph.json" and value["canonical_smiles"] != row["canonical_smiles"]:
            raise ValueError("Plan/source graph identity mismatch")
        value.update(split=row["split"], training_authorized=row["split"] == "train", external_overlap_status="checked")
        if suffix == ".meta.json":
            for other, data in result.items():
                if other != ".meta.json":
                    value["sha256"][other] = sha256(data).hexdigest()
            value["release_provenance"] = {"promotion_plan_sha256": plan_sha256,
                "original_root": original_root, "original_meta_sha256": row["meta_sha256"],
                "changed": "lifecycle flags and dependent metadata hashes only"}
        result[suffix] = encoded(value)
    return result


def publish(spec_path, plan_path, review_path, output):
    spec, plan = json.loads(read(spec_path)), json.loads(read(plan_path))
    review_bytes = read(review_path)
    review = json.loads(review_bytes)
    plan_hash = digest(plan_path)
    if (review.get("schema_version") != "chemvector-promotion-review-0.1"
            or review.get("decision") != "approve_local_research"
            or review.get("approved_plan_sha256") != plan_hash
            or not review.get("reviewer") or not review.get("rationale")
            or review.get("release_checks") != {k: "checked" for k in CHECKS}
            or review.get("source_use_review_sha256") != spec["inputs"]["source_rights"]["sha256"]):
        raise ValueError("Publication needs explicit review of this exact plan and source-use scope")
    if plan != build_plan(spec):
        raise ValueError("Publication inputs or plan changed after review")
    output = Path(output)
    if not output.is_absolute() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Publication needs an explicit non-symlinked output")
    for source_root in plan["manifest_template"]["sources"].values():
        if output.is_relative_to(source_root) or Path(source_root).is_relative_to(output):
            raise ValueError("Publication output must be independent of every original source")
    output.mkdir(parents=True, exist_ok=False)
    # Only the final admission file marks completion. Failure leaves a new,
    # diagnosable partial directory, never an admitted manifest or old-file edit.
    write(output / "promotion-plan.json", plan)
    write(output / "promotion-review.json", review_bytes)
    write(output / "source-use-review.json", read(spec["inputs"]["source_rights"]["path"]))
    manifest = deepcopy(plan["manifest_template"])
    originals = {domain: Path(spec["roots"][domain]) for domain in ("curriculum", "full_molecule")}
    for domain, bindings in plan.get("original_source_manifests", {}).items():
        evidence_dir = output / "source-evidence" / domain
        evidence_dir.mkdir(parents=True)
        for name, expected_hash in bindings.items():
            if name not in {"summary.json", "molecules.json", "samples.jsonl", "sample-sha256.json"}:
                raise ValueError("Unexpected source manifest snapshot")
            payload = read(originals[domain] / name)
            if sha256(payload).hexdigest() != expected_hash:
                raise ValueError("Source manifest changed before snapshot")
            write(evidence_dir / name, payload)
        summary = json.loads(read(originals[domain] / "summary.json"))
        if "source_provenance_sha256" in summary:
            payload = read(originals[domain] / "source-provenance.json")
            if sha256(payload).hexdigest() != summary["source_provenance_sha256"]:
                raise ValueError("Original source provenance binding mismatch")
            write(evidence_dir / "source-provenance.json", payload)
    for domain in originals:
        destination = output / "sources" / domain
        (destination / "samples").mkdir(parents=True)
        manifest["sources"][domain] = str(destination)
    from .data import CurriculumCollator, load_sample
    from .validation_truth import expected_identity
    collator = CurriculumCollator(purpose="diagnostic")
    transferred, preserved = 0, 0
    for row in manifest["samples"]:
        if row["source"] not in originals:
            continue
        root, key = originals[row["source"]], row["sample_key"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError("Unsafe publication sample key")
        planned = plan["source_artifacts"][row["source"] + ":" + key]
        expected = {"samples/" + key + suffix for suffix in SUFFIXES}
        if planned.keys() != expected:
            raise ValueError("Unexpected or missing source artifacts")
        payloads = {}
        for suffix in SUFFIXES:
            path = root / "samples" / (key + suffix)
            data = read(path)
            if sha256(data).hexdigest() != planned["samples/" + key + suffix]:
                raise ValueError("Source artifact changed since identity review")
            payloads[suffix] = data
        sample = load_sample(root, key, purpose="diagnostic")
        batch = collator([sample])
        if batch["spatial_cell_collision_pairs"][0] != 0:
            raise ValueError("Source exceeds existing spatial supervision")
        if expected_identity({"target_molblock": payloads[".sdf"].decode()}) != row["canonical_smiles"]:
            raise ValueError("Source SDF and planned identity disagree")
        converted = transition(payloads, row, plan_sha256=plan_hash, original_root=str(root))
        destination = Path(manifest["sources"][row["source"]]) / "samples"
        for suffix, data in converted.items():
            write(destination / (key + suffix), data)
            if suffix not in {".meta.json", ".layers.json", ".graph.json"}:
                if data != payloads[suffix]:
                    raise ValueError("Publication unexpectedly altered original pixels/labels")
                preserved += 1
        row["meta_sha256"] = sha256(converted[".meta.json"]).hexdigest()
        transferred += 1
        if transferred % 250 == 0:
            print(f"Published independent bundles: {transferred}", flush=True)
    validate_manifest(manifest)
    from .launcher import preflight_records
    preflight_records(manifest)  # Original dev + transitioned same-SVG bundles.
    if plan != build_plan(spec):
        raise ValueError("Source manifests/evidence changed during publication")
    write(output / "manifest.json", manifest)
    report = {"schema_version": "chemvector-local-publication-0.1", "counts": plan["counts"],
        "copied_bundles": transferred, "byte_identical_non_lifecycle_artifacts": preserved,
        "approved_plan_sha256": plan_hash, "review_sha256": sha256(review_bytes).hexdigest(),
        "source_files_modified": False, "model_loaded": False, "gpu_used": False,
        "training_performed": False, "redistribution_authorized": False,
        "limitations": plan["limitations"]}
    write(output / "publication-report.json", report)
    admission = {"schema_version": "chemvector-curriculum-admission-0.1", "training_authorized": True,
        "external_overlap_status": "checked", "coverage_gaps": [], "checked_images": len(manifest["samples"]),
        "approved_manifest_sha256": digest(output / "manifest.json"), "release_checks": review["release_checks"],
        "use": "local_research_only", "approved_plan_sha256": plan_hash,
        "review_sha256": sha256(review_bytes).hexdigest(), "source_evidence": spec["inputs"],
        "limitations": plan["limitations"], "training_performed": False}
    write(output / "admission.json", admission)  # Last; never emitted on a failed preflight.
    try:
        verify_release(output / "manifest.json", output / "admission.json")
    except Exception:
        (output / "admission.json").rename(output / "admission.failed.json")
        raise  # Preserve evidence, but leave no launchable admission filename.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args(argv)
    if bool(args.plan) != bool(args.review):
        parser.error("Publication requires both --plan and --review; neither means plan only")
    if args.plan is None:
        plan = build_plan(json.loads(read(args.spec)))
        write(args.output, plan)
        print(json.dumps({"plan_sha256": digest(args.output), "counts": plan["counts"], "training_authorized": False}))
    else:
        print(json.dumps(publish(args.spec, args.plan, args.review, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
