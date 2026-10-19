"""Public-only identity audit of the unchanged v0.2 request.

The original request did not bind a public cache. Freeze that additional public
input in a separate plan, never rewrite/re-hash the private request or rerun its
review. Reuse the unchanged public scanner and conservative inventory rules.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import time

from rdkit import rdBase

from chemvector.training.curriculum import public_review, private_review
from tools import review_curriculum_v02_private_identity as fixed

BENCHMARK = fixed.REPO / "Data/benchmarks/public_ocsr_v1"
CACHE = fixed.REPO / "Data/sdf/chemvector_joint_v1/work/old_overlap_index.json"
OUTPUT = fixed.REPO / "outputs/audits/curriculum-v02-public-identity-20260919"


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value))


def combine(candidates, public_excluded, prefixes, private_result):
    """Whole-identity union; return exact original remaining rows, no repairs."""
    reasons = {ref: list(items) for ref, items in public_excluded.items()}
    if not set(reasons) <= set(candidates):
        raise ValueError("Foreign public exclusion")
    for ref, row in candidates.items():
        if row["connectivity_prefix"] in prefixes:
            reasons.setdefault(ref, []).append({"reason": "legacy_public_cache_positive"})
    for row in private_result["excluded_candidates"]:
        ref = row["source"] + ":" + row["molecule_id"]
        if ref not in candidates:
            raise ValueError("Foreign private candidate exclusion")
        reasons.setdefault(ref, []).append({"reason": "private_receipt_positive"})
    remaining = [row for ref, row in candidates.items() if ref not in reasons]
    rejected = [{**candidates[ref], "exclusion_reasons": reason} for ref, reason in sorted(reasons.items())]
    return remaining, rejected, reasons


def run(request_dir, benchmark, cache_path, private_path, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only public review required")
    request_dir, benchmark, cache_path, private_path, output = (
        Path(path).absolute() for path in (request_dir, benchmark, cache_path, private_path, output))
    if output.exists() or output.resolve() != output:
        raise FileExistsError("Fresh nonsymlink public output required")
    if any(output.is_relative_to(path) for path in (request_dir, benchmark, cache_path.parent, private_path.parent)):
        raise ValueError("Output cannot overwrite source evidence")
    inputs = {}
    def read(path):
        path = Path(path).absolute()
        raw = fixed.read(path)
        inputs[str(path)] = sha256(raw).hexdigest()
        return raw

    request_bytes = read(request_dir / "request.json")
    candidate_bytes = read(request_dir / "candidate-identities.json")
    request, rows = json.loads(request_bytes), json.loads(candidate_bytes)
    if (sha256(request_bytes).hexdigest() != fixed.REQUEST_SHA
            or sha256(candidate_bytes).hexdigest() != fixed.CANDIDATES_SHA
            or request.get("schema_version") != "chemvector-identity-review-request-0.1"
            or request.get("candidate_identities_sha256") != fixed.CANDIDATES_SHA
            or request.get("training_authorized") is not False or request.get("rdkit_version") != rdBase.rdkitVersion
            or request.get("identity_implementation_sha256") != fixed.digest(private_review.IDENTITIES)
            or (len(rows), sum(row["image_count"] for row in rows)) != fixed.COUNTS):
        raise ValueError("Fixed public candidate request mismatch")
    fixed.verify_bindings(request["input_sha256"])
    inputs.update(request["input_sha256"])
    private_bytes = read(private_path)
    private_result = private_review.validate_receipt(json.loads(private_bytes), request_bytes, candidate_bytes,
                                                      fixed.digest(private_review.WORKER))
    if private_result["status"] not in {"checked", "checked_conservative"}:
        raise ValueError("Complete saved private receipt required; do not query here")
    cache_bytes = read(cache_path)
    cache = json.loads(cache_bytes)
    if cache.get("schema_version") != "chemvector-joint-overlap-index-0.1":
        raise ValueError("Unsupported public cache")
    keys, cache_audit = public_review._cached_keys(cache, "benchmark_inchikeys")
    if not keys:
        raise ValueError("Missing public cache identity keys")
    source_dir = Path(public_review.__file__).parent
    source_names = ("public_review.py", "ctab_inventory.py", "admission.py")
    for path in (Path(__file__), Path(fixed.__file__), Path(private_review.__file__), private_review.WORKER,
                 *(source_dir / name for name in source_names)):
        read(path)
    for subset in public_review.SUBSETS:
        read(benchmark / "manifests" / (subset + ".csv"))
    plan = dict(schema_version="chemvector-v02-public-review-plan-0.1", request_sha256=fixed.REQUEST_SHA,
                candidate_identities_sha256=fixed.CANDIDATES_SHA, candidate_molecules=len(rows),
                candidate_images=sum(row["image_count"] for row in rows), input_sha256=inputs,
                cache_binding="separate_public_sidecar_no_modification_to_original_private_request",
                source_policy="unchanged_public_scanner_full_key_or_prefix_plus_conservative_unresolved_inventory",
                private_identity_queries=0, training_authorized=False, training_performed=False,
                gpu_used=False, model_forwards=0, optimizer_steps=0)
    fixed.verify_bindings(inputs)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", plan)
    start = time.monotonic()
    report = dict(schema_version="chemvector-public-candidate-exclusion-0.1", status="incomplete",
                  complete=False, training_authorized=False, training_performed=False,
                  private_identity_queries=0, gpu_used=False, model_forwards=0, optimizer_steps=0,
                  plan_sha256=fixed.digest(output / "plan.json"))
    try:
        with rdBase.BlockLogs():
            candidates, inventories = public_review.candidate_inventory(rows)
        index, unresolved, evidence, excluded, manifests = public_review.scan_public(benchmark, candidates, inventories)
        if any(manifests[subset] != inputs[str(benchmark / "manifests" / (subset + ".csv"))]
               for subset in public_review.SUBSETS):
            raise ValueError("Public scan manifest changed")
        remaining, rejected, reasons = combine(candidates, excluded, {key[:14] for key in keys}, private_result)
        unresolved_risks = sum(row["excluded_conservatively"] and row["candidate_ref"] not in reasons for row in evidence)
        if len(remaining) + len(rejected) != len(rows) or unresolved_risks:
            raise ValueError("Public exclusion partition or remaining risk mismatch")
        fixed.verify_bindings(inputs)
        documents = {"remaining-candidates.json": remaining, "excluded-candidates.json": rejected,
                     "unresolved-records.json": unresolved, "candidate-element-inventories.json": inventories}
        for name, document in documents.items():
            write(output / name, document)
        for name, records in (("public-records.jsonl", index), ("unresolved-pair-evidence.jsonl", evidence)):
            with (output / name).open("x", encoding="utf-8") as stream:
                for row in records:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        report.update(status="screened_with_conservative_exclusions", complete=True, rdkit_version=rdBase.rdkitVersion,
            request_sha256=fixed.REQUEST_SHA, candidate_identities_sha256=fixed.CANDIDATES_SHA,
            public_manifest_sha256=manifests, cached_index_sha256=sha256(cache_bytes).hexdigest(),
            implementation_sha256={name: inputs[str(source_dir / name)] for name in source_names},
            wrapper_sha256=inputs[str(Path(__file__).absolute())],
            public_source_hashes_verified=len(index), normal_identity_records=len(index) - len(unresolved),
            unresolved_global_identity_records=len(unresolved),
            supplementary_native_keys_generated=sum(bool(row["native_probe"]["key"]) for row in unresolved),
            unsupported_inventory_records=sum(row["unsupported_inventory"] for row in unresolved),
            unresolved_candidate_pairs_checked=len(evidence),
            deficiency_witness_pairs=sum(row["deficiency_witness"] is not None for row in evidence),
            unresolved_risk_pairs=sum(row["excluded_conservatively"] for row in evidence),
            candidate_molecules=len(rows), candidate_images=sum(row["image_count"] for row in rows),
            conservatively_excluded_molecules=len(rejected), conservatively_excluded_images=sum(row["image_count"] for row in rejected),
            remaining_molecules=len(remaining), remaining_images=sum(row["image_count"] for row in remaining),
            remaining_candidates_sha256=fixed.digest(output / "remaining-candidates.json"),
            remaining_molecules_by_source=dict(Counter(row["source"] for row in remaining)),
            remaining_ambiguous_pairs=unresolved_risks, cached_key_audit=cache_audit,
            excluded_candidates_by_reason=dict(Counter(reason for row in rejected for reason in {
                item["reason"] for item in row["exclusion_reasons"]})),
            private_receipt_inheritance={"receipt_sha256": sha256(private_bytes).hexdigest(),
                "status": private_result["status"], "evaluation_commitment_sha256": private_result["evaluation_commitment_sha256"],
                "method": "Unchanged original request and saved allowlisted receipt only; no new private query"},
            policy="Whole candidate exclusion for public key/prefix, cache positive, unresolved possible overlap OR private receipt positive",
            proof_scope="Unchanged public_review scanner and literal non-H lower-bound rules; no source repair or private contents",
            images_read=False, sources_modified=False, cache_rewritten=False, private_sample_data_read=False,
            remaining_blockers=["Identity screening is not a training release, grouped split or source-use approval.",
                                "Predicted endpoints/bond reachability and a new bounded training protocol remain separate.",
                                "Global unresolved public identities and historical evaluation group-isolation issues are not repaired."],
            artifact_sha256={path.name: fixed.digest(path) for path in output.iterdir() if path.is_file()})
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        write(output / "report.json", report)
    return report


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    result = run(fixed.REQUEST, BENCHMARK, CACHE, fixed.OUTPUT / "receipt.json", OUTPUT)
    print(json.dumps({key: result[key] for key in ("status", "candidate_molecules", "candidate_images",
        "remaining_molecules", "remaining_images", "conservatively_excluded_molecules", "training_authorized")}))


if __name__ == "__main__":
    main()
