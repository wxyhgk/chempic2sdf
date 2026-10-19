"""Fixed 20260923 batch: prepare, invoke ONCE, and partition a saved receipt.

The existing custodian, worker and receipt allowlist remain unchanged. Only
candidate manifests and the aggregate evaluation commitment are readable here.
No public benchmark scan, model inference, training or source-label mutation.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from rdkit import rdBase

from chemvector.training.curriculum import admission, private_review

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "outputs/datasets/structure-expansion-batch-12272-20260923"
AUDIT = REPO / "outputs/audits/structure-expansion-batch-12272-20260923/report.json"
REQUEST = REPO / "outputs/audits/structure-expansion-identity-request-20260923"
OUTPUT = REPO / "outputs/audits/structure-expansion-private-identity-20260923/receipt"
PARTITION = REPO / "outputs/audits/structure-expansion-identity-partition-20260923"
AUTHORIZATION = REPO / "docs/model/training/releases/20260923-structure-expansion-identity-authorization.json"
EVALUATION = REPO / "Data/sdf/chemvector_joint_v1/fresh_eval_12k"
COMMITMENT = EVALUATION / "commitment.json"
COUNTS = (1534, 12184)
PROFILES = {"oa00", "oa02", "oa11", "oa13", "oa20", "oa22", "oa31", "oa33"}
AUDIT_SHA = "a126eb8305c3557cec3bbc86c38b7f63f52ded4abc504760e5384bcd40f12be6"
MOLECULES_SHA = "2232f54cec93ec1ff08a05f24e2739e3b0f7046dc605f9d3a6a50d462db52331"
COMMITMENT_SHA = "13a456afb2c475a73fc4dca82df5fe62d4a31277d6b584637d9cd92736024512"
RUNNER_SHA = "af0d6bd2808a6d595289abd28ac28b6b0510bf40ec72529899f9bb7d87b034d0"
WORKER_SHA = "779b01bdf3dd9bd024a6678b540575ae42adaf7c60ce3f82fd24c36e412d360e"
IDENTITY_SHA = "71c84c68773e6ba649dbb815ec2d1470950ecdb1be4559fdef309748f7bc550c"
RULES = ["full_inchikey", "connectivity_prefix", "stereo_isotope_independent_group_key"]
DENIED = ("training_authorized", "model_side_private_sample_access_authorized",
          "images_sdf_or_tar_access_authorized", "source_permission_changes_authorized",
          "evaluation_split_changes_authorized")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    path = Path(path).absolute()
    require(path.resolve() == path and "sealed_test" not in path.parts
            and (not path.is_relative_to(EVALUATION) or path == COMMITMENT),
            "Only nonsymlink candidate inputs or aggregate commitment allowed")
    return path.read_bytes()


def digest(path):
    return sha256(read(path)).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def verify_bindings(bindings):
    for path, expected in bindings.items():
        require(digest(path) == expected, "Fixed candidate-side input changed")


def assemble_candidates(molecules, samples, retained):
    by_id = {row["molecule_id"]: row for row in molecules}
    require(len(by_id) == len(molecules)
            and len({row["canonical_smiles"] for row in molecules}) == len(molecules),
            "Duplicate candidate identity")
    require(len(set(retained)) == len(retained), "Duplicate retained key")
    grouped, seen = defaultdict(list), set()
    for row in samples:
        mid, key, profile = row["molecule_id"], row["sample_key"], row["profile_id"]
        require(mid in by_id and key not in seen and profile in PROFILES
                and re.fullmatch(r"[A-Za-z0-9_-]+", key)
                and key == f"{mid}__{profile}", "Invalid candidate/sample identity")
        seen.add(key)
        grouped[mid].append(key)
    require(seen == set(retained) and set(grouped) == set(by_id),
            "Fixed full-batch membership changed")
    candidates = []
    for mid, keys in sorted(grouped.items()):
        row = by_id[mid]
        require(row.get("split") == "quarantine" and row.get("training_authorized") is False
                and row.get("external_overlap_status") == "unchecked", "Candidate is not quarantined")
        candidates.append(dict(source="curriculum", molecule_id=mid,
            canonical_smiles=row["canonical_smiles"], **admission.identities(row["canonical_smiles"]),
            sample_keys=sorted(keys), image_count=len(keys),
            intended_use="structure_expansion_quarantine_no_split_or_training_authorization"))
    return candidates


def validate_commitment(value):
    unsigned = {key: item for key, item in value.items() if key != "commitment_sha256"}
    canonical = (json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
    require(value.get("schema_version") == "chemvector-joint-commitment-1.0"
            and value.get("dataset") == "fresh_eval_12k"
            and value.get("sealed_access") == "locked_until_checkpoint_gate"
            and value.get("split_counts") == {"fresh_dev": 8000, "sealed_test": 4000}
            and value.get("rdkit_version") == rdBase.rdkitVersion
            and value.get("commitment_sha256") == sha256(canonical).hexdigest() == COMMITMENT_SHA,
            "Locked aggregate commitment mismatch")


def prepare():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only environment required")
    require(REQUEST.resolve() == REQUEST and not REQUEST.exists(), "Fresh request directory required")
    require(digest(AUDIT) == AUDIT_SHA and digest(DATASET / "molecules.json") == MOLECULES_SHA,
            "Fixed batch audit/source mismatch")
    audit = json.loads(read(AUDIT))
    require(audit.get("integrity_passed") is True and audit.get("training_authorized") is False
            and audit.get("external_identity_review") is False
            and (audit.get("retained_structures"), audit.get("retained_images")) == COUNTS
            and audit.get("target_panel_passed") == COUNTS[0]
            and audit.get("target_panel_failures") == [] and audit.get("target_grid_collision_samples") == []
            and Path(audit["dataset"]).absolute() == DATASET, "Completed fixed batch audit required")
    bindings = {str(Path(path).absolute()): value for path, value in audit["input_sha256"].items()}
    bindings.update({str(DATASET / name): value for name, value in audit["manifest_sha256"].items()})
    bindings[str(AUDIT)] = AUDIT_SHA
    for path, expected in ((private_review.__file__, RUNNER_SHA), (private_review.WORKER, WORKER_SHA),
                           (admission.__file__, IDENTITY_SHA)):
        bindings[str(Path(path).absolute())] = expected
    for path in (COMMITMENT, Path(__file__).absolute(), REPO / "tests/data_sdf/test_structure_identity_review.py"):
        bindings[str(path)] = digest(path)
    verify_bindings(bindings)
    commitment = json.loads(read(COMMITMENT))
    validate_commitment(commitment)
    quality = json.loads(read(DATASET / "quality-partition.json"))
    require(quality["conflict_identity_quarantine"] == [] and quality["same_identity_duplicates"] == [],
            "Unexpected internal quality exclusions")
    with rdBase.BlockLogs():
        candidates = assemble_candidates(json.loads(read(DATASET / "molecules.json")),
            json.loads(read(DATASET / "samples.json")), quality["retained"])
    require((len(candidates), sum(row["image_count"] for row in candidates)) == COUNTS, "Fixed count mismatch")
    request = dict(schema_version="chemvector-identity-review-request-0.1", training_authorized=False,
        authorization_status="requires_separate_fixed_authorization_record", external_overlap_status="pending_review",
        candidate_molecules=COUNTS[0], candidate_images=COUNTS[1],
        candidate_molecules_by_source={"curriculum": COUNTS[0]}, candidate_images_by_source={"curriculum": COUNTS[1]},
        candidate_identities_sha256=sha256(encoded(candidates)).hexdigest(),
        evaluation_commitment_sha256=COMMITMENT_SHA, identity_implementation_sha256=IDENTITY_SHA,
        rdkit_version=rdBase.rdkitVersion, requested_checks=RULES, input_sha256=bindings,
        artifact_audit_sha256=AUDIT_SHA, original_molecules_sha256=MOLECULES_SHA,
        payloads_reverified=False, payload_integrity_evidence="hash-bound full artifact audit; not rerun here",
        target_scope="one image per structure; not all styles, predicted candidates or accuracy",
        return_contract=["Counts and candidate-side exclusion IDs only; no evaluation IDs, structures, images or logs.",
                         "One invocation of unchanged custodian/worker; no retries or adaptive replacements."],
        private_identity_review_executed=False, public_review="not_performed",
        training_performed=False, model_forwards=0, gpu_used=False,
        fresh_eval_sample_data_read=False, sealed_test_sample_data_read=False)
    verify_bindings(bindings)
    REQUEST.mkdir(parents=True, exist_ok=False)
    write_new(REQUEST / "candidate-identities.json", candidates)
    write_new(REQUEST / "request.json", request)
    return {"status": "prepared_no_query", "candidate_molecules": COUNTS[0], "candidate_images": COUNTS[1],
            "request_sha256": digest(REQUEST / "request.json"),
            "candidate_identities_sha256": request["candidate_identities_sha256"], "training_authorized": False}


def validate_fixed_request():
    request_bytes, candidate_bytes = read(REQUEST / "request.json"), read(REQUEST / "candidate-identities.json")
    request, candidates = json.loads(request_bytes), json.loads(candidate_bytes)
    require(request.get("schema_version") == "chemvector-identity-review-request-0.1"
            and request.get("training_authorized") is False
            and request.get("requested_checks") == RULES
            and request.get("artifact_audit_sha256") == AUDIT_SHA
            and request.get("original_molecules_sha256") == MOLECULES_SHA
            and request.get("evaluation_commitment_sha256") == COMMITMENT_SHA
            and request.get("identity_implementation_sha256") == IDENTITY_SHA
            and request.get("rdkit_version") == rdBase.rdkitVersion
            and (request.get("candidate_molecules"), request.get("candidate_images")) == COUNTS
            and (len(candidates), sum(row["image_count"] for row in candidates)) == COUNTS
            and request.get("candidate_identities_sha256") == sha256(candidate_bytes).hexdigest(),
            "Fixed request binding mismatch")
    verify_bindings(request["input_sha256"])
    verify_bindings({private_review.__file__: RUNNER_SHA, private_review.WORKER: WORKER_SHA,
                     admission.__file__: IDENTITY_SHA})
    return request_bytes, candidate_bytes, request, candidates


def invoke_once(*, authorized=False):
    require(authorized is True and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Explicit identity-only authorization and CPU environment required")
    require(all(path.resolve() == path for path in (REQUEST, EVALUATION, OUTPUT, AUTHORIZATION)),
            "Symlinked review path")
    attempt = REQUEST / "private-attempt.json"
    if attempt.exists() or OUTPUT.exists():
        raise FileExistsError("Already reserved/executed; never retry")
    request_bytes, candidate_bytes, request, _ = validate_fixed_request()
    auth = json.loads(read(AUTHORIZATION))
    require(auth.get("schema_version") == "chemvector-fixed-identity-authorization-record-0.1"
            and type(auth.get("maximum_real_private_invocations")) is int
            and auth["maximum_real_private_invocations"] == 1
            and type(auth.get("automatic_retries")) is int and auth["automatic_retries"] == 0
            and all(auth.get(field) is False for field in DENIED)
            and Path(auth["request_directory"]).absolute() == REQUEST
            and Path(auth["output_directory"]).absolute() == OUTPUT
            and Path(auth["evaluation_root"]).absolute() == EVALUATION
            and (auth.get("candidate_molecules"), auth.get("candidate_images")) == COUNTS
            and auth.get("request_sha256") == sha256(request_bytes).hexdigest()
            and auth.get("candidate_identities_sha256") == sha256(candidate_bytes).hexdigest()
            and auth.get("evaluation_commitment_sha256") == COMMITMENT_SHA
            and auth.get("runner_sha256") == RUNNER_SHA and auth.get("worker_sha256") == WORKER_SHA
            and auth.get("wrapper_sha256") == digest(__file__), "Fixed authorization binding mismatch")
    bindings = {**request["input_sha256"], str(AUTHORIZATION): digest(AUTHORIZATION),
                str(REQUEST / "request.json"): auth["request_sha256"],
                str(REQUEST / "candidate-identities.json"): auth["candidate_identities_sha256"]}
    write_new(attempt, dict(schema_version="chemvector-private-review-reservation-0.1",
        request_sha256=auth["request_sha256"], candidate_identities_sha256=auth["candidate_identities_sha256"],
        authorization_sha256=bindings[str(AUTHORIZATION)], reserved_at_utc=datetime.now(timezone.utc).isoformat(),
        maximum_invocations=1, automatic_retries=0, status="reserved_no_replay", training_authorized=False))
    fd = os.open(REQUEST, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    # The ONLY real private invocation. Never echo unchecked output or stderr.
    result = private_review.run(REQUEST, EVALUATION, OUTPUT, authorized=True)
    verify_bindings(bindings)
    return result


def partition_candidates(candidates, receipt):
    require(receipt.get("status") in {"checked", "checked_conservative"}, "Complete saved receipt required")
    by_ref = {(row["source"], row["molecule_id"]): row for row in candidates}
    require(len(by_ref) == len(candidates), "Duplicate candidate reference")
    excluded = {}
    for row in receipt["excluded_candidates"]:
        ref = (row["source"], row["molecule_id"])
        require(ref in by_ref and ref not in excluded and row["rules"]
                and set(row["rules"]) <= set(RULES), "Foreign/duplicate exclusion")
        excluded[ref] = row["rules"]
    remaining = [row for ref, row in by_ref.items() if ref not in excluded]
    rejected = [{**row, "private_overlap_rules": excluded[ref]}
                for ref, row in by_ref.items() if ref in excluded]
    return remaining, rejected


def finalize():
    """Saved allowlisted receipt only; never calls the private worker."""
    require(not PARTITION.exists() and PARTITION.resolve() == PARTITION, "Fresh partition directory required")
    request_bytes, candidate_bytes, _, candidates = validate_fixed_request()
    receipt_path = OUTPUT / "receipt.json"
    receipt = private_review.validate_receipt(json.loads(read(receipt_path)), request_bytes, candidate_bytes, WORKER_SHA)
    execution = json.loads(read(OUTPUT / "execution.json"))
    reservation = json.loads(read(REQUEST / "private-attempt.json"))
    require(execution["receipt_sha256"] == digest(receipt_path)
            and execution["request_sha256"] == reservation["request_sha256"] == sha256(request_bytes).hexdigest()
            and reservation["candidate_identities_sha256"] == sha256(candidate_bytes).hexdigest()
            and reservation["authorization_sha256"] == digest(AUTHORIZATION)
            and execution["runner_sha256"] == RUNNER_SHA and execution["worker_sha256"] == WORKER_SHA
            and execution["source_permissions_changed"] is False
            and execution["private_stderr_saved"] is False and execution["raw_worker_stdout_saved"] is False,
            "Saved execution/reservation chain mismatch")
    remaining, rejected = partition_candidates(candidates, receipt)
    report = dict(schema_version="structure-expansion-private-partition-1", status="partitioned_private_receipt_only",
        candidate_molecules=len(candidates), candidate_images=sum(r["image_count"] for r in candidates),
        remaining_molecules=len(remaining), remaining_images=sum(r["image_count"] for r in remaining),
        excluded_molecules=len(rejected), excluded_images=sum(r["image_count"] for r in rejected),
        candidate_overlap_counts_by_rule=receipt["candidate_overlap_counts_by_rule"],
        receipt_status=receipt["status"], receipt_sha256=digest(receipt_path),
        request_sha256=sha256(request_bytes).hexdigest(), candidate_identities_sha256=sha256(candidate_bytes).hexdigest(),
        authorization_sha256=digest(AUTHORIZATION), reservation_sha256=digest(REQUEST / "private-attempt.json"),
        execution_sha256=digest(OUTPUT / "execution.json"), wrapper_sha256=digest(__file__),
        remaining_candidates_sha256=sha256(encoded(remaining)).hexdigest(),
        excluded_candidates_sha256=sha256(encoded(rejected)).hexdigest(),
        real_private_invocations=1, automatic_retries=0, additional_private_queries_in_partition=0,
        training_authorized=False, training_performed=False, model_forwards=0, gpu_used=False,
        sources_modified=False, public_identity_review="not_performed_for_this_batch",
        remaining_blockers=["Public benchmark identity screening is not inherited from earlier batches.",
            "All-style target compatibility and predicted endpoint/bond reachability remain separate.",
            "Grouped release, sample mixture and new bounded training authorization are still required."],
        exclusion_policy="Whole candidate and all its styles; no replacement, source deletion or private correspondence",
        scope="Fixed fresh_eval_12k identity manifests only; not all historical or public evaluation data")
    require(report["remaining_molecules"] + report["excluded_molecules"] == COUNTS[0]
            and report["remaining_images"] + report["excluded_images"] == COUNTS[1], "Partition count mismatch")
    PARTITION.mkdir(parents=True, exist_ok=False)
    write_new(PARTITION / "remaining-candidates.json", remaining)
    write_new(PARTITION / "excluded-candidates.json", rejected)
    write_new(PARTITION / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "invoke", "partition"))
    parser.add_argument("--authorized-identity-only", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "invoke" and not args.authorized_identity_only:
        parser.error("Explicit fixed-batch identity-only authorization required")
    result = prepare() if args.action == "prepare" else (
        invoke_once(authorized=True) if args.action == "invoke" else finalize())
    allowed = ("status", "failure_code", "candidate_molecules", "candidate_images", "request_sha256",
        "candidate_identities_sha256", "remaining_molecules", "remaining_images", "excluded_molecules",
        "excluded_images", "candidate_overlap_counts_by_rule", "unresolved_molecules_by_split", "training_authorized")
    print(json.dumps({key: result[key] for key in allowed if key in result}, sort_keys=True), flush=True)
    return int(result["status"] in {"failed", "incomplete"})


if __name__ == "__main__":
    raise SystemExit(main())
