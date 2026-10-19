"""One explicitly authorized v0.2 identity-only review; never retry or train.

The existing custodian, sandbox, worker and receipt allowlist are unmodified.
Only the durable fixed-batch reservation/authorization wrapper is new.
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

from chemvector.training.curriculum import private_review

REPO = Path(__file__).resolve().parents[1]
REQUEST = REPO / "outputs/audits/curriculum-v02-identity-review-request-20260919"
OUTPUT = REPO / "outputs/audits/curriculum-v02-private-identity-20260919/receipt"
AUTHORIZATION = REPO / "docs/model/training/releases/20260919-curriculum-v02-identity-authorization.json"
EVALUATION = REPO / "Data/sdf/chemvector_joint_v1/fresh_eval_12k"
REQUEST_SHA = "0a74762ff63af549beb52978c783c547e2a31df67ac8656ed7c7dd9e2c5eee58"
CANDIDATES_SHA = "cedcb3700a0bd9082227799a746e74c232f254029c224fd071ba5600cedd389d"
COUNTS = (3468, 20289)
RUNNER_SHA = "af0d6bd2808a6d595289abd28ac28b6b0510bf40ec72529899f9bb7d87b034d0"
WORKER_SHA = "779b01bdf3dd9bd024a6678b540575ae42adaf7c60ce3f82fd24c36e412d360e"


def read(path):
    path = Path(path).absolute()
    if "sealed_test" in path.parts or path.resolve() != path or not path.is_file():
        raise ValueError("Expected explicit nonsealed nonsymlink candidate-side input")
    return path.read_bytes()


def digest(path):
    return sha256(read(path)).hexdigest()


def verify_bindings(bindings):
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError("Fixed candidate-side input changed")


def reserve_once(path, record):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def invoke_once(request_dir, evaluation_root, output, authorization, *, authorized=False):
    if authorized is not True or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Explicit fixed-batch identity-only authorization and CPU environment required")
    request_dir, evaluation_root, output, authorization = (
        Path(path).absolute() for path in (request_dir, evaluation_root, output, authorization))
    if any(path.resolve() != path for path in (request_dir, evaluation_root, output, authorization)):
        raise ValueError("Symlinked review path")
    attempt = request_dir / "private-attempt.json"
    if attempt.exists() or output.exists():
        raise FileExistsError("Already reserved/executed; inspect receipts, never retry")
    auth = json.loads(read(authorization))
    request_path, candidates_path = request_dir / "request.json", request_dir / "candidate-identities.json"
    request_bytes, candidate_bytes = read(request_path), read(candidates_path)
    request, candidates = json.loads(request_bytes), json.loads(candidate_bytes)
    denied = ("training_authorized", "model_side_private_sample_access_authorized",
              "images_sdf_or_tar_access_authorized", "source_permission_changes_authorized",
              "evaluation_split_changes_authorized")
    if (auth.get("schema_version") != "chemvector-fixed-identity-authorization-record-0.1"
            or auth.get("maximum_real_private_invocations") != 1 or auth.get("automatic_retries") != 0
            or any(auth.get(field) is not False for field in denied)
            or Path(auth["request_directory"]).resolve() != request_dir
            or Path(auth["output_directory"]).resolve() != output
            or Path(auth["evaluation_root"]).resolve() != evaluation_root
            or (request.get("candidate_molecules"), request.get("candidate_images")) != COUNTS
            or (auth.get("candidate_molecules"), auth.get("candidate_images")) != COUNTS
            or len(candidates) != COUNTS[0] or sum(row["image_count"] for row in candidates) != COUNTS[1]
            or request.get("schema_version") != "chemvector-identity-review-request-0.1"
            or request.get("training_authorized") is not False
            or sha256(request_bytes).hexdigest() != auth.get("request_sha256") or auth["request_sha256"] != REQUEST_SHA
            or sha256(candidate_bytes).hexdigest() != auth.get("candidate_identities_sha256")
            or auth["candidate_identities_sha256"] != CANDIDATES_SHA
            or request.get("candidate_identities_sha256") != CANDIDATES_SHA
            or request["evaluation_commitment_sha256"] != auth["evaluation_commitment_sha256"]
            or auth.get("runner_sha256") != RUNNER_SHA or digest(private_review.__file__) != RUNNER_SHA
            or auth.get("worker_sha256") != WORKER_SHA or digest(private_review.WORKER) != WORKER_SHA
            or auth.get("wrapper_sha256") != digest(__file__)):
        raise ValueError("Fixed batch/authorization/implementation binding mismatch")
    bindings = {**request["input_sha256"], str(authorization): digest(authorization),
                str(request_path): REQUEST_SHA, str(candidates_path): CANDIDATES_SHA,
                str(Path(__file__).absolute()): auth["wrapper_sha256"],
                str(Path(private_review.__file__).absolute()): RUNNER_SHA,
                str(private_review.WORKER): WORKER_SHA}
    verify_bindings(bindings)
    reserve_once(attempt, dict(schema_version="chemvector-private-review-reservation-0.1",
        request_sha256=REQUEST_SHA, candidate_identities_sha256=CANDIDATES_SHA,
        authorization_sha256=digest(authorization), reserved_at_utc=datetime.now(timezone.utc).isoformat(),
        maximum_invocations=1, automatic_retries=0, status="reserved_no_replay", training_authorized=False))
    # This call alone may open private manifests, inside the unchanged isolated
    # custodian. Never print its raw stdout/stderr or inspect private paths here.
    result = private_review.run(request_dir, evaluation_root, output, authorized=True)
    verify_bindings(bindings)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorized-identity-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.authorized_identity_only:
        parser.error("Explicit identity-only authorization is required")
    result = invoke_once(REQUEST, EVALUATION, OUTPUT, AUTHORIZATION, authorized=True)
    allowed = ("status", "training_authorized", "failure_code", "candidate_molecules", "candidate_images",
               "candidate_overlap_counts_by_rule", "unresolved_molecules_by_split")
    print(json.dumps({key: result[key] for key in allowed if key in result}, sort_keys=True), flush=True)
    return 0 if result["status"] in {"checked", "checked_conservative"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
