"""Authorized, isolated data-side identity review; no training authorization.

The privileged custodian only opens fixed manifests read-only (no chmod). It
feeds them into anonymous read-only sandbox mounts. The model-side parent sees
only a strictly allowlisted receipt, never worker stderr or unchecked stdout.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import sys
import sysconfig

SCHEMA = "chemvector-private-identity-receipt-0.1"
RULES = {"full_inchikey", "connectivity_prefix", "stereo_isotope_independent_group_key"}
FAILURES = {
    "worker_failed", "sandbox_invalid", "request_invalid", "candidate_binding_failed",
    "candidate_invalid", "identity_implementation_mismatch", "commitment_binding_failed",
    "commitment_invalid", "private_manifest_invalid", "private_sample_binding_failed",
    "private_coverage_mismatch", "private_set_binding_failed", "private_identity_mismatch",
    "private_key_mismatch", "private_scaffold_mismatch",
    "custodian_failed", "custodian_timeout", "source_metadata_changed", "unsafe_receipt",
}
PROJECT = Path(__file__).resolve().parents[3]
WORKER = PROJECT / "Data/sdf/chemvector_joint_v1/identity_review_worker.py"
IDENTITIES = Path(__file__).with_name("admission.py")


def failed(code):
    return {"schema_version": SCHEMA, "status": "failed", "failure_code": code, "training_authorized": False}


def validate_receipt(value, request_bytes, candidate_bytes, worker_sha256):
    """Reject all unexpected fields/strings, even on failure. No echo of input."""
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA or value.get("training_authorized") is not False:
        raise ValueError("Unsafe identity receipt")
    if value.get("status") == "failed":
        if set(value) != {"schema_version", "status", "failure_code", "training_authorized"} or value["failure_code"] not in FAILURES:
            raise ValueError("Unsafe identity receipt")
        return value
    keys = {
        "schema_version", "status", "training_authorized", "request_sha256", "candidate_identities_sha256",
        "evaluation_commitment_sha256", "identity_implementation_sha256", "rdkit_version",
        "candidate_molecules", "candidate_images", "evaluation_samples_by_split", "evaluation_molecules_by_split",
        "unresolved_molecules_by_split", "sample_manifest_hashes_verified", "identity_set_commitments_verified",
        "candidate_overlap_counts_by_rule", "excluded_candidates", "cross_eval_split_overlap_counts_by_rule",
        "scope", "images_or_sdf_read", "worker_sha256",
        "legacy_scaffold_equivalent_representation_differences",
        "legacy_scaffold_differences_resolved_by_requested_group_rule", "committed_group_binding_mismatches",
        "group_comparison_policy",
    }
    request, candidates = json.loads(request_bytes), json.loads(candidate_bytes)
    expected = {
        "request_sha256": sha256(request_bytes).hexdigest(), "candidate_identities_sha256": sha256(candidate_bytes).hexdigest(),
        "evaluation_commitment_sha256": request["evaluation_commitment_sha256"],
        "identity_implementation_sha256": request["identity_implementation_sha256"],
        "rdkit_version": request["rdkit_version"], "worker_sha256": worker_sha256,
        "candidate_molecules": len(candidates), "candidate_images": sum(r["image_count"] for r in candidates),
        "evaluation_samples_by_split": {"fresh_dev": 8000, "sealed_test": 4000},
        "evaluation_molecules_by_split": {"fresh_dev": 2000, "sealed_test": 1000},
        "sample_manifest_hashes_verified": 2, "identity_set_commitments_verified": 3,
        "scope": "manifest_identity_only", "images_or_sdf_read": False,
        "group_comparison_policy": "union_of_smiles_derived_and_committed_legacy_groups",
    }
    if set(value) != keys or any(type(value.get(key)) is not type(expected_value) or value.get(key) != expected_value
                                 for key, expected_value in expected.items()):
        raise ValueError("Unsafe identity receipt")
    if value["images_or_sdf_read"] is not False:
        raise ValueError("Unsafe identity receipt")
    for field in ("legacy_scaffold_equivalent_representation_differences",
                  "legacy_scaffold_differences_resolved_by_requested_group_rule", "committed_group_binding_mismatches"):
        if type(value[field]) is not int or not 0 <= value[field] <= 3000:
            raise ValueError("Unsafe identity receipt")
    unresolved = value["unresolved_molecules_by_split"]
    if (not isinstance(unresolved, dict) or set(unresolved) != {"fresh_dev", "sealed_test"}
            or any(type(n) is not int or not 0 <= n <= expected["evaluation_molecules_by_split"][s] for s, n in unresolved.items())
            or value["status"] != ("incomplete" if sum(unresolved.values()) else (
                "checked_conservative" if value["committed_group_binding_mismatches"] else "checked"))):
        raise ValueError("Unsafe identity receipt")
    allowed = {(row["source"], row["molecule_id"]) for row in candidates}
    seen, counts = set(), Counter({rule: 0 for rule in RULES})
    if not isinstance(value["excluded_candidates"], list):
        raise ValueError("Unsafe identity receipt")
    for row in value["excluded_candidates"]:
        if not isinstance(row, dict) or set(row) != {"source", "molecule_id", "rules"}:
            raise ValueError("Unsafe identity receipt")
        pair = (row["source"], row["molecule_id"])
        rules = row["rules"]
        if pair not in allowed or pair in seen or not isinstance(rules, list) or not rules or not set(rules) <= RULES or len(set(rules)) != len(rules):
            raise ValueError("Unsafe identity receipt")
        seen.add(pair)
        counts.update(rules)
    if (value["candidate_overlap_counts_by_rule"] != dict(counts)
            or any(type(n) is not int for n in value["candidate_overlap_counts_by_rule"].values())):
        raise ValueError("Unsafe identity receipt")
    cross = value["cross_eval_split_overlap_counts_by_rule"]
    if not isinstance(cross, dict) or set(cross) != RULES or any(type(n) is not int or not 0 <= n <= 1000 for n in cross.values()):
        raise ValueError("Unsafe identity receipt")
    return value


def sandbox_command(mounts, runtime_mounts):
    """No project/home/network/host PID namespace or writable host mount."""
    command = ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
               "--clearenv", "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1",
               "--setenv", "CUDA_VISIBLE_DEVICES", "", "--ro-bind", "/usr", "/usr",
               "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for fd, destination in runtime_mounts:
        command.extend(["--ro-bind-fd", str(fd), destination])
    for fd, destination in mounts:
        command.extend(["--ro-bind-data", str(fd), destination])
    return command + ["--remount-ro", "/", "--chdir", "/tmp", "/runtime/bin/python3.13", "-I", "-S", "/code/worker.py"]


def _limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # bwrap copies the small manifests into anonymous files before remounting
    # them read-only. Zero would also prevent these in-memory copies.
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    # Root is only needed to open the locked manifests. Drop back to the
    # invoking user before bwrap creates its user namespace; inherited read-only
    # descriptors remain usable without exposing or chmod-ing the source paths.
    os.setgroups([])
    os.setgid(int(os.environ["SUDO_GID"]))
    os.setuid(int(os.environ["SUDO_UID"]))


def _metadata(fd):
    info = os.fstat(fd)
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def custodian(request_dir, evaluation_root, runtime_base, packages):
    """Only called under the explicit user-authorized, privileged reader role."""
    descriptors, mounts, before = [], [], {}
    try:
        request_dir, evaluation_root = Path(request_dir), Path(evaluation_root)
        # Do not traverse work/ aliases or unlock/change sealed permissions.
        if any(path.is_symlink() for path in (request_dir, evaluation_root, evaluation_root / "sealed_test")):
            return failed("custodian_failed")
        pairs = [
            (request_dir / "request.json", "/inputs/request.json"),
            (request_dir / "candidate-identities.json", "/inputs/candidates.json"),
            (evaluation_root / "commitment.json", "/inputs/commitment.json"),
            (WORKER, "/code/worker.py"), (IDENTITIES, "/code/admission.py"),
        ]
        for split, root in (("fresh_dev", evaluation_root), ("sealed_test", evaluation_root / "sealed_test")):
            pairs.extend((root / name, f"/private/{split}_{name}") for name in ("samples.csv", "molecules.csv"))
        for path, destination in pairs:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NOATIME | os.O_NONBLOCK)
            descriptors.append(fd)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 32 * 1024**2:
                return failed("custodian_failed")
            before[fd] = _metadata(fd)
            mounts.append((fd, destination))
        # Open runtime handles before entering the new user namespace. There is
        # no need to expose/traverse the host's private home directory there.
        runtime_mounts = []
        for path, destination in ((Path(runtime_base) / "bin/python3.13", "/runtime/bin/python3.13"),
                                  (Path(runtime_base) / "lib", "/runtime/lib"), (Path(packages), "/packages")):
            fd = os.open(path, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
            descriptors.append(fd)
            before[fd] = _metadata(fd)
            runtime_mounts.append((fd, destination))
        command = sandbox_command(mounts, runtime_mounts)
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 pass_fds=descriptors, preexec_fn=_limits, start_new_session=True)
        try:
            output, _ = child.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()
            return failed("custodian_timeout")
        if any(_metadata(fd) != before[fd] for fd in descriptors):
            return failed("source_metadata_changed")
        if child.returncode or len(output) > 1024 * 1024:
            return failed("custodian_failed")
        request_bytes = (request_dir / "request.json").read_bytes()
        candidate_bytes = (request_dir / "candidate-identities.json").read_bytes()
        try:
            return validate_receipt(json.loads(output), request_bytes, candidate_bytes, sha256(WORKER.read_bytes()).hexdigest())
        except Exception:
            return failed("unsafe_receipt")
    except BaseException:
        return failed("custodian_failed")
    finally:
        for fd in descriptors:
            os.close(fd)


def run(request_dir, evaluation_root, output, *, authorized=False, runtime_base=None, packages=None):
    if authorized is not True:
        raise ValueError("Explicit identity-only data-side authorization is required")
    request_dir, evaluation_root, output = map(Path, (request_dir, evaluation_root, output))
    if output.exists():
        raise FileExistsError("Use a fresh private-review receipt directory")
    request_bytes = (request_dir / "request.json").read_bytes()
    candidate_bytes = (request_dir / "candidate-identities.json").read_bytes()
    request = json.loads(request_bytes)
    if (request.get("schema_version") != "chemvector-identity-review-request-0.1"
            or request.get("training_authorized") is not False
            or request.get("candidate_identities_sha256") != sha256(candidate_bytes).hexdigest()
            or request.get("identity_implementation_sha256") != sha256(IDENTITIES.read_bytes()).hexdigest()):
        raise ValueError("Review request binding failed")
    worker_hash = sha256(WORKER.read_bytes()).hexdigest()
    command = ["sudo", "-n", "/usr/bin/python3", str(Path(__file__).resolve()), "--custodian",
               "--authorized-identity-only", "--request-dir", str(request_dir.resolve()),
               "--evaluation-root", str(evaluation_root.resolve()),
               "--runtime-base", str(runtime_base or sys.base_prefix),
               "--packages", str(packages or sysconfig.get_paths()["purelib"])]
    # Privileged stderr never enters the transcript. No raw worker log is saved.
    child = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=200)
    try:
        value = validate_receipt(json.loads(child.stdout), request_bytes, candidate_bytes, worker_hash) if child.returncode == 0 else failed("custodian_failed")
    except Exception:
        value = failed("unsafe_receipt")
    output.mkdir(parents=True, exist_ok=False)
    receipt_bytes = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with (output / "receipt.json").open("xb") as stream:
        stream.write(receipt_bytes)
    execution = {
        "schema_version": "chemvector-private-review-execution-0.1",
        "authorization": "User approved independent identity-only review; no model-side private sample access or training.",
        "receipt_sha256": sha256(receipt_bytes).hexdigest(),
        "request_sha256": sha256(request_bytes).hexdigest(),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "worker_sha256": worker_hash,
        "isolation": "bubblewrap new user/pid/net/ipc/uts namespaces, all capabilities dropped, read-only manifest copies",
        "source_permissions_changed": False, "private_stderr_saved": False, "raw_worker_stdout_saved": False,
        "training_authorized": False, "status": value["status"],
        "trust_boundary": "Local user-authorized process isolation, not an independently signed third-party attestation.",
    }
    with (output / "execution.json").open("x") as stream:
        json.dump(execution, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authorized-identity-only", action="store_true")
    parser.add_argument("--custodian", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--runtime-base", type=Path)
    parser.add_argument("--packages", type=Path)
    args = parser.parse_args(argv)
    if args.custodian:
        output_fd = os.dup(1)
        with open(os.devnull, "wb") as null:
            os.dup2(null.fileno(), 1)
            os.dup2(null.fileno(), 2)
        value = custodian(args.request_dir, args.evaluation_root, args.runtime_base, args.packages) if args.authorized_identity_only else failed("custodian_failed")
        with os.fdopen(output_fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return
    if args.output is None:
        parser.error("--output is required")
    result = run(args.request_dir, args.evaluation_root, args.output, authorized=args.authorized_identity_only,
                 runtime_base=args.runtime_base, packages=args.packages)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
