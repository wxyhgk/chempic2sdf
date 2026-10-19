"""Build the closed execution protocol without invoking the candidate constructor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PLAN_COMMIT = "1fcd10c490755041a6afbff0d3ae3a48ca9aeab7"
EXPECTED = {
    "axis_plan.json": "06366e83731f1ca9dee116f995da14dd5a7e64c68f45e98f2a847c0e423bcd89",
    "source_manifest.json": "a06489212869743c414d8f87729c4ff6d3836f7bec3b7c5ef8af908ad7410118",
    "EXPLORATORY_EXCLUSION.md": "48e7918c9f940fc38e924764ea8e60d8d1b987cd638bdc931ee6acbc32b55a7d",
    "case_matrix.json": "f1c0905af213592293ce259cfe00d3398b3e36e4075fbb7b232403c6d8403206",
}
TOOLCHAIN = {
    "/usr/bin/bwrap": {
        "sha256": "d78807229d616606e339c5988392b9e0ab4a6a6998fa51e4590837f426a12fca",
        "version": "bubblewrap 0.6.1",
    },
    "/usr/bin/python3": {
        "sha256": "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86",
        "version": "Python 3.10.12",
    },
    "/usr/bin/strace": {
        "sha256": "38a5c75cb29dd85ddd7780d54f5bf595554d7a1b5c42524b23065f5dc4c4b01d",
        "version": "strace 5.16",
    },
}
IMPLEMENTATION_PATHS = (
    "formal/chemgraph/audit/owner_capacity_confirmation_v0_5/confirmation_worker.py",
    "formal/chemgraph/audit/owner_capacity_confirmation_v0_5/generic_verifier.py",
    "formal/chemgraph/audit/owner_capacity_confirmation_v0_5/isolated_runner.py",
    "formal/chemgraph/audit/owner_capacity_confirmation_v0_5/run_sandbox_probe.py",
    "formal/chemgraph/audit/owner_capacity_confirmation_v0_5/sandbox_probe.py",
)
CONSTRUCTOR_DEPENDENCY_PATHS = (
    "formal/chemgraph/audit/fragment_root_constructor_v0_4/fragment_root.py",
    "formal/chemgraph/audit/fragment_support_partition_v0_4/support_partition.py",
    "formal/chemgraph/audit/owner_candidate_domain_v0_5/build_observation.py",
    "formal/chemgraph/audit/owner_candidate_domain_v0_5/candidate_domain.py",
    "formal/chemgraph/audit/owner_candidate_domain_v0_5/common.py",
    "formal/chemgraph/challenge_v0_4/rasterizer_v0_4.py",
)
TEMPLATE_PATHS = (
    "templates/feasible_witness_template.schema.json",
    "templates/max_carriers_L.feasible.template.json",
    "templates/max_pixel_variables_L.feasible.template.json",
    "templates/max_total_rle_runs_L.feasible.template.json",
)


class ExecutionProtocolError(ValueError):
    """The preregistered execution closure is not exact."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExecutionProtocolError("JSON_DUPLICATE_KEY")
        result[key] = value
    return result


def _load_canonical(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=_strict_object)
    if raw != _canonical_bytes(value):
        raise ExecutionProtocolError(f"NONCANONICAL:{path.name}")
    return value, raw


def _binding(path: Path, relative_to: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": _sha256(raw),
    }


def build_protocol() -> dict[str, Any]:
    plan, plan_raw = _load_canonical(ROOT / "axis_plan.json")
    manifest, manifest_raw = _load_canonical(ROOT / "source_manifest.json")
    matrix, matrix_raw = _load_canonical(ROOT / "case_matrix.json")
    for name, expected in EXPECTED.items():
        if _sha256((ROOT / name).read_bytes()) != expected:
            raise ExecutionProtocolError(f"FROZEN_SHA_MISMATCH:{name}")
    if len(matrix["cases"]) != 8 or matrix["training_authorized"] is not False:
        raise ExecutionProtocolError("CASE_MATRIX_INVALID")
    manifest_keys = {
        (item["axis"], item["side"]): item for item in manifest["records"]
    }
    if len(manifest_keys) != 8:
        raise ExecutionProtocolError("MANIFEST_CARDINALITY_INVALID")
    for case in matrix["cases"]:
        source = manifest_keys.get((case["axis"], case["side"]))
        if source is None or not (
            source["source_path"] == case["source_path"]
            and source["raw_source_sha256"] == case["source_sha256"]
            and source["target_count"] == case["target_count"]
        ):
            raise ExecutionProtocolError(f"CASE_MANIFEST_MISMATCH:{case['case_id']}")
    tool_bindings = []
    for path_text, expected in TOOLCHAIN.items():
        path = Path(path_text)
        actual = _sha256(path.read_bytes())
        if actual != expected["sha256"]:
            raise ExecutionProtocolError(f"TOOL_SHA_MISMATCH:{path_text}")
        tool_bindings.append({"path": path_text, **expected})
    receipt_schema = ROOT / "execution_receipt.schema.json"
    template_bindings = [
        _binding(ROOT / relative, ROOT) for relative in TEMPLATE_PATHS
    ]
    return {
        "case_matrix": {
            "case_count": 8,
            "sha256": _sha256(matrix_raw),
        },
        "conditional_certificate_binding": {
            "certificate_checker_sha256": _sha256((ROOT / "certificate_checker.py").read_bytes()),
            "lean_module_path": "formal/chemgraph/ChemGraph/OwnerCapacityReachabilityV05.lean",
            "lean_module_sha256": _sha256(
                (REPOSITORY_ROOT / "formal/chemgraph/ChemGraph/OwnerCapacityReachabilityV05.lean").read_bytes()
            ),
            "premises_recomputed": False,
        },
        "constructor_dependency_closure": [
            _binding(REPOSITORY_ROOT / relative, REPOSITORY_ROOT)
            for relative in CONSTRUCTOR_DEPENDENCY_PATHS
        ],
        "execution_profile": {
            "allowed_case_retries": 0,
            "argv_policy": "derived_only_from_closed_case_matrix",
            "environment": {
                "HOME": "/nonexistent",
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin",
                "TZ": "UTC",
            },
            "filesystem": {
                "app_bundle": "read_only_git_archive",
                "input": "single_read_only_committed_observation",
                "output": "single_fresh_0700_directory",
                "persistent_writable_paths": ["/out"],
            },
            "namespaces": ["user", "pid", "net", "ipc", "uts", "cgroup"],
            "output_policy": "candidate.json_or_rejection.json_exactly_as_case_matrix",
            "resource_limits": {
                "address_space_bytes": 2147483648,
                "core_bytes": 0,
                "cpu_seconds": 120,
                "file_bytes": 20971520,
                "open_files": 64,
                "processes": 1,
                "supervisor_trace_file_bytes": 268435456,
                "wall_seconds": 180,
            },
            "trace_policy": {
                "network_attempt_after_python_exec": "no_go",
                "post_python_fork_or_exec_attempt": "no_go",
                "trace_incomplete_or_unparseable": "no_go",
                "trace_scope": "all_syscalls_full_process_tree",
            },
        },
        "feasible_witness_templates": template_bindings,
        "implementation_bindings": [
            _binding(REPOSITORY_ROOT / relative, REPOSITORY_ROOT)
            for relative in IMPLEMENTATION_PATHS
        ],
        "isolation_claim_boundaries": {
            "cgroup_resource_controller_available": False,
            "geometry_renderer_binding": False,
            "host_stdlib_tree_fully_bound": False,
            "implementation_correctness_proved": False,
            "semantic_claim_authorized": False,
        },
        "permanent_aggregate": {
            "all_12_axes_confirmed": False,
            "authoritative": False,
            "conditional_or_unproven_axis_count": 8,
            "generic_capacity_complete": False,
            "premises_recomputed": False,
            "training_authorized": False,
            "universal_byte_bound_verified": False,
            "verdict": "PARTIAL/NO_GO_GENERIC",
        },
        "plan_binding": {
            "axis_plan_bytes": len(plan_raw),
            "axis_plan_sha256": _sha256(plan_raw),
            "constructor_sha256": plan["constructor"]["sha256"],
            "exclusion_sha256": EXPECTED["EXPLORATORY_EXCLUSION.md"],
            "plan_commit": PLAN_COMMIT,
            "source_manifest_bytes": len(manifest_raw),
            "source_manifest_sha256": _sha256(manifest_raw),
        },
        "receipt_schema": _binding(receipt_schema, ROOT),
        "schema_version": "owner-capacity-execution-protocol-v0.5",
        "status": "executor_preregistration_frozen_pending_probe",
        "toolchain": tool_bindings,
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = _canonical_bytes(build_protocol())
    args.output.write_bytes(raw)


if __name__ == "__main__":
    main()
