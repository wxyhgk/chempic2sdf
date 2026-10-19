from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from .confirmation_worker import _atomic_write
from .isolated_runner import (
    CASE_MATRIX_SHA256,
    EXCLUSION_SHA256,
    MANIFEST_SHA256,
    PLAN_COMMIT,
    PLAN_SHA256,
    REPOSITORY_ROOT,
    IsolatedRunnerError,
    case_record,
    frozen_contract,
    publish_authority_marker,
    sandbox_command,
    source_record,
    validate_closed_outcome,
    validate_trace,
)


ROOT = Path(__file__).resolve().parent
EMPTY_SHA = hashlib.sha256(b"").hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _trace_ok() -> dict[str, object]:
    return {
        "network_attempt_count": 0,
        "output_mutation_count": 3,
        "outside_output_mutation_count": 0,
        "post_python_process_attempt_count": 0,
        "python_exec_count": 1,
        "trace_files": [],
    }


class ExecutionPreregistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.bundle = self.base / "bundle"
        self.output = self.base / "output"
        self.trace = self.base / "trace"
        for path in (self.bundle, self.output, self.trace):
            path.mkdir(mode=0o700)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _install_source(self, case_id: str) -> dict[str, object]:
        case = case_record(case_id)
        destination = self.bundle / (
            Path("formal/chemgraph/audit/owner_capacity_confirmation_v0_5")
            / case["source_path"]
        )
        destination.parent.mkdir(parents=True, mode=0o700)
        shutil.copyfile(ROOT / case["source_path"], destination)
        return case

    def _write_rejection(self, case: dict[str, object]) -> None:
        (self.output / "rejection.json").write_bytes(_canonical({
            "error": case["expected_error"],
            "schema_version": "owner-capacity-worker-rejection-v0.5",
            "status": "rejected",
            "training_authorized": False,
        }))

    def _validate_rejection(self, case: dict[str, object], **changes: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "case": case,
            "output_directory": self.output,
            "exit_code": 20,
            "stdout": b"",
            "stderr": b"",
            "timed_out": False,
            "source_pre_sha256": case["source_sha256"],
            "source_post_sha256": case["source_sha256"],
            "trace_validation": _trace_ok(),
        }
        arguments.update(changes)
        return validate_closed_outcome(**arguments)  # type: ignore[arg-type]

    def test_contract_and_case_matrix_bind_committed_plan(self) -> None:
        plan, manifest = frozen_contract()
        self.assertEqual(PLAN_COMMIT, "1fcd10c490755041a6afbff0d3ae3a48ca9aeab7")
        self.assertEqual(PLAN_SHA256, "06366e83731f1ca9dee116f995da14dd5a7e64c68f45e98f2a847c0e423bcd89")
        self.assertEqual(MANIFEST_SHA256, "a06489212869743c414d8f87729c4ff6d3836f7bec3b7c5ef8af908ad7410118")
        self.assertEqual(EXCLUSION_SHA256, "48e7918c9f940fc38e924764ea8e60d8d1b987cd638bdc931ee6acbc32b55a7d")
        self.assertEqual(CASE_MATRIX_SHA256, hashlib.sha256((ROOT / "case_matrix.json").read_bytes()).hexdigest())
        self.assertEqual(len(plan["axes"]), 12)
        self.assertEqual(manifest["record_count"], 8)
        expected = {
            "max_carriers_L": (0, "candidate", None),
            "max_carriers_H": (20, "rejection", "CAPACITY_EXCEEDED:max_carriers"),
            "max_intersections_L": (20, "rejection", "R0_REQUIRES_TWO_INTERSECTIONS"),
            "max_intersections_H": (20, "rejection", "CAPACITY_EXCEEDED:max_intersections"),
            "max_pixel_variables_L": (0, "candidate", None),
            "max_pixel_variables_H": (20, "rejection", "CAPACITY_EXCEEDED:max_pixel_variables"),
            "max_total_rle_runs_L": (0, "candidate", None),
            "max_total_rle_runs_H": (20, "rejection", "CAPACITY_EXCEEDED:max_total_rle_runs"),
        }
        self.assertEqual(
            {key: tuple(case_record(key)[field] for field in ("expected_exit_code", "expected_kind", "expected_error")) for key in expected},
            expected,
        )

    def test_only_committed_source_pairs_are_allowlisted(self) -> None:
        for axis in ("max_carriers", "max_intersections", "max_pixel_variables", "max_total_rle_runs"):
            for side in ("L", "H"):
                self.assertEqual(source_record(axis, side)["axis"], axis)
        with self.assertRaisesRegex(ValueError, "SOURCE_NOT_ALLOWLISTED"):
            source_record("max_output_bytes", "H")
        with self.assertRaisesRegex(ValueError, "CASE_NOT_ALLOWLISTED"):
            case_record("unregistered")

    def test_profile_is_derived_from_case_and_has_no_live_repo_bind(self) -> None:
        self._install_source("max_carriers_L")
        command = sandbox_command(
            bundle_root=self.bundle,
            output_directory=self.output,
            trace_prefix=self.trace / "syscalls",
            case_id="max_carriers_L",
        )
        joined = " ".join(command)
        for token in (
            "--unshare-user", "--unshare-pid", "--unshare-net", "--unshare-ipc",
            "--unshare-uts", "--unshare-cgroup", "--cap-drop ALL", "--clearenv",
            "--die-with-parent", "--new-session", "--remount-ro /", "trace=all",
            "-I -S -B", "--expected-kind candidate",
        ):
            self.assertIn(token, joined)
        self.assertNotIn("--proc", command)
        self.assertNotIn("--dev", command)
        self.assertEqual(command.count("--bind"), 1)
        self.assertIn("/out/candidate.json", command)
        self.assertNotIn(str(REPOSITORY_ROOT), joined)

    def test_live_repo_bundle_and_symlink_are_rejected(self) -> None:
        with self.assertRaisesRegex(IsolatedRunnerError, "BUNDLE_INSIDE_LIVE_REPOSITORY"):
            sandbox_command(
                bundle_root=ROOT,
                output_directory=self.output,
                trace_prefix=self.trace / "syscalls",
                case_id="max_carriers_L",
            )
        link = self.base / "repo-link"
        link.symlink_to(REPOSITORY_ROOT)
        with self.assertRaisesRegex(IsolatedRunnerError, "BUNDLE_INSIDE_LIVE_REPOSITORY"):
            sandbox_command(
                bundle_root=link,
                output_directory=self.output,
                trace_prefix=self.trace / "syscalls",
                case_id="max_carriers_L",
            )

    def test_worker_import_is_after_limits_and_atomic_publish_is_noreplace(self) -> None:
        source = (ROOT / "confirmation_worker.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_imports = [
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and "formal.chemgraph" in ast.unparse(node)
        ]
        self.assertEqual(top_imports, [])
        self.assertLess(source.index("_apply_resource_limits()"), source.index("from formal.chemgraph"))
        self.assertIn("os.O_EXCL", source)
        self.assertIn("os.O_NOFOLLOW", source)
        self.assertNotIn("subprocess", source)
        target = self.output / "candidate.json"
        target.write_bytes(b"old\n")
        with self.assertRaises(FileExistsError):
            _atomic_write(target, b"new\n")
        self.assertEqual(target.read_bytes(), b"old\n")
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), ["candidate.json"])

    def test_valid_exact_rejection_envelope_passes(self) -> None:
        case = self._install_source("max_carriers_H")
        self._write_rejection(case)
        result = self._validate_rejection(case)
        self.assertEqual(result["output_kind"], "rejection")
        self.assertEqual(result["stdout_sha256"], EMPTY_SHA)

    def test_wrong_h_code_and_kind_swap_are_no_go(self) -> None:
        case = self._install_source("max_carriers_H")
        self._write_rejection(case)
        value = json.loads((self.output / "rejection.json").read_bytes())
        value["error"] = "CAPACITY_EXCEEDED:max_intersections"
        (self.output / "rejection.json").write_bytes(_canonical(value))
        with self.assertRaisesRegex(IsolatedRunnerError, "CASE_REJECTION_NOT_EXACT"):
            self._validate_rejection(case)
        (self.output / "rejection.json").unlink()
        (self.output / "candidate.json").write_bytes(_canonical({"status": "candidate_domain"}))
        with self.assertRaisesRegex(IsolatedRunnerError, "CASE_OUTPUT_SET_NOT_CLOSED"):
            self._validate_rejection(case)

    def test_partial_extra_noncanonical_and_truncated_outputs_are_no_go(self) -> None:
        case = self._install_source("max_carriers_H")
        self._write_rejection(case)
        (self.output / ".candidate.json.tmp").write_bytes(b"partial")
        with self.assertRaisesRegex(IsolatedRunnerError, "CASE_OUTPUT_SET_NOT_CLOSED"):
            self._validate_rejection(case)
        (self.output / ".candidate.json.tmp").unlink()
        raw = (self.output / "rejection.json").read_bytes()
        (self.output / "rejection.json").write_bytes(raw.replace(b'"error":', b'"error" :'))
        with self.assertRaisesRegex(IsolatedRunnerError, "NONCANONICAL"):
            self._validate_rejection(case)
        (self.output / "rejection.json").write_bytes(raw[:-2])
        with self.assertRaisesRegex(IsolatedRunnerError, "JSON_INVALID"):
            self._validate_rejection(case)

    def test_timeout_exit_stdio_and_toctou_are_no_go(self) -> None:
        case = self._install_source("max_carriers_H")
        self._write_rejection(case)
        mutations = (
            ({"timed_out": True}, "CASE_TIMEOUT"),
            ({"exit_code": 0}, "CASE_EXIT_CODE_MISMATCH"),
            ({"stderr": b"diagnostic"}, "CASE_STDIO_NOT_EMPTY"),
            ({"source_post_sha256": "0" * 64}, "CASE_SOURCE_TOCTOU"),
        )
        for changes, code in mutations:
            with self.subTest(code=code), self.assertRaisesRegex(IsolatedRunnerError, code):
                self._validate_rejection(case, **changes)

    def test_trace_network_process_outside_write_and_truncation_are_no_go(self) -> None:
        trace = self.trace / "syscalls.123"
        prefix = self.trace / "syscalls"
        valid = (
            '1 execve("/usr/bin/python3", ["python3"], []) = 0\n'
            '2 openat(AT_FDCWD, "/out/.rejection.json.tmp", O_WRONLY|O_CREAT, 0600) = 3\n'
            '3 +++ exited with 20 +++\n'
        )
        trace.write_text(valid, encoding="utf-8")
        self.assertEqual(validate_trace(prefix)["python_exec_count"], 1)
        for syscall, code in (
            ('socket(AF_INET, SOCK_STREAM, 0) = -1 EPERM', "TRACE_NETWORK_ATTEMPT"),
            ('fork() = -1 EAGAIN', "TRACE_POST_PYTHON_PROCESS_ATTEMPT"),
            ('openat(AT_FDCWD, "/tmp/leak", O_WRONLY|O_CREAT, 0600) = -1 EROFS', "TRACE_OUTSIDE_OUTPUT_MUTATION_ATTEMPT"),
        ):
            trace.write_text(valid.replace("3 +++", f"2 {syscall}\n3 +++"), encoding="utf-8")
            with self.subTest(code=code), self.assertRaisesRegex(IsolatedRunnerError, code):
                validate_trace(prefix)
        trace.write_text(valid.replace("3 +++ exited with 20 +++\n", ""), encoding="utf-8")
        with self.assertRaisesRegex(IsolatedRunnerError, "TRACE_INCOMPLETE_OR_UNPARSEABLE"):
            validate_trace(prefix)

    def test_authority_publish_collision_is_no_go(self) -> None:
        publish_authority_marker(self.output, "a" * 64)
        with self.assertRaises(FileExistsError):
            publish_authority_marker(self.output, "b" * 64)
        self.assertEqual((self.output / "AUTHORITY").read_text(encoding="ascii"), "a" * 64 + "\n")


if __name__ == "__main__":
    unittest.main()
