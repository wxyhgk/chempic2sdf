from __future__ import annotations

import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from verify_offline import (
    CanonicalJSONError,
    CONTRACT_SCHEMA_SHA256,
    CONTRACT_SHA256,
    EmbeddedSchemaRegistry,
    VerificationError,
    canonical_json_bytes,
    decode_canonical_json,
    framed_json_sha256,
    load_canonical_json,
    main,
    verify_exact_four_contract,
    verify_fixed_contract,
    verify_inbox_record,
    verify_offline,
    verify_protocol_hashes,
    verify_relation_registry,
    verify_role_matrix_hash,
    verify_state_machine_totality,
)


HERE = Path(__file__).resolve().parent
CHALLENGE = HERE.parent
CONTRACT_PATH = CHALLENGE / "prereg/e2_launcher_contract_v0_4.json"
SCHEMA_PATH = CHALLENGE / "schemas/e2_launcher_contract_v0_4.schema.json"


def valid_inbox_record() -> dict:
    entry = {
        "byte_length": 1,
        "fsynced": True,
        "gid": 0,
        "mode": "0444",
        "nlink": 1,
        "path": "authorization.json",
        "sha256": "1" * 64,
        "uid": 0,
    }
    return {
        "all_files_and_parent_fsynced": True,
        "all_paths_unique_sorted": True,
        "atomic_noreplace_passed": True,
        "attestation_entry_count": 2,
        "authorization_entry_count": 1,
        "combined_root_sha256": "2" * 64,
        "content_root_sha256": "3" * 64,
        "entries": [dict(entry, path=f"entry-{index}.json") for index in range(4)],
        "entry_count": 4,
        "identity_registry_entry_count": 1,
        "inbox_root": "/opt/chemgraph/authorization-inbox-v0.3",
        "metadata_root_sha256": "4" * 64,
        "reopen_full_bytes_equal": True,
        "schema_version": "chemgraph-e2-authorization-inbox-commitment-v0.4",
        "status": "frozen",
    }


class OfflineVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_canonical_json(CONTRACT_PATH)
        cls.registry = EmbeddedSchemaRegistry(cls.contract)

    def test_frozen_contract_and_schema(self) -> None:
        contract, schema = verify_fixed_contract(CONTRACT_PATH, SCHEMA_PATH)
        self.assertEqual(set(contract), set(schema["required"]))
        self.assertEqual(CONTRACT_SHA256, __import__("hashlib").sha256(CONTRACT_PATH.read_bytes()).hexdigest())
        self.assertEqual(CONTRACT_SCHEMA_SHA256, __import__("hashlib").sha256(SCHEMA_PATH.read_bytes()).hexdigest())

    def test_duplicate_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(VerificationError, "duplicate JSON key"):
            decode_canonical_json(b'{"x":1,"x":2}\n')

    def test_noncanonical_json_is_rejected(self) -> None:
        with self.assertRaisesRegex(VerificationError, "not frozen canonical"):
            decode_canonical_json(b'{"x": 1}\n')

    def test_nonfinite_json_constants_are_rejected_everywhere(self) -> None:
        cases = (
            b"NaN\n",
            b"Infinity\n",
            b"-Infinity\n",
            b'{"nested":[1,NaN,{"value":Infinity},-Infinity]}\n',
        )
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaisesRegex(CanonicalJSONError, "non-standard"):
                decode_canonical_json(raw)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_json_bytes(value)

    def test_cli_malformed_and_nonstandard_json_exit_two(self) -> None:
        cases = (b'{"broken":\n', b'{"value":NaN}\n', b'"\xff"\n')
        with tempfile.TemporaryDirectory() as directory:
            for index, raw in enumerate(cases):
                path = Path(directory) / f"bad-{index}.json"
                path.write_bytes(raw)
                output = io.StringIO()
                with self.subTest(raw=raw), contextlib.redirect_stdout(output):
                    self.assertEqual(2, main(["--contract", str(path), "--schema", str(SCHEMA_PATH)]))
                report = json.loads(output.getvalue())
                self.assertEqual("NO_GO", report["status"])
                self.assertEqual(1, report["summary"]["failed"])

    def test_protocol_hashes(self) -> None:
        verify_protocol_hashes(self.contract)
        mutations = (
            ("record_schema_bundle", "schema_count", 16),
            ("relation_protocol", "relation_count", 45),
            ("state_machine_protocol", "failure_first_hop_expected_count", 260),
        )
        for payload, key, value in mutations:
            mutated = copy.deepcopy(self.contract)
            mutated[payload][key] = value
            with self.subTest(payload=payload), self.assertRaisesRegex(VerificationError, payload):
                verify_protocol_hashes(mutated)

    def test_framing_is_length_and_domain_sensitive(self) -> None:
        value = {"x": 1}
        digest = framed_json_sha256("domain-a", value)
        self.assertNotEqual(digest, framed_json_sha256("domain-b", value))
        self.assertNotEqual(digest, framed_json_sha256("domain-a", {"x": 2}))

    def test_all_embedded_schemas_check_and_select(self) -> None:
        self.assertEqual(17, len(self.registry.schemas))
        for name, schema in self.registry.schemas.items():
            Draft202012Validator.check_schema(schema)
            if name == "relation_obligation":
                selected, _ = self.registry.select(self.contract["relation_protocol"]["relations"][0], name)
                self.assertEqual(name, selected)
                continue
            versions = set()
            stack = [schema]
            while stack:
                current = stack.pop()
                if isinstance(current, dict):
                    version = current.get("properties", {}).get("schema_version", {}).get("const")
                    if isinstance(version, str):
                        versions.add(version)
                    stack.extend(current.values())
                elif isinstance(current, list):
                    stack.extend(current)
            self.assertTrue(versions, name)
            for version in versions:
                selected, _ = self.registry.select({"schema_version": version})
                self.assertEqual(name, selected)

    def test_unknown_schema_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(VerificationError, "no embedded schema"):
            self.registry.select({"schema_version": "unknown"})
        with self.assertRaisesRegex(VerificationError, "explicit"):
            self.registry.select({})

    def test_embedded_id_collision_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.contract)
        schemas = mutated["record_schema_bundle"]["schemas"]
        schemas["authority_marker"]["$id"] = schemas["artifact_tree_certificate"]["$id"]
        with self.assertRaisesRegex(VerificationError, "globally unique"):
            EmbeddedSchemaRegistry(mutated)

    def test_schema_version_collision_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.contract)
        schemas = mutated["record_schema_bundle"]["schemas"]
        schemas["authority_marker"]["properties"]["schema_version"]["const"] = schemas[
            "artifact_tree_certificate"
        ]["properties"]["schema_version"]["const"]
        with self.assertRaisesRegex(VerificationError, "ambiguously selects"):
            EmbeddedSchemaRegistry(mutated)

    def test_explicit_wrong_schema_is_rejected(self) -> None:
        with self.assertRaisesRegex(VerificationError, "fails artifact_tree_certificate"):
            self.registry.validate(valid_inbox_record(), "artifact_tree_certificate")

    def test_relation_registry(self) -> None:
        verify_relation_registry(self.contract, self.registry)
        mutated = copy.deepcopy(self.contract)
        mutated["relation_protocol"]["relations"][1]["id"] = "R01"
        with self.assertRaisesRegex(VerificationError, "R01..R46"):
            verify_relation_registry(mutated, EmbeddedSchemaRegistry(mutated))

    def test_state_machine_261_totality(self) -> None:
        verify_state_machine_totality(self.contract)
        mutated = copy.deepcopy(self.contract)
        mutated["state_machine_protocol"]["failure_first_hop_transitions"].pop()
        with self.assertRaisesRegex(VerificationError, "total and deterministic"):
            verify_state_machine_totality(mutated)

    def test_state_machine_duplicate_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.contract)
        transitions = mutated["state_machine_protocol"]["failure_first_hop_transitions"]
        transitions[-1] = copy.deepcopy(transitions[0])
        with self.assertRaisesRegex(VerificationError, "total and deterministic"):
            verify_state_machine_totality(mutated)

    def test_failure_cleanup_cannot_reach_authority(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["state_machine_protocol"]["failure_cleanup_transitions"][0]["target"] = "authoritative"
        with self.assertRaisesRegex(VerificationError, "reach authority"):
            verify_state_machine_totality(mutated)

    def test_dry_run_cannot_enter_official_state(self) -> None:
        mutated = copy.deepcopy(self.contract)
        paths = mutated["state_machine_protocol"]["phase_success_transitions"]["dry_run"]
        paths[-1] = "dryRunPostchecked->authoritative"
        mutated["state_machine_protocol"]["phase_terminal_success"]["dry_run"] = "authoritative"
        mutated["state_machine_protocol"]["success_transitions"].append("dryRunPostchecked->authoritative")
        with self.assertRaisesRegex(VerificationError, "phase success terminals"):
            verify_state_machine_totality(mutated)

    def test_official_terminal_cannot_be_synchronously_rewritten(self) -> None:
        mutated = copy.deepcopy(self.contract)
        paths = mutated["state_machine_protocol"]["phase_success_transitions"]["official"]
        paths[-1] = "detachedMarkerInstalled->dryRunPassed"
        mutated["state_machine_protocol"]["phase_terminal_success"]["official"] = "dryRunPassed"
        mutated["state_machine_protocol"]["success_transitions"].remove(
            "detachedMarkerInstalled->authoritative"
        )
        mutated["state_machine_protocol"]["success_transitions"].append(
            "detachedMarkerInstalled->dryRunPassed"
        )
        with self.assertRaisesRegex(VerificationError, "phase success terminals"):
            verify_state_machine_totality(mutated)

    def test_outer_death_targets_are_frozen_to_orphaned(self) -> None:
        sources = ("timedOut", "watchdogDead", "deadlineExecutorDead")
        for source in sources:
            mutated = copy.deepcopy(self.contract)
            transitions = mutated["state_machine_protocol"]["failure_cleanup_transitions"]
            next(item for item in transitions if item["source"] == source)["target"] = "quarantine"
            with self.subTest(source=source), self.assertRaisesRegex(
                VerificationError, "orphanedNonAuthoritativeCandidate"
            ):
                verify_state_machine_totality(mutated)

    def test_role_matrix_hash(self) -> None:
        verify_role_matrix_hash(self.contract)
        mutated = copy.deepcopy(self.contract)
        mutated["isolation_matrix"]["rows"][0]["uid"] += 1
        with self.assertRaisesRegex(VerificationError, "matrix"):
            verify_role_matrix_hash(mutated)

    def test_exact_four_contract(self) -> None:
        verify_exact_four_contract(self.contract)
        mutated = copy.deepcopy(self.contract)
        del mutated["record_schema_bundle"]["schemas"]["authorization_inbox_commitment"]["properties"]["entries"]["maxItems"]
        with self.assertRaisesRegex(VerificationError, "exactly four"):
            verify_exact_four_contract(mutated)

    def test_fifth_inbox_entry_is_rejected(self) -> None:
        schema = self.registry.schemas["authorization_inbox_commitment"]
        record = valid_inbox_record()
        self.registry.validate(record, "authorization_inbox_commitment")
        verify_inbox_record(record)
        record["entries"].append(dict(record["entries"][0], path="fifth.json"))
        self.assertFalse(Draft202012Validator(schema).is_valid(record))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fifth.json"
            path.write_bytes(canonical_json_bytes(record))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(
                    [
                        "--record",
                        f"authorization_inbox_commitment={path}",
                        "--contract",
                        str(CONTRACT_PATH),
                        "--schema",
                        str(SCHEMA_PATH),
                    ]
                )
            self.assertEqual(2, code)
            report = json.loads(output.getvalue())
            self.assertEqual("NO_GO", report["status"])
            self.assertEqual(1, report["summary"]["failed"])

    def test_inbox_sum_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(VerificationError, "entry_count"):
            verify_inbox_record(
                {
                    "entries": [{}, {}, {}, {}],
                    "entry_count": 4,
                    "authorization_entry_count": 1,
                    "attestation_entry_count": 3,
                    "identity_registry_entry_count": 1,
                }
            )

    def test_modified_contract_bytes_fail_fixed_hash(self) -> None:
        mutated = copy.deepcopy(self.contract)
        mutated["execution_authorized"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_bytes(canonical_json_bytes(mutated))
            with self.assertRaisesRegex(VerificationError, "contract SHA256"):
                verify_fixed_contract(path, SCHEMA_PATH)

    def test_modified_schema_bytes_fail_fixed_hash_and_cli(self) -> None:
        schema = load_canonical_json(SCHEMA_PATH)
        schema["$id"] = "drifted.schema.json"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schema.json"
            path.write_bytes(canonical_json_bytes(schema))
            with self.assertRaisesRegex(VerificationError, "schema SHA256"):
                verify_fixed_contract(CONTRACT_PATH, path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    2,
                    main(["--contract", str(CONTRACT_PATH), "--schema", str(path)]),
                )
            self.assertEqual("NO_GO", json.loads(output.getvalue())["status"])

    def test_value_schema_and_unexpected_errors_become_no_go(self) -> None:
        errors = (ValueError("bad value"), SchemaError("bad schema"))
        for error in errors:
            with self.subTest(error=type(error).__name__), mock.patch(
                "verify_offline.verify_fixed_contract", side_effect=error
            ):
                report = verify_offline(CONTRACT_PATH, SCHEMA_PATH)
            self.assertEqual("NO_GO", report["status"])
            self.assertEqual(1, report["summary"]["failed"])
        output = io.StringIO()
        with mock.patch("verify_offline.verify_offline", side_effect=RuntimeError("unexpected")), contextlib.redirect_stdout(output):
            self.assertEqual(2, main([]))
        self.assertEqual("NO_GO", json.loads(output.getvalue())["status"])

    def test_offline_report_is_always_no_go(self) -> None:
        report = verify_offline(CONTRACT_PATH, SCHEMA_PATH)
        self.assertEqual("NO_GO", report["status"])
        self.assertFalse(report["execution_authorized"])
        self.assertFalse(report["authorization_record_created"])
        self.assertEqual(46, report["summary"]["relation_count"])
        self.assertEqual(3, report["summary"]["passed_relations"])
        self.assertEqual(43, report["summary"]["unobserved_relations"])
        self.assertEqual(0, report["summary"]["failed"])


if __name__ == "__main__":
    unittest.main()
