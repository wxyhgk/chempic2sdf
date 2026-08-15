#!/usr/bin/env python3
"""Offline-only verifier for the frozen E2 launcher v0.4 contract.

This module never launches a workload, creates authorization, or reads runtime
cgroup/mount/pidfd state. Runtime-owned relations stay ``unobserved`` and make
the aggregate decision ``NO_GO``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from jsonschema import Draft202012Validator
except ImportError as error:  # pragma: no cover - exercised by deployment, not unit tests
    Draft202012Validator = None  # type: ignore[assignment]
    JSONSCHEMA_IMPORT_ERROR = error
else:
    JSONSCHEMA_IMPORT_ERROR = None


CONTRACT_SHA256 = "ddb03c311e1e3a41f63d2438a8e4c7bf7b7fa9c3044fe6b15fe399e10c47a6ea"
CONTRACT_SCHEMA_SHA256 = "dfa0e0087cf7f065f9219c5d4bdb129723a8a1b3309a7db789828a60819bc5d6"
EXPECTED_RELATION_IDS = tuple(f"R{index:02d}" for index in range(1, 47))
OFFLINE_COMPLETE_RELATIONS = frozenset({"R27", "R37", "R38"})
RUNTIME_EVIDENCE_RELATIONS = frozenset(EXPECTED_RELATION_IDS) - OFFLINE_COMPLETE_RELATIONS


class VerificationError(ValueError):
    """Raised when frozen bytes or a local verifier invariant are invalid."""


class CanonicalJSONError(VerificationError):
    """Raised when input is outside the frozen canonical JSON language."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(token: str) -> None:
    raise CanonicalJSONError(f"non-standard JSON numeric constant: {token}")


def decode_canonical_json(raw: bytes, *, source: str = "<bytes>") -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError(f"{source}: JSON is not UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except CanonicalJSONError as error:
        raise CanonicalJSONError(f"{source}: invalid JSON: {error}") from error
    except (json.JSONDecodeError, VerificationError) as error:
        raise VerificationError(f"{source}: invalid JSON: {error}") from error
    if raw != canonical_json_bytes(value):
        raise VerificationError(f"{source}: JSON bytes are not frozen canonical JSON with one trailing LF")
    return value


def load_canonical_json(path: Path) -> Any:
    return decode_canonical_json(path.read_bytes(), source=str(path))


def framed_json_sha256(domain: str, value: Any) -> str:
    try:
        domain_bytes = domain.encode("ascii")
    except UnicodeEncodeError as error:
        raise VerificationError("framing domain must be ASCII") from error
    payload = canonical_json_bytes(value)
    framed = (
        struct.pack(">I", len(domain_bytes))
        + domain_bytes
        + struct.pack(">Q", len(payload))
        + payload
    )
    return sha256_bytes(framed)


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: str
    detail: str

    def as_json(self) -> dict[str, str]:
        return {"id": self.check_id, "status": self.status, "detail": self.detail}


def _require_jsonschema() -> None:
    if Draft202012Validator is None:
        raise VerificationError(f"jsonschema is required: {JSONSCHEMA_IMPORT_ERROR}")


def _walk_ids(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        schema_id = value.get("$id")
        if isinstance(schema_id, str):
            yield schema_id
        for child in value.values():
            yield from _walk_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_ids(child)


def _schema_versions(schema: Mapping[str, Any]) -> set[str]:
    versions: set[str] = set()
    properties = schema.get("properties")
    if isinstance(properties, dict):
        version_schema = properties.get("schema_version")
        if isinstance(version_schema, dict) and isinstance(version_schema.get("const"), str):
            versions.add(version_schema["const"])
    for branch_name in ("oneOf", "anyOf", "allOf"):
        branches = schema.get(branch_name, [])
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, dict):
                    versions.update(_schema_versions(branch))
    return versions


class EmbeddedSchemaRegistry:
    def __init__(self, contract: Mapping[str, Any]) -> None:
        _require_jsonschema()
        schemas = contract["record_schema_bundle"]["schemas"]
        if not isinstance(schemas, dict) or len(schemas) != 17:
            raise VerificationError("record schema bundle must contain exactly 17 schemas")
        self.schemas: dict[str, dict[str, Any]] = schemas
        self.by_version: dict[str, str] = {}
        for name, schema in self.schemas.items():
            Draft202012Validator.check_schema(schema)
            for version in _schema_versions(schema):
                previous = self.by_version.setdefault(version, name)
                if previous != name:
                    raise VerificationError(
                        f"schema_version {version!r} ambiguously selects {previous!r} and {name!r}"
                    )
        ids = list(_walk_ids(self.schemas))
        if len(ids) != 45 or len(set(ids)) != 45:
            raise VerificationError(f"expected 45 globally unique embedded $id values, got {len(ids)}")

    def select(self, record: Mapping[str, Any], explicit_name: str | None = None) -> tuple[str, dict[str, Any]]:
        if explicit_name is not None:
            if explicit_name not in self.schemas:
                raise VerificationError(f"unknown embedded schema: {explicit_name}")
            return explicit_name, self.schemas[explicit_name]
        version = record.get("schema_version")
        if not isinstance(version, str):
            raise VerificationError("record has no schema_version; pass an explicit embedded schema name")
        name = self.by_version.get(version)
        if name is None:
            raise VerificationError(f"no embedded schema selects schema_version {version!r}")
        return name, self.schemas[name]

    def validate(self, record: Mapping[str, Any], explicit_name: str | None = None) -> str:
        name, schema = self.select(record, explicit_name)
        errors = sorted(Draft202012Validator(schema).iter_errors(record), key=lambda item: list(item.path))
        if errors:
            first = errors[0]
            path = "/".join(str(part) for part in first.path) or "<root>"
            raise VerificationError(f"record fails {name} at {path}: {first.message}")
        return name


def verify_fixed_contract(contract_path: Path, schema_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_raw = contract_path.read_bytes()
    schema_raw = schema_path.read_bytes()
    contract = decode_canonical_json(contract_raw, source=str(contract_path))
    schema = decode_canonical_json(schema_raw, source=str(schema_path))
    if sha256_bytes(contract_raw) != CONTRACT_SHA256:
        raise VerificationError("contract SHA256 differs from frozen v0.4")
    if sha256_bytes(schema_raw) != CONTRACT_SCHEMA_SHA256:
        raise VerificationError("contract schema SHA256 differs from frozen v0.4")
    _require_jsonschema()
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(contract))
    if errors:
        raise VerificationError(f"contract fails frozen schema: {errors[0].message}")
    if set(schema.get("required", [])) != set(contract):
        raise VerificationError("main schema required fields differ from contract fields")
    if any(schema["properties"].get(key) != {"const": value} for key, value in contract.items()):
        raise VerificationError("main schema is not an exact per-field const projection of the contract")
    return contract, schema


def verify_protocol_hashes(contract: Mapping[str, Any]) -> None:
    checks = (
        ("record_schema_bundle_hash_protocol", "record_schema_bundle", "record_schema_bundle_sha256"),
        ("relation_protocol_hash_protocol", "relation_protocol", "relation_protocol_sha256"),
        ("state_machine_hash_protocol", "state_machine_protocol", "state_machine_sha256"),
    )
    for protocol_key, payload_key, digest_key in checks:
        protocol = contract[protocol_key]
        actual = framed_json_sha256(protocol["domain"], contract[payload_key])
        if actual != protocol["sha256"] or actual != contract[digest_key]:
            raise VerificationError(f"{payload_key} domain-framed SHA256 mismatch")


def verify_relation_registry(contract: Mapping[str, Any], registry: EmbeddedSchemaRegistry) -> None:
    protocol = contract["relation_protocol"]
    relations = protocol["relations"]
    if protocol["relation_count"] != 46 or len(relations) != 46:
        raise VerificationError("relation registry must contain exactly 46 entries")
    if tuple(record.get("id") for record in relations) != EXPECTED_RELATION_IDS:
        raise VerificationError("relation registry IDs must be exactly R01..R46 in order")
    verifier_ids = [record.get("verifier_id") for record in relations]
    theorem_ids = [record.get("lean_theorem_id") for record in relations]
    if len(set(verifier_ids)) != 46 or len(set(theorem_ids)) != 46:
        raise VerificationError("relation verifier/theorem IDs must be unique")
    for record in relations:
        registry.validate(record, "relation_obligation")


def verify_state_machine_totality(contract: Mapping[str, Any]) -> None:
    state = contract["state_machine_protocol"]
    events = state["failure_event_tags"]
    sources_by_phase = state["failure_sources_by_phase"]
    transitions = state["failure_first_hop_transitions"]
    expected = {
        (phase, source, event)
        for phase, sources in sources_by_phase.items()
        for source in sources
        for event in events
    }
    observed = [(item["phase"], item["source"], item["event"]) for item in transitions]
    if len(expected) != 261 or state["failure_first_hop_expected_count"] != 261:
        raise VerificationError("state-machine Cartesian failure domain is not exactly 261")
    if len(observed) != 261 or len(set(observed)) != 261 or set(observed) != expected:
        raise VerificationError("failure first-hop table is not total and deterministic")
    failure_targets = set(state["failure_targets"])
    if any(item["target"] not in failure_targets for item in transitions):
        raise VerificationError("failure first-hop table contains a non-failure target")
    if not state["failure_first_hop_total_and_deterministic"]:
        raise VerificationError("contract does not assert deterministic failure first hops")

    states = set(state["states"])
    cleanup_graph: dict[str, set[str]] = {}
    for transition in state["failure_cleanup_transitions"]:
        source = transition["source"]
        target = transition["target"]
        if source not in states or target not in states:
            raise VerificationError("failure cleanup transition references an unknown state")
        cleanup_graph.setdefault(source, set()).add(target)
    nonauthority_terminals = {"quarantine", "orphanedNonAuthoritativeCandidate"}
    required_outer_death_targets = {
        "timedOut": {"orphanedNonAuthoritativeCandidate"},
        "watchdogDead": {"orphanedNonAuthoritativeCandidate"},
        "deadlineExecutorDead": {"orphanedNonAuthoritativeCandidate"},
    }
    for source, targets in required_outer_death_targets.items():
        if cleanup_graph.get(source) != targets:
            raise VerificationError(f"{source} must transition only to orphanedNonAuthoritativeCandidate")
    for first_hop in failure_targets:
        reachable = {first_hop}
        frontier = [first_hop]
        while frontier:
            source = frontier.pop()
            for target in cleanup_graph.get(source, set()):
                if target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        if "authoritative" in reachable:
            raise VerificationError("a failure first-hop target can reach authority")
        if not reachable.intersection(nonauthority_terminals):
            raise VerificationError(f"failure state {first_hop} has no non-authority terminal")

    phase_paths = state["phase_success_transitions"]
    if state["phase_terminal_success"] != {
        "dry_run": "dryRunPassed",
        "official": "authoritative",
    }:
        raise VerificationError("phase success terminals are not the frozen dryRunPassed/authoritative pair")
    union: set[str] = set()
    for phase, transitions_for_phase in phase_paths.items():
        if not transitions_for_phase:
            raise VerificationError(f"{phase} success path is empty")
        parsed: list[tuple[str, str]] = []
        for encoded in transitions_for_phase:
            parts = encoded.split("->")
            if len(parts) != 2 or parts[0] not in states or parts[1] not in states:
                raise VerificationError(f"invalid {phase} success transition: {encoded}")
            parsed.append((parts[0], parts[1]))
            union.add(encoded)
        if parsed[0][0] != "initial" or any(left[1] != right[0] for left, right in zip(parsed, parsed[1:])):
            raise VerificationError(f"{phase} success path is not contiguous from initial")
        if parsed[-1][1] != state["phase_terminal_success"][phase]:
            raise VerificationError(f"{phase} success path has the wrong terminal")
    dry_states = {part for encoded in phase_paths["dry_run"] for part in encoded.split("->")}
    if dry_states.intersection({"renamedCandidate", "postchecked", "detachedMarkerInstalled", "authoritative"}):
        raise VerificationError("dry-run success path enters an official-only state")
    if union != set(state["success_transitions"]):
        raise VerificationError("global success transition set differs from phase paths")


def verify_role_matrix_hash(contract: Mapping[str, Any]) -> None:
    matrix = contract["isolation_matrix"]
    actual = framed_json_sha256(matrix["matrix_hash_domain"], matrix["rows"])
    if actual != matrix["matrix_sha256"]:
        raise VerificationError("role permission matrix framed SHA256 mismatch")
    if actual != contract["contract_conformance"]["frozen_values"]["role_matrix_sha256"]:
        raise VerificationError("role matrix hash is not bound into ContractConformance")


def verify_exact_four_contract(contract: Mapping[str, Any]) -> None:
    schema = contract["record_schema_bundle"]["schemas"]["authorization_inbox_commitment"]
    properties = schema["properties"]
    entries = properties["entries"]
    if entries.get("minItems") != 4 or entries.get("maxItems") != 4:
        raise VerificationError("authorization inbox entries are not frozen to exactly four")
    if properties.get("entry_count") != {"const": 4}:
        raise VerificationError("authorization inbox entry_count is not const 4")
    if contract["authorization_inbox_protocol"].get("exact_entry_count") != 4:
        raise VerificationError("authorization_inbox_protocol exact_entry_count is not 4")
    relation = contract["relation_protocol"]["relations"][28]
    required_text = (
        "entry_count=len(entries)=authorization_entry_count+attestation_entry_count+"
        "identity_registry_entry_count=1+2+1=4"
    )
    if relation.get("id") != "R29" or required_text not in relation.get("obligation", ""):
        raise VerificationError("R29 does not bind the exact-four equality")


def verify_inbox_record(record: Mapping[str, Any]) -> None:
    entries = record["entries"]
    count_sum = (
        record["authorization_entry_count"]
        + record["attestation_entry_count"]
        + record["identity_registry_entry_count"]
    )
    if record["entry_count"] != len(entries) or record["entry_count"] != count_sum or count_sum != 4:
        raise VerificationError("inbox record violates entry_count=len(entries)=1+2+1=4")


def _record_arguments(values: Sequence[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise VerificationError("--record must use EMBEDDED_SCHEMA=PATH")
        parsed.append((name, Path(path)))
    return parsed


def _fatal_report(error: Exception) -> dict[str, Any]:
    return {
        "schema_version": "chemgraph-e2-launcher-offline-verification-v0.4",
        "status": "NO_GO",
        "execution_authorized": False,
        "training_authorized": False,
        "launcher_executed": False,
        "authorization_record_created": False,
        "checks": [CheckResult("fatal", "failed", str(error)).as_json()],
        "relations": [],
        "summary": {
            "failed": 1,
            "passed_relations": 0,
            "unobserved_relations": 0,
            "relation_count": 0,
            "reason": "offline verification failed before runtime authority could be established",
        },
    }


def verify_offline(
    contract_path: Path,
    schema_path: Path,
    record_paths: Sequence[tuple[str, Path]] = (),
) -> dict[str, Any]:
    checks: list[CheckResult] = []
    relations: list[CheckResult] = []
    try:
        contract, _ = verify_fixed_contract(contract_path, schema_path)
        checks.append(CheckResult("fixed_contract_bytes", "passed", "canonical bytes and frozen hashes match"))
        verify_protocol_hashes(contract)
        checks.append(CheckResult("framed_protocol_hashes", "passed", "bundle, relation and state hashes recomputed"))
        registry = EmbeddedSchemaRegistry(contract)
        checks.append(CheckResult("embedded_schemas", "passed", "17 Draft 2020-12 schemas; 45 unique $id values"))
        verify_relation_registry(contract, registry)
        checks.append(CheckResult("relation_registry", "passed", "46 ordered and schema-valid relation records"))
        verify_state_machine_totality(contract)
        checks.append(CheckResult("state_machine", "passed", "261/261 deterministic failure first hops"))
        verify_role_matrix_hash(contract)
        checks.append(CheckResult("role_matrix_hash", "passed", "R37 framing recomputed"))
        verify_exact_four_contract(contract)
        checks.append(CheckResult("exact_four_contract", "passed", "inbox schema and R29 equality are frozen"))

        for explicit_name, path in record_paths:
            record = load_canonical_json(path)
            selected = registry.validate(record, explicit_name)
            if selected == "authorization_inbox_commitment":
                verify_inbox_record(record)
            checks.append(CheckResult(f"record:{selected}", "passed", f"validated {path}"))

        for relation_id in EXPECTED_RELATION_IDS:
            if relation_id in OFFLINE_COMPLETE_RELATIONS:
                detail = {
                    "R27": "static state graph and 261-entry first-hop table verified",
                    "R37": "static role-matrix domain framing verified",
                    "R38": "17 schemas check; 45 embedded $id values are globally unique",
                }[relation_id]
                relations.append(CheckResult(relation_id, "passed", detail))
            else:
                detail = "requires runtime/external full-byte evidence; deliberately not inferred offline"
                if relation_id == "R29":
                    detail = "exact-four structure checked; ownership/fsync/NOREPLACE/full-byte membership remain unobserved"
                relations.append(CheckResult(relation_id, "unobserved", detail))
    except Exception as error:
        return _fatal_report(error)

    failed = sum(result.status == "failed" for result in checks + relations)
    unobserved = sum(result.status == "unobserved" for result in relations)
    return {
        "schema_version": "chemgraph-e2-launcher-offline-verification-v0.4",
        "status": "NO_GO",
        "execution_authorized": False,
        "training_authorized": False,
        "launcher_executed": False,
        "authorization_record_created": False,
        "checks": [result.as_json() for result in checks],
        "relations": [result.as_json() for result in relations],
        "summary": {
            "failed": failed,
            "passed_relations": sum(result.status == "passed" for result in relations),
            "unobserved_relations": unobserved,
            "relation_count": len(relations),
            "reason": "offline verification cannot establish runtime authority",
        },
    }


class NoExitArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise VerificationError(f"command line: {message}")


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    challenge = here.parent
    parser = NoExitArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=challenge / "prereg/e2_launcher_contract_v0_4.json",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=challenge / "schemas/e2_launcher_contract_v0_4.schema.json",
    )
    parser.add_argument(
        "--record",
        action="append",
        default=[],
        metavar="EMBEDDED_SCHEMA=PATH",
        help="validate a canonical record with an explicitly selected embedded schema",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        records = _record_arguments(args.record)
        report = verify_offline(args.contract, args.schema, records)
    except Exception as error:
        report = _fatal_report(error)
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 2 if report["summary"]["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
