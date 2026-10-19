from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path
from typing import Any

from . import generic_verifier as verifier


ROOT = Path(__file__).resolve().parent
TEMPLATE_ROOT = ROOT / "templates"
WITNESSES_ROOT = ROOT / "witnesses"
TEMPLATE_SCHEMA = "owner-capacity-feasible-world-template-v0.5"
PLACEHOLDER = "${ARTIFACT_SHA256}"
TEMPLATE_KEYS = {
    "ambiguous_overrides",
    "artifact_binding",
    "assignment_mode",
    "axis",
    "observation_sha256",
    "schema_version",
    "side",
    "source_sha256",
    "training_authorized",
}
BINDING_KEYS = {
    "digest_domain",
    "kind",
    "placeholder",
    "replacement_count",
    "required_format",
}
OVERRIDE_KEYS = {"owner_option_id", "variable_id"}
AXES = ("max_carriers", "max_pixel_variables", "max_total_rle_runs")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = verifier.load_canonical_json(raw, path.name)
    if not isinstance(value, dict):
        raise ValueError("TEMPLATE_NOT_OBJECT")
    return value, raw


def _validate_template(template: Any, raw: bytes) -> None:
    if not isinstance(template, dict) or set(template) != TEMPLATE_KEYS:
        raise ValueError("TEMPLATE_NOT_CLOSED")
    if raw != verifier.canonical_bytes(template):
        raise ValueError("TEMPLATE_NONCANONICAL")
    if (
        template["schema_version"] != TEMPLATE_SCHEMA
        or template["axis"] not in AXES
        or template["side"] != "L"
        or template["assignment_mode"] != "singleton_plus_overrides"
        or template["training_authorized"] is not False
    ):
        raise ValueError("TEMPLATE_HEADER_INVALID")
    if not (
        _is_sha256(template["source_sha256"])
        and _is_sha256(template["observation_sha256"])
        and template["source_sha256"] == template["observation_sha256"]
    ):
        raise ValueError("TEMPLATE_SOURCE_BINDING_INVALID")
    binding = template["artifact_binding"]
    if not isinstance(binding, dict) or set(binding) != BINDING_KEYS:
        raise ValueError("TEMPLATE_ARTIFACT_BINDING_NOT_CLOSED")
    if binding != {
        "digest_domain": "canonical_candidate_artifact_bytes",
        "kind": "deferred_exact_sha256",
        "placeholder": PLACEHOLDER,
        "replacement_count": 1,
        "required_format": "lowercase_hex_64",
    }:
        raise ValueError("TEMPLATE_ARTIFACT_BINDING_INVALID")
    if raw.count(PLACEHOLDER.encode("ascii")) != 1:
        raise ValueError("TEMPLATE_ARTIFACT_PLACEHOLDER_COUNT_INVALID")
    overrides = template["ambiguous_overrides"]
    if not isinstance(overrides, list):
        raise ValueError("TEMPLATE_OVERRIDES_INVALID")
    previous = -1
    for item in overrides:
        if not isinstance(item, dict) or set(item) != OVERRIDE_KEYS:
            raise ValueError("TEMPLATE_OVERRIDE_NOT_CLOSED")
        variable_id = item["variable_id"]
        owner_id = item["owner_option_id"]
        if (
            type(variable_id) is not int
            or type(owner_id) is not int
            or variable_id <= previous
            or owner_id < 0
        ):
            raise ValueError("TEMPLATE_OVERRIDE_INVALID")
        previous = variable_id


def _materialize_for_regression(
    template: dict[str, Any], artifact_raw: bytes
) -> tuple[dict[str, Any], bytes]:
    """Bind an independently rebuilt artifact in memory; never emits an L result."""
    _validate_template(template, verifier.canonical_bytes(template))
    artifact_sha = _sha256(artifact_raw)
    if not _is_sha256(artifact_sha):
        raise ValueError("ARTIFACT_SHA_INVALID")
    witness = {key: copy.deepcopy(value) for key, value in template.items() if key != "artifact_binding"}
    witness["artifact_sha256"] = artifact_sha
    witness["schema_version"] = verifier.WITNESS_SCHEMA
    return witness, verifier.canonical_bytes(witness)


def _validate_template_for_source(
    template: dict[str, Any],
    raw: bytes,
    observation_raw: bytes,
    domains: list[tuple[int, ...]],
) -> None:
    _validate_template(template, raw)
    source_sha = _sha256(observation_raw)
    if not (
        template["source_sha256"] == source_sha
        and template["observation_sha256"] == source_sha
    ):
        raise ValueError("TEMPLATE_FROZEN_SOURCE_MISMATCH")
    ambiguous_ids = {index for index, domain in enumerate(domains) if len(domain) > 1}
    override_by_id = {
        item["variable_id"]: item["owner_option_id"]
        for item in template["ambiguous_overrides"]
    }
    if len(override_by_id) != len(template["ambiguous_overrides"]):
        raise ValueError("TEMPLATE_OVERRIDE_DUPLICATE")
    if set(override_by_id) != ambiguous_ids:
        raise ValueError("TEMPLATE_OVERRIDE_COVERAGE_MISMATCH")
    for variable_id, owner_id in override_by_id.items():
        if owner_id not in domains[variable_id]:
            raise ValueError("TEMPLATE_OWNER_OUTSIDE_DOMAIN")


class FeasibleWitnessTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        plan, _ = _load_canonical(ROOT / "axis_plan.json")
        cls.limits = plan["limits"]
        cls.constructor_sha = plan["constructor"]["sha256"]
        cls.cases: dict[str, dict[str, Any]] = {}
        for axis in AXES:
            template_path = TEMPLATE_ROOT / f"{axis}_L.feasible.template.json"
            template, template_raw = _load_canonical(template_path)
            observation_path = WITNESSES_ROOT / f"{axis}_L.observation.json"
            observation, observation_raw = _load_canonical(observation_path)
            artifact, context = verifier.rebuild_expected_artifact(
                observation, cls.limits, cls.constructor_sha
            )
            cls.cases[axis] = {
                "artifact": artifact,
                "artifact_raw": verifier.canonical_bytes(artifact),
                "context": context,
                "observation_raw": observation_raw,
                "template": template,
                "template_raw": template_raw,
            }

    def test_schema_and_templates_are_closed_and_canonical(self) -> None:
        schema, schema_raw = _load_canonical(
            TEMPLATE_ROOT / "feasible_witness_template.schema.json"
        )
        self.assertEqual(schema_raw, verifier.canonical_bytes(schema))
        self.assertEqual(schema["additionalProperties"], False)
        self.assertEqual(set(schema["properties"]), TEMPLATE_KEYS)
        self.assertEqual(set(schema["required"]), TEMPLATE_KEYS)
        for case in self.cases.values():
            _validate_template(case["template"], case["template_raw"])

    def test_sources_and_override_domains_are_exact(self) -> None:
        for axis, case in self.cases.items():
            template = case["template"]
            source_sha = _sha256(case["observation_raw"])
            self.assertEqual(template["axis"], axis)
            self.assertEqual(template["source_sha256"], source_sha)
            self.assertEqual(template["observation_sha256"], source_sha)
            domains = case["context"]["domains"]
            _validate_template_for_source(
                template, case["template_raw"], case["observation_raw"], domains
            )
            expected_ids = {index for index, domain in enumerate(domains) if len(domain) > 1}
            override_by_id = {
                item["variable_id"]: item["owner_option_id"]
                for item in template["ambiguous_overrides"]
            }
            self.assertEqual(set(override_by_id), expected_ids)
            for variable_id, owner_id in override_by_id.items():
                self.assertIn(owner_id, domains[variable_id])

    def test_in_memory_regression_materialization_is_feasible(self) -> None:
        for axis, case in self.cases.items():
            witness, witness_raw = _materialize_for_regression(
                case["template"], case["artifact_raw"]
            )
            result = verifier.verify_feasible_witness(
                case["artifact"],
                case["context"]["domains"],
                witness,
                witness_raw,
                axis=axis,
                observation_sha256=_sha256(case["observation_raw"]),
                source_sha256=_sha256(case["observation_raw"]),
                artifact_sha256=_sha256(case["artifact_raw"]),
            )
            self.assertEqual(
                result["assignment_materialized_count"],
                len(case["context"]["domains"]),
            )

    def test_deferred_template_is_not_a_formal_witness(self) -> None:
        case = self.cases["max_carriers"]
        with self.assertRaisesRegex(
            verifier.GenericCapacityVerificationError,
            "FEASIBLE_WITNESS_NOT_CLOSED",
        ):
            verifier.verify_feasible_witness(
                case["artifact"],
                case["context"]["domains"],
                case["template"],
                case["template_raw"],
                axis="max_carriers",
                observation_sha256=_sha256(case["observation_raw"]),
                source_sha256=_sha256(case["observation_raw"]),
                artifact_sha256=_sha256(case["artifact_raw"]),
            )

    def test_artifact_binding_mutations_are_rejected(self) -> None:
        template = copy.deepcopy(self.cases["max_carriers"]["template"])
        template["artifact_binding"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "ARTIFACT_BINDING_NOT_CLOSED"):
            _validate_template(template, verifier.canonical_bytes(template))

        for field, value in (
            ("placeholder", "${CANDIDATE_SHA256}"),
            ("digest_domain", "arbitrary_json_bytes"),
            ("replacement_count", 2),
        ):
            mutated = copy.deepcopy(self.cases["max_carriers"]["template"])
            mutated["artifact_binding"][field] = value
            with self.assertRaises(ValueError):
                _validate_template(mutated, verifier.canonical_bytes(mutated))

        duplicated = copy.deepcopy(self.cases["max_carriers"]["template"])
        duplicated["artifact_binding"]["placeholder"] = PLACEHOLDER + PLACEHOLDER
        with self.assertRaises(ValueError):
            _validate_template(duplicated, verifier.canonical_bytes(duplicated))

    def test_override_and_source_mutations_are_rejected(self) -> None:
        case = self.cases["max_carriers"]
        domains = case["context"]["domains"]
        for overrides, error in (
            (
                case["template"]["ambiguous_overrides"][:-1],
                "OVERRIDE_COVERAGE_MISMATCH",
            ),
            (
                case["template"]["ambiguous_overrides"]
                + [copy.deepcopy(case["template"]["ambiguous_overrides"][-1])],
                "OVERRIDE",
            ),
        ):
            mutated = copy.deepcopy(case["template"])
            mutated["ambiguous_overrides"] = overrides
            raw = verifier.canonical_bytes(mutated)
            with self.assertRaisesRegex(ValueError, error):
                _validate_template_for_source(
                    mutated, raw, case["observation_raw"], domains
                )

        mutated = copy.deepcopy(case["template"])
        mutated["source_sha256"] = mutated["observation_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "FROZEN_SOURCE_MISMATCH"):
            _validate_template_for_source(
                mutated,
                verifier.canonical_bytes(mutated),
                case["observation_raw"],
                domains,
            )


if __name__ == "__main__":
    unittest.main()
