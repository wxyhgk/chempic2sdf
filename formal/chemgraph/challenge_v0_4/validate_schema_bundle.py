#!/usr/bin/env python3
"""Validate the Hidden Generator v0.4 schema bundle using only stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


EXPECTED_SCHEMAS = frozenset(
    {
        "audit_validation.schema.json",
        "candidate_set.schema.json",
        "challenge_spec.schema.json",
        "commitment.schema.json",
        "common.schema.json",
        "freeze_record.schema.json",
        "graph_set.schema.json",
        "image_pack_manifest.schema.json",
        "ink_program.schema.json",
        "interpretation.schema.json",
        "observation_gt.schema.json",
        "prediction_record.schema.json",
        "prediction_run_manifest.schema.json",
        "rejection_log.schema.json",
        "reveal.schema.json",
        "schema_bundle_manifest.schema.json",
        "sealed_gt_index.schema.json",
        "sealed_gt_manifest.schema.json",
        "transform.schema.json",
    }
)

SUPPORTED_KEYWORDS = frozenset(
    {
        "$defs",
        "$id",
        "$ref",
        "$schema",
        "additionalProperties",
        "anyOf",
        "const",
        "enum",
        "items",
        "maxItems",
        "minLength",
        "maximum",
        "minItems",
        "minimum",
        "pattern",
        "prefixItems",
        "properties",
        "required",
        "title",
        "type",
        "uniqueItems",
    }
)


class SchemaBundleError(ValueError):
    """The schema bundle violates the frozen stdlib-verifiable subset."""


class SchemaInstanceError(ValueError):
    """A JSON value does not conform to one schema in the bundle."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaBundleError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SchemaBundleError(f"nonfinite_json_number:{value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SchemaBundleError(f"invalid_schema_json:{path.name}:{error}") from error
    if not isinstance(value, dict):
        raise SchemaBundleError(f"schema_root_not_object:{path.name}")
    return value


def _resolve_fragment(document: dict[str, Any], fragment: str, label: str) -> None:
    if not fragment:
        return
    if not fragment.startswith("/"):
        raise SchemaBundleError(f"unsupported_ref_fragment:{label}:#{fragment}")
    current: Any = document
    for encoded_part in fragment[1:].split("/"):
        part = encoded_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise SchemaBundleError(f"unresolved_ref_fragment:{label}:#{fragment}")
        current = current[part]


def _validate_schema_node(
    node: Any,
    *,
    label: str,
    documents: dict[str, dict[str, Any]],
) -> None:
    if isinstance(node, bool):
        return
    if not isinstance(node, dict):
        raise SchemaBundleError(f"schema_node_not_object_or_bool:{label}")
    unknown = set(node) - SUPPORTED_KEYWORDS
    if unknown:
        raise SchemaBundleError(f"unsupported_schema_keywords:{label}:{sorted(unknown)}")

    if node.get("type") == "object" and node.get("additionalProperties") is not False:
        raise SchemaBundleError(f"object_not_closed:{label}")
    pattern = node.get("pattern")
    if pattern is not None:
        try:
            re.compile(pattern)
        except re.error as error:
            raise SchemaBundleError(f"invalid_pattern:{label}:{error}") from error

    reference = node.get("$ref")
    if reference is not None:
        if not isinstance(reference, str):
            raise SchemaBundleError(f"ref_not_string:{label}")
        file_name, marker, fragment = reference.partition("#")
        target_name = file_name or label.split("/", 1)[0]
        target = documents.get(target_name)
        if target is None:
            raise SchemaBundleError(f"unresolved_ref_file:{label}:{target_name}")
        _resolve_fragment(target, fragment if marker else "", label)

    for mapping_name in ("$defs", "properties"):
        mapping = node.get(mapping_name, {})
        if not isinstance(mapping, dict):
            raise SchemaBundleError(f"schema_mapping_not_object:{label}:{mapping_name}")
        for key, child in mapping.items():
            _validate_schema_node(
                child,
                label=f"{label}/{mapping_name}/{key}",
                documents=documents,
            )
    for sequence_name in ("anyOf", "prefixItems"):
        sequence = node.get(sequence_name, [])
        if not isinstance(sequence, list):
            raise SchemaBundleError(f"schema_sequence_not_array:{label}:{sequence_name}")
        for index, child in enumerate(sequence):
            _validate_schema_node(
                child,
                label=f"{label}/{sequence_name}/{index}",
                documents=documents,
            )
    if "items" in node:
        _validate_schema_node(
            node["items"], label=f"{label}/items", documents=documents
        )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_equal(left: Any, right: Any) -> bool:
    """JSON equality without Python's bool == int shortcut."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "array": isinstance(value, list),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "null": value is None,
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "object": isinstance(value, dict),
        "string": isinstance(value, str),
    }.get(expected, False)


def _resolve_ref(
    reference: str,
    *,
    current_name: str,
    documents: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | bool, str]:
    file_name, marker, fragment = reference.partition("#")
    target_name = file_name or current_name
    target: Any = documents.get(target_name)
    if target is None:
        raise SchemaInstanceError(f"unresolved_ref_file:{target_name}")
    if marker and fragment:
        if not fragment.startswith("/"):
            raise SchemaInstanceError(f"unsupported_ref_fragment:{reference}")
        for encoded_part in fragment[1:].split("/"):
            part = encoded_part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                raise SchemaInstanceError(f"unresolved_ref_fragment:{reference}")
            target = target[part]
    if not isinstance(target, (dict, bool)):
        raise SchemaInstanceError(f"ref_target_not_schema:{reference}")
    return target, target_name


def _validate_instance_node(
    value: Any,
    schema: dict[str, Any] | bool,
    *,
    path: str,
    current_name: str,
    documents: dict[str, dict[str, Any]],
) -> None:
    if schema is True:
        return
    if schema is False:
        raise SchemaInstanceError(f"false_schema:{path}")
    if "$ref" in schema:
        target, target_name = _resolve_ref(
            schema["$ref"], current_name=current_name, documents=documents
        )
        _validate_instance_node(
            value,
            target,
            path=path,
            current_name=target_name,
            documents=documents,
        )
        return
    if "anyOf" in schema:
        failures = []
        for branch in schema["anyOf"]:
            try:
                _validate_instance_node(
                    value,
                    branch,
                    path=path,
                    current_name=current_name,
                    documents=documents,
                )
                return
            except SchemaInstanceError as error:
                failures.append(str(error))
        raise SchemaInstanceError(f"any_of_failed:{path}:{failures}")

    expected_type = schema.get("type")
    if expected_type is not None:
        choices = [expected_type] if isinstance(expected_type, str) else expected_type
        if not isinstance(choices, list) or not all(isinstance(item, str) for item in choices):
            raise SchemaInstanceError(f"invalid_type_declaration:{path}")
        if not any(_matches_type(value, item) for item in choices):
            raise SchemaInstanceError(f"wrong_type:{path}:expected={choices}:actual={type(value).__name__}")
    if "const" in schema and not _json_equal(value, schema["const"]):
        raise SchemaInstanceError(f"const_mismatch:{path}")
    if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
        raise SchemaInstanceError(f"enum_mismatch:{path}:{value!r}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise SchemaInstanceError(f"string_too_short:{path}")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise SchemaInstanceError(f"pattern_mismatch:{path}:{value!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaInstanceError(f"below_minimum:{path}")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaInstanceError(f"above_maximum:{path}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise SchemaInstanceError(f"too_few_items:{path}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaInstanceError(f"too_many_items:{path}")
        if schema.get("uniqueItems"):
            for index, item in enumerate(value):
                if any(_json_equal(item, previous) for previous in value[:index]):
                    raise SchemaInstanceError(f"duplicate_array_item:{path}/{index}")
        prefixes = schema.get("prefixItems", [])
        for index, item_schema in enumerate(prefixes):
            if index < len(value):
                _validate_instance_node(
                    value[index], item_schema, path=f"{path}/{index}",
                    current_name=current_name, documents=documents,
                )
        item_schema = schema.get("items")
        if item_schema is not None:
            start = len(prefixes) if prefixes else 0
            for index in range(start, len(value)):
                _validate_instance_node(
                    value[index], item_schema, path=f"{path}/{index}",
                    current_name=current_name, documents=documents,
                )
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise SchemaInstanceError(f"missing_required:{path}:{missing}")
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise SchemaInstanceError(f"additional_properties:{path}:{extras}")
        for key, item in value.items():
            if key in properties:
                _validate_instance_node(
                    item,
                    properties[key],
                    path=f"{path}/{key}",
                    current_name=current_name,
                    documents=documents,
                )


def validate_instance(value: Any, schema_name: str, schema_dir: Path) -> None:
    """Validate an instance against the closed stdlib schema subset."""
    documents = {
        path.name: _load(path)
        for path in sorted(schema_dir.glob("*.schema.json"), key=lambda item: item.name)
    }
    schema = documents.get(schema_name)
    if schema is None:
        raise SchemaInstanceError(f"unknown_schema:{schema_name}")
    _validate_instance_node(
        value,
        schema,
        path="$",
        current_name=schema_name,
        documents=documents,
    )


def validate_bundle(schema_dir: Path) -> dict[str, Any]:
    paths = sorted(schema_dir.glob("*.schema.json"), key=lambda path: path.name)
    names = {path.name for path in paths}
    if names != EXPECTED_SCHEMAS:
        raise SchemaBundleError(
            "schema_file_set_mismatch:"
            f"missing={sorted(EXPECTED_SCHEMAS - names)}:extra={sorted(names - EXPECTED_SCHEMAS)}"
        )
    documents = {path.name: _load(path) for path in paths}
    ids: set[str] = set()
    for path in paths:
        document = documents[path.name]
        if document.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise SchemaBundleError(f"wrong_schema_dialect:{path.name}")
        if document.get("$id") != path.name:
            raise SchemaBundleError(f"schema_id_filename_mismatch:{path.name}")
        if document["$id"] in ids:
            raise SchemaBundleError(f"duplicate_schema_id:{document['$id']}")
        ids.add(document["$id"])
        _validate_schema_node(document, label=path.name, documents=documents)

    entries = [
        {"file_name": path.name, "sha256": _file_sha256(path)} for path in paths
    ]
    digest_input = b"".join(
        entry["file_name"].encode("ascii")
        + b"\0"
        + entry["sha256"].encode("ascii")
        + b"\n"
        for entry in entries
    )
    return {
        "bundle_sha256": hashlib.sha256(digest_input).hexdigest(),
        "schema_count": len(entries),
        "schemas": entries,
        "status": "passed",
        "validator": "stdlib-hidden-schema-bundle-validator-0.4",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "schemas",
    )
    args = parser.parse_args()
    result = validate_bundle(args.schema_dir.resolve())
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
