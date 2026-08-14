"""Run the finite micro-scene collision audit and emit machine-readable reports."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import inspect
import json
from pathlib import Path
import time
from typing import Any, Iterable

from .finite_geometry_grammar_v0_1 import (
    ADVERSARIAL_INVALID,
    CROSSING_ALLOWED,
    DOMAIN_LAYERS,
    GRAMMAR_LAYERS,
    SIMPLE_LINE_ANGLE_VALID,
    classify_geometry,
)
from .finite_micro_raster_v0_1 import (
    observation_for_scene,
    preprocessing_source_audit,
    production_tensor_digest,
)
from .finite_micro_scene_v0_1 import (
    EXPECTED_CORE_SCENES,
    GRID_EDGES,
    SCHEMA_VERSION,
    Scene,
    canonical_graph_signature,
    crossing_junction_fixtures,
    double_parallel_fixtures,
    fixture_scenes,
    iter_core_scenes,
    relabel_scene,
    scene_record,
)


AUDIT_SCHEMA_VERSION = "finite-micro-scene-audit-0.2"
MANIFEST_SCHEMA_VERSION = "finite-micro-scene-manifest-0.1"
EXPECTED_FULL_LAYER_SCENE_COUNTS = {
    SIMPLE_LINE_ANGLE_VALID: 19_824,
    CROSSING_ALLOWED: 16_833,
    ADVERSARIAL_INVALID: 30_059,
}
EXPECTED_FULL_DOMAIN_COUNTS = {
    "simple_domain": {
        "scene_count": 19_824,
        "observation_count": 19_823,
        "unique_graph_observation_count": 19_822,
        "unique_graph_scene_count": 19_822,
        "different_graph_collision_class_count": 1,
        "different_graph_collision_scene_count": 2,
    },
    "crossing_domain": {
        "scene_count": 36_657,
        "observation_count": 36_649,
        "unique_graph_observation_count": 36_641,
        "unique_graph_scene_count": 36_641,
        "different_graph_collision_class_count": 8,
        "different_graph_collision_scene_count": 16,
    },
    "all_domain": {
        "scene_count": 66_716,
        "observation_count": 53_206,
        "unique_graph_observation_count": 49_688,
        "unique_graph_scene_count": 49_688,
        "different_graph_collision_class_count": 3_518,
        "different_graph_collision_scene_count": 17_028,
    },
}


@dataclass(slots=True)
class GraphWitness:
    signature: str
    scene_id: str
    family: str
    scene: dict[str, Any]
    grammar_layer: str
    grammar_reasons: tuple[str, ...]


@dataclass(slots=True)
class ObservationBucket:
    observation_sha256: str
    scene_count: int = 0
    raw_raster_digests: set[str] = field(default_factory=set)
    witnesses: dict[tuple[str, str], GraphWitness] = field(default_factory=dict)
    layer_scene_counts: Counter[str] = field(default_factory=Counter)
    layer_graph_signatures: dict[str, set[str]] = field(default_factory=dict)

    def add(
        self,
        scene: Scene,
        raw_digest: str,
        graph_signature: str,
        grammar_layer: str,
        grammar_reasons: tuple[str, ...],
    ) -> None:
        self.scene_count += 1
        self.raw_raster_digests.add(raw_digest)
        self.layer_scene_counts[grammar_layer] += 1
        self.layer_graph_signatures.setdefault(grammar_layer, set()).add(
            graph_signature
        )
        self.witnesses.setdefault(
            (graph_signature, grammar_layer),
            GraphWitness(
                graph_signature,
                scene.scene_id,
                scene.family,
                scene_record(scene),
                grammar_layer,
                grammar_reasons,
            ),
        )

    @property
    def graph_signatures(self) -> set[str]:
        return {witness.signature for witness in self.witnesses.values()}


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    data = _json_bytes(value)
    path.write_bytes(data)
    return _sha256(data)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as output:
        for row in rows:
            data = (
                json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
                + "\n"
            ).encode("utf-8")
            output.write(data)
            digest.update(data)
    return digest.hexdigest()


def _generator_source_audit() -> dict[str, str]:
    audit_root = Path(__file__).resolve().parent
    names = (
        "finite_geometry_grammar_v0_1.py",
        "finite_micro_scene_v0_1.py",
        "finite_micro_raster_v0_1.py",
        "run_finite_micro_scene_v0_1.py",
    )
    return {name: _sha256((audit_root / name).read_bytes()) for name in names}


def _domain_graph_signatures(
    bucket: ObservationBucket,
    allowed_layers: frozenset[str],
) -> set[str]:
    signatures: set[str] = set()
    for layer in allowed_layers:
        signatures.update(bucket.layer_graph_signatures.get(layer, set()))
    return signatures


def _collision_class(
    bucket: ObservationBucket,
    allowed_layers: frozenset[str],
) -> str:
    active_layers = [
        layer for layer in sorted(allowed_layers) if bucket.layer_scene_counts[layer]
    ]
    if not active_layers:
        return "excluded"
    signatures = _domain_graph_signatures(bucket, allowed_layers)
    if len(signatures) <= 1:
        return "unique_graph"
    within_layer_collision = any(
        len(bucket.layer_graph_signatures.get(layer, set())) > 1
        for layer in active_layers
    )
    if within_layer_collision and len(active_layers) > 1:
        return "mixed_layer_graph_collision"
    if within_layer_collision:
        return "same_layer_graph_collision"
    return "cross_layer_graph_collision"


def _scene_domain_collision_class(
    bucket: ObservationBucket,
    scene_layer: str,
    allowed_layers: frozenset[str],
) -> str:
    if scene_layer not in allowed_layers:
        return "excluded"
    return _collision_class(bucket, allowed_layers)


def _exclusive_layer_statistics(
    buckets: dict[str, ObservationBucket],
    layer: str,
) -> dict[str, int]:
    current = [bucket for bucket in buckets.values() if bucket.layer_scene_counts[layer]]
    colliding = [
        bucket
        for bucket in current
        if len(bucket.layer_graph_signatures.get(layer, set())) > 1
    ]
    return {
        "scene_count": sum(bucket.layer_scene_counts[layer] for bucket in current),
        "observation_count": len(current),
        "different_graph_collision_class_count": len(colliding),
        "different_graph_collision_scene_count": sum(
            bucket.layer_scene_counts[layer] for bucket in colliding
        ),
    }


def _domain_statistics(
    buckets: dict[str, ObservationBucket],
    allowed_layers: frozenset[str],
) -> dict[str, int]:
    current = [
        bucket
        for bucket in buckets.values()
        if any(bucket.layer_scene_counts[layer] for layer in allowed_layers)
    ]
    colliding = [
        bucket
        for bucket in current
        if len(_domain_graph_signatures(bucket, allowed_layers)) > 1
    ]
    colliding_digests = {bucket.observation_sha256 for bucket in colliding}
    unique = [
        bucket
        for bucket in current
        if bucket.observation_sha256 not in colliding_digests
    ]
    return {
        "scene_count": sum(
            sum(bucket.layer_scene_counts[layer] for layer in allowed_layers)
            for bucket in current
        ),
        "observation_count": len(current),
        "unique_graph_observation_count": len(unique),
        "unique_graph_scene_count": sum(
            sum(bucket.layer_scene_counts[layer] for layer in allowed_layers)
            for bucket in unique
        ),
        "different_graph_collision_class_count": len(colliding),
        "different_graph_collision_scene_count": sum(
            sum(bucket.layer_scene_counts[layer] for layer in allowed_layers)
            for bucket in colliding
        ),
    }


def _prior_resolved_counts(
    buckets: dict[str, ObservationBucket],
) -> dict[str, int]:
    simple = DOMAIN_LAYERS["simple_domain"]
    crossing = DOMAIN_LAYERS["crossing_domain"]
    all_layers = DOMAIN_LAYERS["all_domain"]

    def resolved(broader: frozenset[str], narrower: frozenset[str]) -> int:
        return sum(
            _collision_class(bucket, broader) not in {"excluded", "unique_graph"}
            and _collision_class(bucket, narrower) == "unique_graph"
            for bucket in buckets.values()
        )

    return {
        "all_to_crossing_domain": resolved(all_layers, crossing),
        "crossing_to_simple_domain": resolved(crossing, simple),
        "all_to_simple_domain": resolved(all_layers, simple),
    }


def _fixture_result(left: Scene, right: Scene, *, image_is_prepared: bool) -> dict[str, Any]:
    left_observation = observation_for_scene(left, image_is_prepared=image_is_prepared)
    right_observation = observation_for_scene(right, image_is_prepared=image_is_prepared)
    left_graph = canonical_graph_signature(left)
    right_graph = canonical_graph_signature(right)
    return {
        "left_scene_id": left.scene_id,
        "right_scene_id": right.scene_id,
        "raw_raster_equal": (
            left_observation["raw_raster_sha256"]
            == right_observation["raw_raster_sha256"]
        ),
        "production_observation_equal": (
            left_observation["production_tensor_sha256"]
            == right_observation["production_tensor_sha256"]
        ),
        "graph_signatures_different": left_graph != right_graph,
        "left_graph_signature": left_graph,
        "right_graph_signature": right_graph,
        "observation_sha256": left_observation["production_tensor_sha256"],
    }


def _runtime_invariance_checks() -> dict[str, bool]:
    crossing, junction = crossing_junction_fixtures()
    replay_digests = [
        observation_for_scene(scene)["production_tensor_sha256"]
        for scene in fixture_scenes()
        for _ in range(3)
    ]
    replay_groups = [replay_digests[index : index + 3] for index in range(0, 12, 3)]
    atom_ids = [node.atom_id for node in junction.nodes]
    mapping = dict(zip(atom_ids, reversed([value + 100 for value in atom_ids])))
    relabeled = relabel_scene(junction, mapping, reverse_bonds=True)
    prepared_equivalent = all(
        observation_for_scene(scene)["production_tensor_sha256"]
        == observation_for_scene(scene, image_is_prepared=True)[
            "production_tensor_sha256"
        ]
        for scene in (crossing, junction)
    )
    digest_parameters = set(inspect.signature(production_tensor_digest).parameters)
    return {
        "fixture_three_replays_are_identical": all(
            len(set(group)) == 1 for group in replay_groups
        ),
        "graph_signature_ignores_atom_ids_and_bond_order": (
            canonical_graph_signature(junction)
            == canonical_graph_signature(relabeled)
        ),
        "prepared_path_matches_default_path": prepared_equivalent,
        "production_digest_accepts_image_only": digest_parameters
        == {"image", "image_is_prepared"},
    }


def _iter_audit_scenes(core_limit: int | None) -> Iterable[tuple[str, Scene]]:
    for scene in iter_core_scenes(core_limit):
        yield "core", scene
    for scene in fixture_scenes():
        yield "fixture", scene


def run_audit(
    output: Path,
    *,
    core_limit: int | None = None,
    image_is_prepared: bool = False,
) -> dict[str, Any]:
    if core_limit is not None and core_limit < 0:
        raise ValueError("Core limit must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    buckets: dict[str, ObservationBucket] = {}
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    grammar_reason_counts: Counter[str] = Counter()
    manifest_rows: list[dict[str, Any]] = []
    tensor_header = None

    for source, scene in _iter_audit_scenes(core_limit):
        observation = observation_for_scene(scene, image_is_prepared=image_is_prepared)
        tensor_header = observation["tensor_header"]
        graph_signature = canonical_graph_signature(scene)
        structure_signature = canonical_graph_signature(scene, erase_text=True)
        grammar = classify_geometry(scene)
        digest = observation["production_tensor_sha256"]
        bucket = buckets.setdefault(digest, ObservationBucket(digest))
        bucket.add(
            scene,
            observation["raw_raster_sha256"],
            graph_signature,
            grammar.layer,
            grammar.reasons,
        )
        manifest_rows.append(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "scene_id": scene.scene_id,
                "source": source,
                "raw_raster_sha256": observation["raw_raster_sha256"],
                "production_observation_sha256": digest,
                "canonical_graph_signature_full": graph_signature,
                "canonical_graph_signature_structure": structure_signature,
                "exclusive_grammar_layer": grammar.layer,
                "grammar_reasons": list(grammar.reasons),
                "grammar_scope": "solid_lane_v0_1",
                "carrier_status": "not_in_current_grammar",
                "active_prior_id": None,
            }
        )
        family_counts[scene.family] += 1
        source_counts[source] += 1
        grammar_reason_counts.update(grammar.reasons)

    collision_certificates = []
    graph_collision_scene_count = 0
    preprocessing_collision_classes = 0
    for digest, bucket in sorted(buckets.items()):
        if len(bucket.graph_signatures) <= 1:
            continue
        graph_collision_scene_count += bucket.scene_count
        preprocessing_collision = len(bucket.raw_raster_digests) > 1
        preprocessing_collision_classes += int(preprocessing_collision)
        witnesses = sorted(
            (
                {
                    "family": witness.family,
                    "graph_signature": witness.signature,
                    "grammar_layer": witness.grammar_layer,
                    "grammar_reasons": list(witness.grammar_reasons),
                    "scene": witness.scene,
                    "scene_id": witness.scene_id,
                }
                for witness in bucket.witnesses.values()
            ),
            key=lambda item: (item["graph_signature"], item["scene_id"]),
        )
        collision_certificates.append(
            {
                "classification": (
                    "preprocess_graph_collision"
                    if preprocessing_collision
                    else "opaque_graph_collision"
                ),
                "global_collision_class": _collision_class(
                    bucket, DOMAIN_LAYERS["all_domain"]
                ),
                "domain_collision_classes": {
                    domain: _collision_class(bucket, allowed_layers)
                    for domain, allowed_layers in sorted(DOMAIN_LAYERS.items())
                },
                "graph_signature_count": len(bucket.graph_signatures),
                "observation_sha256": digest,
                "raw_raster_sha256": sorted(bucket.raw_raster_digests),
                "scene_count": bucket.scene_count,
                "witnesses": witnesses,
            }
        )

    for row in manifest_rows:
        bucket = buckets[row["production_observation_sha256"]]
        global_class = _collision_class(
            bucket, DOMAIN_LAYERS["all_domain"]
        )
        row["collision_origin"] = (
            "none"
            if global_class == "unique_graph"
            else (
                "preprocess" if len(bucket.raw_raster_digests) > 1 else "opaque"
            )
        )
        row["global_collision_class"] = global_class
        for domain, allowed_layers in sorted(DOMAIN_LAYERS.items()):
            row[f"{domain}_collision_class"] = _scene_domain_collision_class(
                bucket,
                row["exclusive_grammar_layer"],
                allowed_layers,
            )
        row["single_answer_candidate"] = (
            row["exclusive_grammar_layer"] == SIMPLE_LINE_ANGLE_VALID
            and row["simple_domain_collision_class"] == "unique_graph"
        )

    crossing, junction = crossing_junction_fixtures()
    double, parallel = double_parallel_fixtures()
    fixture_results = {
        "crossing_vs_junction": _fixture_result(
            crossing, junction, image_is_prepared=image_is_prepared
        ),
        "double_vs_parallel_singles": _fixture_result(
            double, parallel, image_is_prepared=image_is_prepared
        ),
    }
    fixture_layers = {
        scene.scene_id: classify_geometry(scene).layer
        for scene in (crossing, junction, double, parallel)
    }
    fixture_checks_passed = all(
        result["raw_raster_equal"]
        and result["production_observation_equal"]
        and result["graph_signatures_different"]
        for result in fixture_results.values()
    )
    processed_core = source_counts["core"]
    full_run = core_limit is None or core_limit >= EXPECTED_CORE_SCENES
    expected_processed_core = EXPECTED_CORE_SCENES if full_run else core_limit
    scene_ids = [row["scene_id"] for row in manifest_rows]
    layer_scene_counts = Counter(
        row["exclusive_grammar_layer"] for row in manifest_rows
    )
    domain_statistics = {
        domain: _domain_statistics(buckets, allowed_layers)
        for domain, allowed_layers in sorted(DOMAIN_LAYERS.items())
    }
    maximum_graph_signatures = max(
        (len(bucket.graph_signatures) for bucket in buckets.values()),
        default=0,
    )
    invalid_single_answer_candidates = sum(
        row["single_answer_candidate"]
        and row["exclusive_grammar_layer"] == ADVERSARIAL_INVALID
        for row in manifest_rows
    )
    manifest_domain_classes_match_scope = all(
        row[f"{domain}_collision_class"]
        == _scene_domain_collision_class(
            buckets[row["production_observation_sha256"]],
            row["exclusive_grammar_layer"],
            allowed_layers,
        )
        for row in manifest_rows
        for domain, allowed_layers in DOMAIN_LAYERS.items()
    )
    checks = {
        "core_edge_catalog_is_36": len(GRID_EDGES) == 36,
        "core_formula_is_66712": EXPECTED_CORE_SCENES == 66_712,
        "processed_expected_core_count": processed_core == expected_processed_core,
        "fixture_collisions_match_lean": fixture_checks_passed,
        "fixture_grammar_layers_match_preregistered_rules": fixture_layers
        == {
            "fixture-crossing": CROSSING_ALLOWED,
            "fixture-junction": SIMPLE_LINE_ANGLE_VALID,
            "fixture-double-bond": SIMPLE_LINE_ANGLE_VALID,
            "fixture-parallel-singles": SIMPLE_LINE_ANGLE_VALID,
        },
        "manifest_covers_every_processed_scene": len(manifest_rows)
        == sum(source_counts.values()),
        "manifest_scene_ids_are_unique": len(scene_ids) == len(set(scene_ids)),
        "manifest_domain_collision_classes_match_scene_scope": (
            manifest_domain_classes_match_scope
        ),
        "every_scene_has_exactly_one_known_layer": sum(layer_scene_counts.values())
        == len(manifest_rows)
        and set(layer_scene_counts).issubset(GRAMMAR_LAYERS),
        "adversarial_invalid_single_answer_candidate_count_is_zero": (
            invalid_single_answer_candidates == 0
        ),
        "grammar_metadata_not_used_by_production_digest": fixture_results[
            "crossing_vs_junction"
        ]["production_observation_equal"],
        "preprocessing_source_recorded": bool(preprocessing_source_audit()["source_sha256"]),
        **_runtime_invariance_checks(),
    }
    full_reproduction_checks = {
        "processed_total_scene_count_is_66716": len(manifest_rows) == 66_716,
        "exclusive_layer_counts_match_frozen_v0_1": dict(layer_scene_counts)
        == EXPECTED_FULL_LAYER_SCENE_COUNTS,
        "cumulative_domain_counts_match_frozen_v0_1": domain_statistics
        == EXPECTED_FULL_DOMAIN_COUNTS,
        "all_domain_collision_class_count_is_3518": len(collision_certificates)
        == 3_518,
        "all_domain_collision_scene_count_is_17028": graph_collision_scene_count
        == 17_028,
        "maximum_graph_signatures_per_observation_is_9": (
            maximum_graph_signatures == 9
        ),
        "preprocessing_collision_class_count_is_zero": (
            preprocessing_collision_classes == 0
        ),
    }
    if full_run:
        checks.update(full_reproduction_checks)
    validation_status = (
        "passed" if full_run and all(checks.values()) else "passed_smoke"
    )
    if not all(checks.values()):
        validation_status = "failed"

    elapsed = time.monotonic() - started
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "scene_schema_version": SCHEMA_VERSION,
        "mode": "full" if full_run else "smoke",
        "image_is_prepared": image_is_prepared,
        "expected_core_scene_count": EXPECTED_CORE_SCENES,
        "processed_core_scene_count": processed_core,
        "processed_fixture_scene_count": source_counts["fixture"],
        "processed_total_scene_count": sum(source_counts.values()),
        "family_counts": dict(sorted(family_counts.items())),
        "unique_observation_count": len(buckets),
        "graph_collision_class_count": len(collision_certificates),
        "graph_collision_scene_count": graph_collision_scene_count,
        "maximum_graph_signatures_per_observation": maximum_graph_signatures,
        "preprocessing_collision_class_count": preprocessing_collision_classes,
        "graph_strict_observation_count": len(buckets) - len(collision_certificates),
        "fixture_results": fixture_results,
        "fixture_grammar_layers": fixture_layers,
        "grammar_reason_counts": dict(sorted(grammar_reason_counts.items())),
        "layer_conservation": {
            "classified_scene_count": sum(layer_scene_counts.values()),
            "duplicate_scene_id_count": len(scene_ids) - len(set(scene_ids)),
            "unclassified_scene_count": len(manifest_rows)
            - sum(layer_scene_counts.values()),
        },
        "exclusive_grammar_layers": {
            layer: _exclusive_layer_statistics(buckets, layer)
            for layer in GRAMMAR_LAYERS
        },
        "cumulative_domains": domain_statistics,
        "prior_resolved_observation_counts": _prior_resolved_counts(buckets),
        "grammar_capability_status": {
            "carrier_analysis": "not_in_current_grammar",
            "dash_and_wedge": "rejected_by_solid_lane_mapping",
            "exclusive_layer_model_input": "forbidden",
            "maximum_crossing_owners": 2,
            "semantic_multiplicity": "audited_by_observation_collision",
        },
    }
    collisions_payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "certificate_count": len(collision_certificates),
        "certificates": collision_certificates,
    }
    summary_sha256 = _write_json(output / "summary.json", summary)
    collisions_sha256 = _write_json(output / "collision_certificates.json", collisions_payload)
    manifest_sha256 = _write_jsonl(output / "scene_manifest.jsonl", manifest_rows)
    validation = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": (
            "passed_full_audit"
            if validation_status == "passed"
            else (
                "passed_smoke_audit"
                if validation_status == "passed_smoke"
                else "failed"
            )
        ),
        "training_authorized": False,
        "identifiability_gate_status": (
            "go"
            if domain_statistics["simple_domain"][
                "different_graph_collision_class_count"
            ]
            == 0
            else "no_go"
        ),
        "identifiability_gate_reason": (
            "simple_domain_has_no_same_observation_different_graph_class"
            if domain_statistics["simple_domain"][
                "different_graph_collision_class_count"
            ]
            == 0
            else "simple_domain_contains_same_observation_different_graph_classes"
        ),
        "checks": checks,
        "generator_sources": _generator_source_audit(),
        "preprocessing": preprocessing_source_audit(),
        "tensor_header": tensor_header,
        "artifacts": {
            "summary.json": summary_sha256,
            "collision_certificates.json": collisions_sha256,
            "scene_manifest.jsonl": manifest_sha256,
        },
    }
    _write_json(output / "validation.json", validation)
    _write_json(
        output / "runtime.json",
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "elapsed_seconds": round(elapsed, 6),
            "note": "Non-commitment runtime metadata; excluded from artifact hashes.",
        },
    )
    return {
        "summary": summary,
        "validation": validation,
        "collisions": collisions_payload,
        "manifest_rows": manifest_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("formal/chemgraph/audit/results/finite_micro_scene_v0_1"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N core scenes; all four fixtures still run.",
    )
    parser.add_argument(
        "--image-is-prepared",
        action="store_true",
        help="Prepare the raster before hashing, then use the equivalent prepared-image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(
        args.output,
        core_limit=args.limit,
        image_is_prepared=args.image_is_prepared,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
