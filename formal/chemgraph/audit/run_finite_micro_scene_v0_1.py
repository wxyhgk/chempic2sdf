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


AUDIT_SCHEMA_VERSION = "finite-micro-scene-audit-0.1"


@dataclass(slots=True)
class GraphWitness:
    signature: str
    scene_id: str
    family: str
    scene: dict[str, Any]


@dataclass(slots=True)
class ObservationBucket:
    observation_sha256: str
    scene_count: int = 0
    raw_raster_digests: set[str] = field(default_factory=set)
    witnesses: dict[str, GraphWitness] = field(default_factory=dict)

    def add(self, scene: Scene, raw_digest: str, graph_signature: str) -> None:
        self.scene_count += 1
        self.raw_raster_digests.add(raw_digest)
        self.witnesses.setdefault(
            graph_signature,
            GraphWitness(
                graph_signature,
                scene.scene_id,
                scene.family,
                scene_record(scene),
            ),
        )


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


def _generator_source_audit() -> dict[str, str]:
    audit_root = Path(__file__).resolve().parent
    names = (
        "finite_micro_scene_v0_1.py",
        "finite_micro_raster_v0_1.py",
        "run_finite_micro_scene_v0_1.py",
    )
    return {name: _sha256((audit_root / name).read_bytes()) for name in names}


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
    tensor_header = None

    for source, scene in _iter_audit_scenes(core_limit):
        observation = observation_for_scene(scene, image_is_prepared=image_is_prepared)
        tensor_header = observation["tensor_header"]
        graph_signature = canonical_graph_signature(scene)
        digest = observation["production_tensor_sha256"]
        bucket = buckets.setdefault(digest, ObservationBucket(digest))
        bucket.add(scene, observation["raw_raster_sha256"], graph_signature)
        family_counts[scene.family] += 1
        source_counts[source] += 1

    collision_certificates = []
    graph_collision_scene_count = 0
    preprocessing_collision_classes = 0
    for digest, bucket in sorted(buckets.items()):
        if len(bucket.witnesses) <= 1:
            continue
        graph_collision_scene_count += bucket.scene_count
        preprocessing_collision = len(bucket.raw_raster_digests) > 1
        preprocessing_collision_classes += int(preprocessing_collision)
        witnesses = sorted(
            (
                {
                    "family": witness.family,
                    "graph_signature": witness.signature,
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
                "graph_signature_count": len(witnesses),
                "observation_sha256": digest,
                "raw_raster_sha256": sorted(bucket.raw_raster_digests),
                "scene_count": bucket.scene_count,
                "witnesses": witnesses,
            }
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
    fixture_checks_passed = all(
        result["raw_raster_equal"]
        and result["production_observation_equal"]
        and result["graph_signatures_different"]
        for result in fixture_results.values()
    )
    processed_core = source_counts["core"]
    full_run = core_limit is None or core_limit >= EXPECTED_CORE_SCENES
    expected_processed_core = EXPECTED_CORE_SCENES if full_run else core_limit
    checks = {
        "core_edge_catalog_is_36": len(GRID_EDGES) == 36,
        "core_formula_is_66712": EXPECTED_CORE_SCENES == 66_712,
        "processed_expected_core_count": processed_core == expected_processed_core,
        "fixture_collisions_match_lean": fixture_checks_passed,
        "preprocessing_source_recorded": bool(preprocessing_source_audit()["source_sha256"]),
        **_runtime_invariance_checks(),
    }
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
        "maximum_graph_signatures_per_observation": max(
            (len(bucket.witnesses) for bucket in buckets.values()),
            default=0,
        ),
        "preprocessing_collision_class_count": preprocessing_collision_classes,
        "graph_strict_observation_count": len(buckets) - len(collision_certificates),
        "fixture_results": fixture_results,
    }
    collisions_payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "certificate_count": len(collision_certificates),
        "certificates": collision_certificates,
    }
    summary_sha256 = _write_json(output / "summary.json", summary)
    collisions_sha256 = _write_json(output / "collision_certificates.json", collisions_payload)
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
        "checks": checks,
        "generator_sources": _generator_source_audit(),
        "preprocessing": preprocessing_source_audit(),
        "tensor_header": tensor_header,
        "artifacts": {
            "summary.json": summary_sha256,
            "collision_certificates.json": collisions_sha256,
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
    return {"summary": summary, "validation": validation, "collisions": collisions_payload}


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
