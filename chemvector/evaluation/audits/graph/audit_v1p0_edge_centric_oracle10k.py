#!/usr/bin/env python3
"""Freeze the legacy benchmark and validate the v1 edge-centric Oracle protocol."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tarfile
import time
from typing import Any

from Data.sdf.chemscene_v0_1 import ChemSceneDepictionConflictError
from Data.sdf.chemscene_v0_2 import (
    build_chemscene_v0_2,
    parse_chemscene_v0_2,
    sdf_from_chemscene_v0_2,
    validate_chemscene_v0_2_roundtrip,
)
from Data.sdf.chemsvg_sequence_v4 import ChemSVGCoordinateCollisionError
from chemvector.g3_structural_data import file_sha256
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


AUDIT_NAME = "chemvector-v1p0-edge-centric-oracle10k"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
DATA_ROOT = Path("Data/sdf/chemvision_layers_v0_1/pilot_10k")
SHARD_ROOT = DATA_ROOT / "shards"
SHARDS = (
    SHARD_ROOT / "internal_test-000000.tar",
    SHARD_ROOT / "train-000000.tar",
    SHARD_ROOT / "train-000001.tar",
    SHARD_ROOT / "train-000002.tar",
    SHARD_ROOT / "validation-000000.tar",
)
SHARD_SHA256 = {
    "internal_test-000000.tar": "6837b8238c17a3d3de5e61dde76643fa74cc914c614865d85e5518baae8e443d",
    "train-000000.tar": "6e5a21d14cd0d1521de44f67eaea111e4c0272d7b7652cb1e88c13f82443b042",
    "train-000001.tar": "9ca92e15d260d4372135177990e1f9fdeca771d8e78cbcd7ba2c735128035aea",
    "train-000002.tar": "5d9eba5abe8277368b68ec9bdd6fc3c4abeaaf339acd0874e00a815d95311aa7",
    "validation-000000.tar": "ac26505c3c1b8835fab3a5e86473541b30c3f065f28b1ba9f32c63239854d4c1",
}
DATA_SUMMARY = DATA_ROOT / "summary.json"
DATA_SUMMARY_SHA256 = (
    "0ca09a9759bfd902e265555e07592b71f87e95a464fc5ceaa21e4257031ef8f7"
)
DATA_VALIDATION = DATA_ROOT / "validation.json"
DATA_VALIDATION_SHA256 = (
    "51f4c703043266107e3e48f4225668535f3fc10169d0c62bee196690ab7f3255"
)
BASELINE_ROOT = Path(
    "outputs/evaluations/mr-tadf-v1-v01021-invalid-only-ocr-fallback-live"
)
BASELINE_MANIFEST = Path("Data/benchmarks/mr_tadf_v1/manifest.csv")
BASELINE_MANIFEST_SHA256 = (
    "19f28be1386ade7ef47932a18e97759d79e21c506c5cfce273c795bff665ccad"
)
BASELINE_SUMMARY = BASELINE_ROOT / "summary.json"
BASELINE_SUMMARY_SHA256 = (
    "b5d10e1bd10ab96b6a0bdf328ca59794cebed2d80ec7bb95fd802efd1fcd4661"
)
BASELINE_OBJECTS = BASELINE_ROOT / "object_decomposition.json"
BASELINE_OBJECTS_SHA256 = (
    "0a4e2033a71c5ac56c093bbe5dc699ee2c0f8e554ff31323f1f4682cddb2de4c"
)
BASELINE_RECORDS = BASELINE_ROOT / "records.jsonl"
BASELINE_RECORDS_SHA256 = (
    "117c5758d888b62a4143c5ef3e1f1dc77752d221dbfd5a9db1c0fc7fafa1d553"
)
BASELINE_TAG_COMMIT = "dce6f8d9b44880e511a7f8cfc1b6b863e55e7c4c"
EXPECTED_SAMPLES = 10_000
MIN_ORACLE_COVERAGE = 0.995
JSON_SUFFIXES = (".chemmol.json", ".render.json", ".layers.json")


def _load_json(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    handle = archive.extractfile(name)
    if handle is None:
        raise ValueError(f"Missing tar member: {name}")
    return json.load(handle)


def validate_object_invariants(
    chemmol: dict[str, Any], layers: dict[str, Any], scene: dict[str, Any]
) -> dict[str, int]:
    parsed = parse_chemscene_v0_2(scene)
    atom_count = len(chemmol["atoms"])
    bond_count = len(chemmol["bonds"])
    layer_atom_count = len(layers["atoms"])
    layer_bond_count = len(layers["bonds"])
    if not atom_count == layer_atom_count == len(parsed.atom_objects):
        raise ValueError("v1 Oracle AtomObject count differs")
    if not bond_count == layer_bond_count == len(parsed.bond_objects):
        raise ValueError("v1 Oracle BondObject count differs")
    if {int(item["atom_id"]) for item in layers["atoms"]} != set(range(atom_count)):
        raise ValueError("v1 Oracle layer atom IDs are not dense")
    if {int(item["bond_id"]) for item in layers["bonds"]} != set(range(bond_count)):
        raise ValueError("v1 Oracle layer bond IDs are not dense")
    incident_ports = sum(len(atom.incident_ports) for atom in parsed.atom_objects)
    if incident_ports != 2 * bond_count:
        raise ValueError("v1 Oracle endpoint ownership differs")
    text_atoms = sum(int(atom.role == "TEXT") for atom in parsed.atom_objects)
    visible_text = sum(int(item["visible_text"]) for item in layers["atoms"])
    if text_atoms != visible_text:
        raise ValueError("v1 Oracle TextObject count differs")
    scene_display = Counter(bond.display_type for bond in parsed.bond_objects)
    layer_display = Counter(str(item["display_type"]) for item in layers["bonds"])
    if scene_display != layer_display:
        raise ValueError("v1 Oracle display BondObject types differ")
    scene_chemical = Counter(bond.chemical_type for bond in parsed.bond_objects)
    layer_chemical = Counter(str(item["chemical_type"]) for item in layers["bonds"])
    if scene_chemical != layer_chemical:
        raise ValueError("v1 Oracle chemical BondObject types differ")
    return {
        "atoms": atom_count,
        "bonds": bond_count,
        "text_atoms": text_atoms,
        "ports": incident_ports,
        "double_bond_objects": scene_display["DOUBLE"],
        "triple_bond_objects": scene_display["TRIPLE"],
        "aromatic_bond_objects": scene_display["AROMATIC"],
    }


def audit_archive(path: Path, limit: int | None = None) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    stress_operations: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    skips: list[dict[str, str | None]] = []
    with tarfile.open(path) as archive:
        names = set(archive.getnames())
        keys = sorted(
            name[: -len(JSON_SUFFIXES[0])]
            for name in names
            if name.endswith(JSON_SUFFIXES[0])
        )
        if limit is not None:
            keys = keys[:limit]
        for key in keys:
            counts["total"] += 1
            try:
                required = [key + suffix for suffix in JSON_SUFFIXES]
                missing = [name for name in required if name not in names]
                if missing:
                    raise ValueError(f"Missing Oracle members: {missing}")
                chemmol, render, layers = (
                    _load_json(archive, name) for name in required
                )
                scene = build_chemscene_v0_2(chemmol, render, layers)
                current = validate_object_invariants(chemmol, layers, scene)
                validate_chemscene_v0_2_roundtrip(
                    scene,
                    chemmol["canonical_isomeric_smiles"],
                    chemmol["inchikey"],
                )
                sdf = sdf_from_chemscene_v0_2(scene)
                if "M  END" not in sdf or not sdf.rstrip().endswith("$$$$"):
                    raise ValueError("v1 Oracle SDF framing differs")
                totals.update(current)
                variants[str(render["variant"])] += 1
                operation = render.get("render_parameters", {}).get(
                    "stress_operation"
                )
                if operation is not None:
                    stress_operations[str(operation)] += 1
                counts["passed"] += 1
            except ChemSVGCoordinateCollisionError as error:
                counts["skipped_coordinate_collision"] += 1
                skips.append(
                    {
                        "sample_key": key,
                        "molecule_id": str(chemmol["molecule_id"]),
                        "variant": str(render["variant"]),
                        "stress_operation": render.get("render_parameters", {}).get(
                            "stress_operation"
                        ),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            except ChemSceneDepictionConflictError as error:
                counts["skipped_depiction_conflict"] += 1
                skips.append(
                    {
                        "sample_key": key,
                        "molecule_id": str(chemmol["molecule_id"]),
                        "variant": str(render["variant"]),
                        "stress_operation": render.get("render_parameters", {}).get(
                            "stress_operation"
                        ),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            except Exception as error:
                counts["failed"] += 1
                if len(failures) < 100:
                    failures.append(
                        {
                            "sample_key": key,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
    return {
        "archive": str(path),
        "counts": dict(counts),
        "object_totals": dict(totals),
        "variants": dict(sorted(variants.items())),
        "stress_operations": dict(sorted(stress_operations.items())),
        "skips": skips,
        "failures": failures,
        "generated_png": False,
    }


def aggregate_archives(reports: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    failures = []
    skips = []
    for report in reports:
        counts.update(report["counts"])
        totals.update(report["object_totals"])
        variants.update(report["variants"])
        operations.update(report["stress_operations"])
        skips.extend(report.get("skips", []))
        failures.extend(report["failures"])
    total = counts["total"]
    passed = counts["passed"]
    return {
        "counts": dict(counts),
        "coverage": passed / max(total, 1),
        "object_totals": dict(totals),
        "variants": dict(sorted(variants.items())),
        "stress_operations": dict(sorted(operations.items())),
        "skips": skips,
        "failures": failures,
        "generated_png": False,
    }


def load_baseline() -> dict[str, Any]:
    summary = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    objects = json.loads(BASELINE_OBJECTS.read_text(encoding="utf-8"))
    record_count = sum(
        1 for line in BASELINE_RECORDS.read_text(encoding="utf-8").splitlines() if line
    )
    return {
        "tag": "v0.10.21",
        "tag_commit": BASELINE_TAG_COMMIT,
        "manifest_sha256": BASELINE_MANIFEST_SHA256,
        "samples": summary["samples"],
        "records": record_count,
        "sdf_valid": summary["sdf_valid_count"],
        "skeleton_exact": objects["skeleton_exact_count"],
        "atom_typed_exact": objects["atom_typed_skeleton_exact_count"],
        "bond_typed_exact": objects["bond_typed_skeleton_exact_count"],
        "chemical_exact": summary["chemical_match_count"],
        "average_fingerprint_similarity": summary["average_fingerprint_similarity"],
        "summary_sha256": BASELINE_SUMMARY_SHA256,
        "object_decomposition_sha256": BASELINE_OBJECTS_SHA256,
        "records_sha256": BASELINE_RECORDS_SHA256,
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"v1 P0 output is not empty: {OUTPUT}")
    inputs = (
        (DATA_SUMMARY, DATA_SUMMARY_SHA256),
        (DATA_VALIDATION, DATA_VALIDATION_SHA256),
        (BASELINE_MANIFEST, BASELINE_MANIFEST_SHA256),
        (BASELINE_SUMMARY, BASELINE_SUMMARY_SHA256),
        (BASELINE_OBJECTS, BASELINE_OBJECTS_SHA256),
        (BASELINE_RECORDS, BASELINE_RECORDS_SHA256),
        *((path, SHARD_SHA256[path.name]) for path in SHARDS),
    )
    for path, expected in inputs:
        if file_sha256(path) != expected:
            raise ValueError(f"v1 P0 input SHA256 differs: {path}")
    started = time.perf_counter()
    reports = [audit_archive(path) for path in SHARDS]
    oracle = aggregate_archives(reports)
    baseline = load_baseline()
    counts = oracle["counts"]
    checks = {
        "input_samples_exact10k": counts["total"] == EXPECTED_SAMPLES,
        "oracle_coverage_min99_5pct": oracle["coverage"] >= MIN_ORACLE_COVERAGE,
        "oracle_no_unexpected_failure": counts.get("failed", 0) == 0,
        "oracle_all_accepted_roundtrip": counts["passed"]
        == counts["total"]
        - counts.get("skipped_coordinate_collision", 0)
        - counts.get("skipped_depiction_conflict", 0),
        "one_atom_object_per_chemical_atom": oracle["object_totals"]["atoms"] > 0,
        "one_bond_object_per_chemical_bond": oracle["object_totals"]["bonds"] > 0,
        "endpoint_ownership_exact": oracle["object_totals"]["ports"]
        == 2 * oracle["object_totals"]["bonds"],
        "multiple_bond_objects_present": (
            oracle["object_totals"]["double_bond_objects"] > 0
            and oracle["object_totals"]["triple_bond_objects"] > 0
        ),
        "all_four_variants_covered": set(oracle["variants"])
        == {"coordgen", "original", "rdkit", "stress"},
        "all_four_stress_operations_covered": set(oracle["stress_operations"])
        == {"crossed_bonds", "crowded_subgraph", "long_bond", "near_nonbonded"},
        "no_png_generated_for_skips": not oracle["generated_png"],
        "legacy_benchmark_complete": (
            baseline["samples"] == baseline["records"] == 1_100
            and baseline["skeleton_exact"] == 529
            and baseline["chemical_exact"] == 264
        ),
    }
    decision = (
        "edge_centric_oracle10k_go"
        if all(checks.values())
        else "edge_centric_oracle10k_no_go"
    )
    result = {
        "schema_version": "chemvector-v1p0-edge-centric-oracle10k-result-0.1",
        "audit_name": AUDIT_NAME,
        "decision": decision,
        "checks": checks,
        "legacy_benchmark": baseline,
        "oracle": oracle,
        "archive_reports": reports,
        "protocol": {
            "atom_object": "one chemical atom with anchor, optional text, and incident ports",
            "bond_object": "one chemical bond with two owned ports and one display/chemical type",
            "multiple_bond_policy": "double/triple/aromatic remain one BondObject",
            "atom_derivation": "implicit atoms are endpoint equivalence classes; text atoms own boundary ports",
            "generated_png_for_rejected_or_skipped": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
