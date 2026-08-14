#!/usr/bin/env python3
"""Build and validate the molecule-complete safe manifest after v1 P0."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import tarfile
import time
from typing import Any, Iterable

from Data.sdf.chemscene_v0_2 import (
    build_chemscene_v0_2,
    sdf_from_chemscene_v0_2,
    validate_chemscene_v0_2_roundtrip,
)
from chemvector.evaluation.audits.graph.audit_v1p0_edge_centric_oracle10k import (
    DATA_ROOT,
    SHARDS,
    SHARD_SHA256,
    _load_json,
    validate_object_invariants,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


AUDIT_NAME = "chemvector-v1p0r-edge-centric-safe-manifest"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
P0_OUTPUT = Path("outputs/audits/chemvector-v1p0-edge-centric-oracle10k")
P0_RESULT = P0_OUTPUT / "result.json"
P0_RESULT_SHA256 = (
    "2061c375971eac10cee49018d1fffedf18d66e57a41e361833b09139ceeaf143"
)
P0_DIAGNOSTICS = P0_OUTPUT / "skip_diagnostics.json"
P0_DIAGNOSTICS_SHA256 = (
    "86268c54faf905d5e359befab45da0ab25da51a73b00441335d6c1eefb241ceb"
)
VARIANTS = ("original", "rdkit", "coordgen", "stress")
MEMBER_SUFFIXES = (
    ".png",
    ".text.png",
    ".bond.png",
    ".semantic.png",
    ".layers.json",
    ".chemmol.json",
    ".render.json",
    ".meta.json",
)
EXPECTED_INPUT_SAMPLES = 10_000
EXPECTED_INPUT_MOLECULES = 2_500
EXPECTED_EXCLUDED_MOLECULES = 38
EXPECTED_SAFE_MOLECULES = 2_462
EXPECTED_SAFE_SAMPLES = 9_848


def partition_complete_molecule_rows(
    rows: Iterable[dict[str, Any]], excluded_molecules: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exclude complete four-variant molecule groups without partial retention."""

    by_molecule: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        molecule_id = str(row["molecule_id"])
        variant = str(row["variant"])
        if variant not in VARIANTS:
            raise ValueError(f"Unknown ChemVision variant: {variant!r}")
        if variant in by_molecule[molecule_id]:
            raise ValueError(f"Duplicate {variant} row for {molecule_id}")
        by_molecule[molecule_id][variant] = row

    for molecule_id, variants in by_molecule.items():
        if set(variants) != set(VARIANTS):
            raise ValueError(f"Incomplete four-variant molecule: {molecule_id}")
    missing_exclusions = excluded_molecules - set(by_molecule)
    if missing_exclusions:
        raise ValueError(f"Excluded molecules are absent: {sorted(missing_exclusions)}")

    safe: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for molecule_id in sorted(by_molecule):
        destination = excluded if molecule_id in excluded_molecules else safe
        destination.extend(by_molecule[molecule_id][variant] for variant in VARIANTS)
    safe.sort(key=lambda row: str(row["sample_key"]))
    excluded.sort(key=lambda row: str(row["sample_key"]))
    return safe, excluded


def _read_rows() -> list[dict[str, Any]]:
    rows = []
    for shard in SHARDS:
        relative_shard = shard.relative_to(DATA_ROOT).as_posix()
        with tarfile.open(shard) as archive:
            names = set(archive.getnames())
            keys = sorted(
                name.removesuffix(".meta.json")
                for name in names
                if name.endswith(".meta.json")
            )
            for key in keys:
                required = [key + suffix for suffix in MEMBER_SUFFIXES]
                missing = [name for name in required if name not in names]
                if missing:
                    raise ValueError(f"Missing source members for {key}: {missing}")
                meta = _load_json(archive, key + ".meta.json")
                render = _load_json(archive, key + ".render.json")
                if meta["sample_key"] != key:
                    raise ValueError(f"Meta sample key differs: {key}")
                if meta["molecule_id"] != render["molecule_id"]:
                    raise ValueError(f"Meta/render molecule differs: {key}")
                if meta["variant"] != render["variant"]:
                    raise ValueError(f"Meta/render variant differs: {key}")
                if meta["output_shard"] != relative_shard:
                    raise ValueError(f"Meta output shard differs: {key}")
                rows.append(
                    {
                        "sample_key": key,
                        "molecule_id": str(meta["molecule_id"]),
                        "split": str(meta["split"]),
                        "variant": str(meta["variant"]),
                        "stress_operation": render.get("render_parameters", {}).get(
                            "stress_operation"
                        ),
                        "source_shard": relative_shard,
                        "members": {
                            suffix.removeprefix("."): key + suffix
                            for suffix in MEMBER_SUFFIXES
                        },
                    }
                )
    if len({str(row["sample_key"]) for row in rows}) != len(rows):
        raise ValueError("ChemVision pilot contains duplicate sample keys")
    return rows


def _validate_safe_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_shard[str(row["source_shard"])].append(row)

    counts: Counter[str] = Counter()
    object_totals: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    for relative_shard, shard_rows in sorted(by_shard.items()):
        with tarfile.open(DATA_ROOT / relative_shard) as archive:
            for row in shard_rows:
                key = str(row["sample_key"])
                chemmol, render, layers = (
                    _load_json(archive, key + suffix)
                    for suffix in (".chemmol.json", ".render.json", ".layers.json")
                )
                scene = build_chemscene_v0_2(chemmol, render, layers)
                object_totals.update(
                    validate_object_invariants(chemmol, layers, scene)
                )
                validate_chemscene_v0_2_roundtrip(
                    scene,
                    chemmol["canonical_isomeric_smiles"],
                    chemmol["inchikey"],
                )
                sdf = sdf_from_chemscene_v0_2(scene)
                if "M  END" not in sdf or not sdf.rstrip().endswith("$$$$"):
                    raise ValueError(f"Safe manifest SDF framing differs: {key}")
                counts[str(row["split"])] += 1
                variants[str(row["variant"])] += 1
                if row["stress_operation"] is not None:
                    operations[str(row["stress_operation"])] += 1
                counts["passed"] += 1
    return {
        "counts": dict(sorted(counts.items())),
        "object_totals": dict(sorted(object_totals.items())),
        "variants": dict(sorted(variants.items())),
        "stress_operations": dict(sorted(operations.items())),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"v1 P0-R output is not empty: {OUTPUT}")
    if file_sha256(P0_RESULT) != P0_RESULT_SHA256:
        raise ValueError("v1 P0 result SHA256 differs")
    if file_sha256(P0_DIAGNOSTICS) != P0_DIAGNOSTICS_SHA256:
        raise ValueError("v1 P0 skip diagnostics SHA256 differs")
    for shard in SHARDS:
        if file_sha256(shard) != SHARD_SHA256[shard.name]:
            raise ValueError(f"v1 P0-R source shard SHA256 differs: {shard}")

    p0_result = json.loads(P0_RESULT.read_text(encoding="utf-8"))
    diagnostics = json.loads(P0_DIAGNOSTICS.read_text(encoding="utf-8"))
    if p0_result["decision"] != "edge_centric_oracle10k_no_go":
        raise ValueError("v1 P0 must remain NO-GO")
    if diagnostics["counts"].get("failed", 0) != 0:
        raise ValueError("v1 P0 contains unexpected failures")
    invalid_molecules = {
        str(row["molecule_id"]) for row in diagnostics["skips"]
    }
    skipped_keys = {str(row["sample_key"]) for row in diagnostics["skips"]}

    started = time.perf_counter()
    all_rows = _read_rows()
    safe_rows, excluded_rows = partition_complete_molecule_rows(
        all_rows, invalid_molecules
    )
    excluded_keys = {str(row["sample_key"]) for row in excluded_rows}
    validation = _validate_safe_rows(safe_rows)
    safe_molecules = {str(row["molecule_id"]) for row in safe_rows}
    checks = {
        "input_samples_exact10k": len(all_rows) == EXPECTED_INPUT_SAMPLES,
        "input_molecules_exact2500": len({row["molecule_id"] for row in all_rows})
        == EXPECTED_INPUT_MOLECULES,
        "excluded_molecules_exact38": len(invalid_molecules)
        == EXPECTED_EXCLUDED_MOLECULES,
        "all_skips_excluded": skipped_keys <= excluded_keys,
        "molecule_level_exclusion_exact": len(excluded_rows)
        == 4 * len(invalid_molecules),
        "safe_molecules_exact2462": len(safe_molecules) == EXPECTED_SAFE_MOLECULES,
        "safe_samples_exact9848": len(safe_rows) == EXPECTED_SAFE_SAMPLES,
        "safe_oracle_coverage_100pct": validation["counts"]["passed"]
        == len(safe_rows),
        "safe_variants_complete": set(validation["variants"].values())
        == {EXPECTED_SAFE_MOLECULES}
        and set(validation["variants"]) == set(VARIANTS),
        "safe_endpoint_ownership_exact": validation["object_totals"]["ports"]
        == 2 * validation["object_totals"]["bonds"],
        "no_png_generated": True,
        "p0_remains_no_go": p0_result["decision"]
        == "edge_centric_oracle10k_no_go",
    }
    decision = (
        "edge_centric_safe_manifest_go"
        if all(checks.values())
        else "edge_centric_safe_manifest_no_go"
    )

    OUTPUT.mkdir(parents=True, exist_ok=False)
    manifest_path = OUTPUT / "manifest.jsonl"
    _write_jsonl(manifest_path, safe_rows)
    result = {
        "schema_version": "chemvector-v1p0r-edge-centric-safe-manifest-0.1",
        "audit_name": AUDIT_NAME,
        "decision": decision,
        "checks": checks,
        "input": {
            "p0_result_sha256": P0_RESULT_SHA256,
            "p0_skip_diagnostics_sha256": P0_DIAGNOSTICS_SHA256,
            "source_shards": {
                shard.name: SHARD_SHA256[shard.name] for shard in SHARDS
            },
        },
        "counts": {
            "input_samples": len(all_rows),
            "input_molecules": len({row["molecule_id"] for row in all_rows}),
            "excluded_samples": len(excluded_rows),
            "excluded_molecules": len(invalid_molecules),
            "safe_samples": len(safe_rows),
            "safe_molecules": len(safe_molecules),
        },
        "validation": validation,
        "manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "rows": len(safe_rows),
        },
        "policy": {
            "exclusion_unit": "molecule_all_four_variants",
            "replacement_or_rerender": False,
            "generated_png": False,
            "model_or_threshold_used_for_exclusion": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
