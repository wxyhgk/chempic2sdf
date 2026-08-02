#!/usr/bin/env python3
"""Build a deterministic, reviewable MR-TADF causal failure audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Callable

from PIL import Image, ImageDraw
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

from Data.sdf.chemscene_v0_1 import chemsvg_from_chemscene_v0_1
from chemvector.benchmark_diagnostics import diagnose_chemsvg
from chemvector.paths import PROJECT_ROOT


SCHEMA_VERSION = "mr-tadf-causal-audit-0.1"
DEFAULT_EVALUATION = (
    PROJECT_ROOT
    / "outputs/evaluations/mr-tadf-v1-v01021-invalid-only-ocr-fallback-live"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "Data/benchmarks/mr_tadf_v1/manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/audits/mr-tadf-v01021-causal-300"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_score(seed: str, molecule_id: str) -> str:
    return hashlib.sha256(f"{seed}:{molecule_id}".encode()).hexdigest()


def stratified_sample(
    rows: list[dict],
    count: int,
    *,
    seed: str,
    stratum: Callable[[dict], tuple[str, ...]],
) -> list[dict]:
    """Allocate one sample per stratum, then fill proportionally to capacity."""

    if count < 0 or count > len(rows):
        raise ValueError(f"Cannot select {count} rows from {len(rows)}")
    if count == 0:
        return []
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[stratum(row)].append(row)
    for key in grouped:
        grouped[key].sort(
            key=lambda row: deterministic_score(seed, row["molecule_id"])
        )

    keys = sorted(grouped)
    allocations = {key: 0 for key in keys}
    if count >= len(keys):
        for key in keys:
            allocations[key] = 1
    else:
        keys_by_size = sorted(keys, key=lambda key: (-len(grouped[key]), key))
        for key in keys_by_size[:count]:
            allocations[key] = 1

    remaining = count - sum(allocations.values())
    while remaining:
        capacities = {
            key: len(grouped[key]) - allocations[key]
            for key in keys
            if allocations[key] < len(grouped[key])
        }
        total_capacity = sum(capacities.values())
        if not total_capacity:
            raise RuntimeError("Stratified allocation exhausted before reaching target")
        ideal = {
            key: remaining * capacity / total_capacity
            for key, capacity in capacities.items()
        }
        added = 0
        for key in sorted(capacities):
            increment = min(int(ideal[key]), capacities[key])
            allocations[key] += increment
            added += increment
        remaining -= added
        if not remaining:
            break
        remainder_order = sorted(
            capacities,
            key=lambda key: (
                -(ideal[key] - int(ideal[key])),
                deterministic_score(seed, "|".join(key)),
            ),
        )
        for key in remainder_order:
            if remaining == 0:
                break
            if allocations[key] < len(grouped[key]):
                allocations[key] += 1
                remaining -= 1

    selected = [
        row
        for key in keys
        for row in grouped[key][: allocations[key]]
    ]
    return sorted(selected, key=lambda row: deterministic_score(seed, row["molecule_id"]))


def automatic_bucket(row: dict) -> str:
    diagnostics = row["diagnostics"]
    if row["cohort"] == "control":
        return "control_exact"
    missing_atoms = int(diagnostics["missing_atoms"])
    extra_atoms = int(diagnostics["extra_atoms"])
    if missing_atoms or extra_atoms:
        if missing_atoms and extra_atoms:
            return "atom_count_mixed"
        return "atom_under_detection" if missing_atoms else "atom_over_detection"
    if diagnostics["element_mismatches"]:
        return "ocr_transcription"
    if diagnostics["missing_bonds"] and diagnostics["extra_bonds"]:
        return "bond_rewire"
    if diagnostics["extra_bonds"]:
        if diagnostics["crossing_extra_bonds"]:
            return "crossing_extra_bond"
        return "bond_over_detection"
    if diagnostics["missing_bonds"]:
        return "bond_under_detection"
    if diagnostics["bond_type_mismatches"]:
        return "bond_type"
    if diagnostics["atom_property_mismatches"]:
        return "atom_property"
    if not row["sdf_valid"]:
        return "invalid_compilation"
    return "unresolved"


def _q_to_pixel(value: int, maximum: int) -> float:
    return float(value) / 1023.0 * maximum


def write_overlay(
    image_path: Path,
    annotation: dict,
    scene: dict,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    expected_points = {
        int(atom["atom_id"]): (float(atom["image_x"]), float(atom["image_y"]))
        for atom in annotation["atoms"]
    }
    for bond in annotation["bonds"]:
        draw.line(
            [expected_points[int(bond["begin_atom"])], expected_points[int(bond["end_atom"])]],
            fill=(25, 130, 82, 150),
            width=3,
        )
    for atom in annotation["atoms"]:
        x, y = expected_points[int(atom["atom_id"])]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(16, 105, 62, 230), width=2)
        if atom["element"] != "C":
            draw.text((x + 5, y - 12), atom["element"], fill=(12, 91, 54, 255))

    for bond in scene["bond_objects"]:
        begin, end = bond["endpoints"]
        draw.line(
            [
                (_q_to_pixel(begin[0], width), _q_to_pixel(begin[1], height)),
                (_q_to_pixel(end[0], width), _q_to_pixel(end[1], height)),
            ],
            fill=(194, 43, 112, 170),
            width=2,
        )
    for junction in scene["junctions"]:
        x = _q_to_pixel(junction["coordinate"][0], width)
        y = _q_to_pixel(junction["coordinate"][1], height)
        draw.rectangle((x - 3, y - 3, x + 3, y + 3), outline=(171, 23, 92, 255), width=2)
    for text in scene["text_objects"]:
        x = _q_to_pixel(text["center"][0], width)
        y = _q_to_pixel(text["center"][1], height)
        draw.text((x + 5, y + 3), text["text"], fill=(139, 76, 8, 255))

    Image.alpha_composite(image, overlay).convert("RGB").save(output_path)


def _load_sdf(path: Path, *, sanitize: bool) -> Chem.Mol | None:
    mol = Chem.MolFromMolFile(
        str(path), sanitize=sanitize, removeHs=False, strictParsing=False
    )
    if mol is not None and not sanitize:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    return mol


def render_sdf(path: Path, output_path: Path) -> None:
    mol = _load_sdf(path, sanitize=True)
    sanitized = mol is not None
    if mol is None:
        mol = _load_sdf(path, sanitize=False)
    if mol is None:
        raise ValueError(f"Cannot render SDF: {path}")
    drawer = rdMolDraw2D.MolDraw2DCairo(768, 768)
    options = drawer.drawOptions()
    options.prepareMolsBeforeDrawing = sanitized
    options.bondLineWidth = 2.0
    options.fixedFontSize = 16
    options.padding = 0.08
    options.addStereoAnnotation = False
    options.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    output_path.write_bytes(drawer.GetDrawingText())


def write_placeholder(path: Path, message: str) -> None:
    image = Image.new("RGB", (768, 768), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 32, 736, 736), outline=(184, 190, 184), width=2)
    draw.text((64, 360), message[:90], fill=(105, 113, 107))
    image.save(path)


def load_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_audit(
    evaluation_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    error_count: int,
    control_count: int,
    seed: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = {row["molecule_id"]: row for row in csv.DictReader(handle)}
    records = load_records(evaluation_dir / "records.jsonl")
    decomposition = {
        row["molecule_id"]: row
        for row in load_records(evaluation_dir / "object_decomposition_records.jsonl")
    }

    diagnosed: list[dict] = []
    for record in records:
        molecule_id = record["molecule_id"]
        row = manifest[molecule_id]
        annotation_path = manifest_path.parent / row["annotation_file"]
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        scene_path = evaluation_dir / "primary_candidates" / f"{molecule_id}.chemscene.json"
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        sequence = chemsvg_from_chemscene_v0_1(scene)
        diagnostics = diagnose_chemsvg(
            annotation,
            sequence,
            sdf_valid=bool(record["sdf_valid"]),
            chemical_match=bool(record["chemical_match"]),
            nonstereo_structure_match=bool(record["nonstereo_structure_match"]),
            failure_category=record.get("failure_category"),
        )
        diagnosed.append(
            {
                **record,
                "cohort": "control" if record["chemical_match"] else "error",
                "diagnostics": diagnostics,
                "decomposition": decomposition[molecule_id],
                "_manifest": row,
                "_annotation": annotation,
                "_scene": scene,
            }
        )

    errors = stratified_sample(
        [row for row in diagnosed if row["cohort"] == "error"],
        error_count,
        seed=f"{seed}:error",
        stratum=lambda row: (
            row["diagnostics"]["primary_error"],
            row["difficulty"],
        ),
    )
    controls = stratified_sample(
        [row for row in diagnosed if row["cohort"] == "control"],
        control_count,
        seed=f"{seed}:control",
        stratum=lambda row: (row["difficulty"],),
    )
    selected = sorted(
        errors + controls,
        key=lambda row: (row["cohort"] != "error", deterministic_score(seed, row["molecule_id"])),
    )

    public_records: list[dict] = []
    for index, row in enumerate(selected, start=1):
        molecule_id = row["molecule_id"]
        sample_dir = assets_dir / molecule_id
        sample_dir.mkdir(exist_ok=True)
        manifest_row = row["_manifest"]
        image_path = manifest_path.parent / manifest_row["image_file"]
        expected_sdf = manifest_path.parent / manifest_row["sdf_file"]
        predicted_sdf = evaluation_dir / row["prediction_sdf_file"]
        shutil.copy2(image_path, sample_dir / "input.png")
        shutil.copy2(expected_sdf, sample_dir / "expected.sdf")
        shutil.copy2(predicted_sdf, sample_dir / "prediction.sdf")
        write_overlay(
            image_path,
            row["_annotation"],
            row["_scene"],
            sample_dir / "overlay.png",
        )
        for source, target, label in (
            (expected_sdf, sample_dir / "expected.png", "真值 SDF 无法渲染"),
            (predicted_sdf, sample_dir / "prediction.png", "预测 SDF 无法渲染"),
        ):
            try:
                render_sdf(source, target)
            except Exception as error:
                write_placeholder(target, f"{label}: {error}")

        public = {
            "index": index,
            "molecule_id": molecule_id,
            "cohort": row["cohort"],
            "difficulty": row["difficulty"],
            "sdf_valid": bool(row["sdf_valid"]),
            "chemical_match": bool(row["chemical_match"]),
            "skeleton_exact": bool(row["decomposition"]["skeleton_exact"]),
            "atom_typed_exact": bool(row["decomposition"]["atom_typed_skeleton_exact"]),
            "bond_typed_exact": bool(row["decomposition"]["bond_typed_skeleton_exact"]),
            "expected_atoms": int(row["expected_atoms"]),
            "predicted_atoms": int(row["predicted_atoms"]),
            "expected_bonds": int(row["expected_bonds"]),
            "predicted_bonds": int(row["predicted_bonds"]),
            "fingerprint_similarity": float(row.get("fingerprint_similarity", 0.0)),
            "automatic_bucket": automatic_bucket(row),
            "primary_error": row["diagnostics"]["primary_error"],
            "error_flags": row["diagnostics"]["error_flags"],
            "diagnostics": row["diagnostics"],
            "artifacts": {
                "input": f"assets/{molecule_id}/input.png",
                "overlay": f"assets/{molecule_id}/overlay.png",
                "expected_image": f"assets/{molecule_id}/expected.png",
                "prediction_image": f"assets/{molecule_id}/prediction.png",
                "expected_sdf": f"assets/{molecule_id}/expected.sdf",
                "prediction_sdf": f"assets/{molecule_id}/prediction.sdf",
            },
        }
        public_records.append(public)

    selection_payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "error_count": error_count,
        "control_count": control_count,
        "molecule_ids": [row["molecule_id"] for row in public_records],
    }
    selection_sha = hashlib.sha256(
        json.dumps(selection_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "review_required",
        "samples": len(public_records),
        "errors": error_count,
        "controls": control_count,
        "model_version": json.loads(
            (evaluation_dir / "benchmark_config.json").read_text(encoding="utf-8")
        )["model_version"],
        "selection_sha256": selection_sha,
        "source_sha256": {
            "manifest": sha256_file(manifest_path),
            "benchmark_config": sha256_file(evaluation_dir / "benchmark_config.json"),
            "records": sha256_file(evaluation_dir / "records.jsonl"),
            "object_decomposition": sha256_file(
                evaluation_dir / "object_decomposition_records.jsonl"
            ),
        },
        "cohort_counts": dict(Counter(row["cohort"] for row in public_records)),
        "difficulty_counts": dict(Counter(row["difficulty"] for row in public_records)),
        "automatic_bucket_counts": dict(
            sorted(Counter(row["automatic_bucket"] for row in public_records).items())
        ),
        "primary_error_counts": dict(
            sorted(Counter(row["primary_error"] for row in public_records).items())
        ),
        "universe": {
            "samples": len(diagnosed),
            "errors": sum(row["cohort"] == "error" for row in diagnosed),
            "controls": sum(row["cohort"] == "control" for row in diagnosed),
            "difficulty_counts": dict(
                sorted(Counter(row["difficulty"] for row in diagnosed).items())
            ),
            "automatic_bucket_counts": dict(
                sorted(
                    Counter(automatic_bucket(row) for row in diagnosed).items()
                )
            ),
            "primary_error_counts": dict(
                sorted(
                    Counter(
                        row["diagnostics"]["primary_error"] for row in diagnosed
                    ).items()
                )
            ),
        },
    }
    payload = {"summary": summary, "records": public_records}
    (output_dir / "audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    annotations_path = output_dir / "annotations.json"
    if annotations_path.is_file():
        existing_annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
        if existing_annotations.get("selection_sha256") != selection_sha:
            raise ValueError(
                "Existing annotations belong to a different audit selection"
            )
    else:
        annotations_path.write_text(
            json.dumps(
                {
                    "schema_version": "mr-tadf-causal-audit-annotations-0.1",
                    "selection_sha256": selection_sha,
                    "annotations": {},
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--errors", type=int, default=200)
    parser.add_argument("--controls", type=int, default=100)
    parser.add_argument("--seed", default="mr-tadf-v01021-causal-audit-v1")
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    summary = build_audit(
        args.evaluation_dir.resolve(),
        args.manifest.resolve(),
        args.output_dir.resolve(),
        error_count=args.errors,
        control_count=args.controls,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
