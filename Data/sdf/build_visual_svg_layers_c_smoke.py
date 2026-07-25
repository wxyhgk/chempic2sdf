#!/usr/bin/env python3
"""Build a train-only 12-member smoke package for Visual SVG Layers C v0.1.

This package validates the data contract only.  It deliberately reuses the
already exposed ChemVision pilot train split and is never eligible for model
selection, fresh audit, or public generalization claims.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
from typing import Any, Sequence

import fastjsonschema
from PIL import Image
from rdkit import Chem, rdBase

try:
    from . import build_chemvision_layers as layer_builder
    from .build_visual_svg_capacity_100 import (
        PilotArchives,
        build_task,
        json_bytes,
        sha256_bytes,
        stable_int,
        tar_add_bytes,
    )
    from .visual_svg_layers_c import (
        HASHED_SUFFIXES,
        MEMBER_SUFFIXES,
        assemble_c_sample,
    )
    from .visual_svg_sequence_v0_1 import parse_visual_svg_v0_1
    from .visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
    from .visual_svg_v0_2 import (
        flatten_visual_svg_v0_2_to_v0_1,
        parse_visual_svg_v0_2,
    )
except ImportError:
    import build_chemvision_layers as layer_builder
    from build_visual_svg_capacity_100 import (
        PilotArchives,
        build_task,
        json_bytes,
        sha256_bytes,
        stable_int,
        tar_add_bytes,
    )
    from visual_svg_layers_c import (
        HASHED_SUFFIXES,
        MEMBER_SUFFIXES,
        assemble_c_sample,
    )
    from visual_svg_sequence_v0_1 import parse_visual_svg_v0_1
    from visual_svg_tokenizer_v0_1 import VisualSVGTokenizerV01
    from visual_svg_v0_2 import (
        flatten_visual_svg_v0_2_to_v0_1,
        parse_visual_svg_v0_2,
    )


SCHEMA_VERSION = "visual-svg-layers-c-smoke-0.1"
SEED = 20260722
VARIANTS = ("original", "rdkit", "coordgen", "stress")
LAYER_SUFFIXES = (
    ".png",
    ".text.png",
    ".bond.png",
    ".semantic.png",
    ".layers.json",
    ".render.json",
)
FORBIDDEN_INPUT_MARKERS = (
    "validation",
    "internal_test",
    "internal-test",
    "sealed",
    "mr-tadf",
    "mr_tadf",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_train_only(value: object, *, context: str) -> None:
    normalized = str(value).lower()
    if any(marker in normalized for marker in FORBIDDEN_INPUT_MARKERS):
        raise ValueError(f"Forbidden non-train input in {context}: {value}")


def load_exposed_train_rows(pilot: Path) -> list[dict[str, str]]:
    """Build a safe manifest exclusively from explicitly named train shards."""

    summary_path = pilot / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split_counts = summary.get("sample_split_counts")
    if not isinstance(split_counts, dict) or int(split_counts.get("train", 0)) <= 0:
        raise ValueError("Pilot summary lacks a positive train sample count")
    expected_train_count = int(split_counts["train"])
    rows: list[dict[str, str]] = []
    train_shards = sorted((pilot / "shards").glob("train-*.tar"))
    if not train_shards:
        raise ValueError("Pilot has no explicitly named train shards")
    for shard in train_shards:
        if shard.is_symlink():
            raise ValueError(f"Train shard cannot be a symlink: {shard}")
        relative_shard = shard.relative_to(pilot).as_posix()
        _assert_train_only(relative_shard, context="pilot_train_shard")
        with tarfile.open(shard, "r") as archive:
            for member in archive:
                if not member.name.endswith(".layers.json"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Cannot read {member.name} from {shard}")
                layers = json.load(extracted)
                if layers.get("split") != "train":
                    raise ValueError(
                        f"Train shard contains split={layers.get('split')!r}"
                    )
                key = member.name.removesuffix(".layers.json")
                source = layers.get("source") or {}
                render = layers.get("render_parameters") or {}
                font_size = int(render["font_size"])
                if font_size <= 12:
                    font_bucket = "small"
                elif font_size <= 19:
                    font_bucket = "normal"
                elif font_size <= 28:
                    font_bucket = "large"
                else:
                    font_bucket = "extreme"
                row = {
                    "sample_key": key,
                    "molecule_id": str(layers["molecule_id"]),
                    "split": "train",
                    "variant": str(layers["variant"]),
                    "source_sample_key": str(source["source_sample_key"]),
                    "source_shard": str(source["source_shard"]),
                    "output_shard": relative_shard,
                    "font_family_id": str(render["font_family_id"]),
                    "font_size": str(font_size),
                    "font_bucket": font_bucket,
                    "bond_line_width": str(render["bond_line_width"]),
                    "rotation_degrees": str(render["rotation_degrees"]),
                }
                for field in (
                    "sample_key",
                    "molecule_id",
                    "source_sample_key",
                    "source_shard",
                    "output_shard",
                ):
                    _assert_train_only(row[field], context=f"train_shard:{field}")
                if not row["source_sample_key"].startswith("train_"):
                    raise ValueError("Pilot source_sample_key is not train-prefixed")
                if not row["source_shard"].startswith("shards/train-"):
                    raise ValueError("Pilot source_shard is not train-only")
                rows.append(row)
    if len(rows) != expected_train_count:
        raise ValueError(
            f"Train shard rows={len(rows)}, expected {expected_train_count}"
        )
    if len({row["sample_key"] for row in rows}) != len(rows):
        raise ValueError("Pilot train shards contain duplicate sample keys")
    return rows


def select_complete_molecules(
    rows: Sequence[dict[str, str]], molecule_count: int
) -> list[dict[str, str]]:
    if molecule_count <= 0:
        raise ValueError("molecule_count must be positive")
    by_molecule: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        variant = row["variant"]
        if variant not in VARIANTS:
            continue
        molecule = row["molecule_id"]
        if variant in by_molecule[molecule]:
            raise ValueError(f"Duplicate {variant} row for {molecule}")
        by_molecule[molecule][variant] = row
    complete = [
        molecule
        for molecule, variants in by_molecule.items()
        if set(variants) == set(VARIANTS)
    ]
    ranked = sorted(
        complete,
        key=lambda molecule: (
            stable_int(SEED, "visual-svg-layers-c-smoke", molecule),
            molecule,
        ),
    )
    selected = ranked[:molecule_count]
    if len(selected) != molecule_count:
        raise ValueError(
            f"Only {len(selected)} complete train molecules, expected {molecule_count}"
        )
    return [
        by_molecule[molecule][variant]
        for molecule in selected
        for variant in VARIANTS
    ]


def _sdf_filename(molecule_id: str) -> str:
    if ":" not in molecule_id:
        raise ValueError(f"Unsupported molecule_id for SDF lookup: {molecule_id}")
    return molecule_id.replace(":", "_", 1) + ".sdf"


def find_train_sdf(source: Path, molecule_id: str) -> Path:
    filename = _sdf_filename(molecule_id)
    matches = sorted((source / "structures").glob(f"train-*/{filename}"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one train SDF for {molecule_id}, found {len(matches)}"
        )
    relative = matches[0].relative_to(source).as_posix()
    _assert_train_only(relative, context="structure_ref")
    if not relative.startswith("structures/train-"):
        raise ValueError(f"SDF path is outside train structures: {relative}")
    return matches[0]


def parse_single_sdf(data: bytes, *, molecule_id: str) -> tuple[Chem.Mol, str]:
    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(data),
        sanitize=True,
        removeHs=False,
        strictParsing=True,
    )
    molecules = list(supplier)
    if len(molecules) != 1 or molecules[0] is None:
        raise ValueError(f"SDF for {molecule_id} must contain one valid molecule")
    molecule = molecules[0]
    key = Chem.MolToInchiKey(molecule)
    if not key:
        raise ValueError(f"SDF for {molecule_id} lacks a valid InChIKey")
    return molecule, key


def load_train_sdfs(
    source: Path, molecule_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    result = {}
    for molecule_id in sorted(set(molecule_ids)):
        path = find_train_sdf(source, molecule_id)
        data = path.read_bytes()
        molecule, inchikey = parse_single_sdf(data, molecule_id=molecule_id)
        result[molecule_id] = {
            "path": path,
            "relative_path": path.relative_to(source).as_posix(),
            "bytes": data,
            "sha256": sha256_bytes(data),
            "inchikey": inchikey,
            "atom_count": molecule.GetNumAtoms(),
            "bond_count": molecule.GetNumBonds(),
        }
    return result


def _image_contract(data: bytes, *, suffix: str) -> None:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.size != (768, 768):
            raise ValueError(f"{suffix} must be 768x768, got {image.size}")
        expected_mode = "RGB" if suffix == ".png" else "L"
        if image.mode != expected_mode:
            raise ValueError(
                f"{suffix} must use mode {expected_mode}, got {image.mode}"
            )


def prepare_smoke_sample(
    key: str,
    row: dict[str, str],
    archives: PilotArchives,
    tokenizer: VisualSVGTokenizerV01,
    primitive_validator,
    sdf: dict[str, Any],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    expected_layer_members = {
        suffix: archives.read(key, suffix) for suffix in LAYER_SUFFIXES
    }
    chemmol_bytes = archives.read(key, ".chemmol.json")
    source_render = json.loads(archives.read(key, ".render.json"))
    members, diagnostics = assemble_c_sample(
        build_task(row),
        chemmol_bytes,
        sdf["bytes"],
        source_render,
        primitive_validator=primitive_validator,
        tokenizer=tokenizer,
        output_split="smoke_train",
        source_dataset="chemvision_layers_v0_1/pilot_10k",
        structure_ref=sdf["relative_path"],
        fresh_audit_eligible=False,
        expected_layer_members=expected_layer_members,
    )
    return members, {
        "sample_key": key,
        "molecule_id": row["molecule_id"],
        "split": "smoke_train",
        "variant": row["variant"],
        "source_sample_key": row["source_sample_key"],
        "source_shard": row["source_shard"],
        "source_output_shard": row["output_shard"],
        "sdf_ref": sdf["relative_path"],
        "sdf_inchikey": sdf["inchikey"],
        "sequence_tokens": int(diagnostics["sequence_tokens"]),
        "primitive_count": int(diagnostics["primitive_count"]),
        "visual_svg_v02_group_count": int(
            diagnostics["visual_svg_v02_group_count"]
        ),
        "semantic_svg_sha256": diagnostics["semantic_svg_sha256"],
        "sha256": diagnostics["sha256"],
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_shard(
    path: Path,
    samples: Sequence[tuple[str, dict[str, bytes], dict[str, Any]]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with tarfile.open(temporary, "w") as archive:
        for key, members, _ in sorted(samples, key=lambda item: item[0]):
            for suffix in MEMBER_SUFFIXES:
                tar_add_bytes(archive, key + suffix, members[suffix])
    temporary.replace(path)


def _content_sha256(
    samples: Sequence[tuple[str, dict[str, bytes], dict[str, Any]]]
) -> str:
    digest = hashlib.sha256()
    for key, members, _ in sorted(samples, key=lambda item: item[0]):
        for suffix in MEMBER_SUFFIXES:
            name = (key + suffix).encode("utf-8")
            value = members[suffix]
            digest.update(len(name).to_bytes(4, "big"))
            digest.update(name)
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def validate_package(
    shard: Path,
    samples: Sequence[tuple[str, dict[str, bytes], dict[str, Any]]],
    primitive_validator,
) -> dict[str, Any]:
    errors: list[str] = []
    expected_names = [
        key + suffix
        for key, _, _ in sorted(samples, key=lambda item: item[0])
        for suffix in MEMBER_SUFFIXES
    ]
    expected = {key: members for key, members, _ in samples}
    variant_by_molecule: dict[str, set[str]] = defaultdict(set)
    sdf_hashes_by_molecule: dict[str, set[str]] = defaultdict(set)
    checked_members = 0
    checked_meta_hashes = 0
    with tarfile.open(shard, "r") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if names != expected_names:
            errors.append("tar_member_order_or_names")
        if len(names) != len(set(names)):
            errors.append("duplicate_tar_member")
        for member in members:
            if (
                member.mtime != 0
                or member.mode != 0o644
                or member.uid != 0
                or member.gid != 0
                or member.uname
                or member.gname
            ):
                errors.append(f"nondeterministic_tar_metadata={member.name}")
        for key, _, row in sorted(samples, key=lambda item: item[0]):
            actual: dict[str, bytes] = {}
            for suffix in MEMBER_SUFFIXES:
                extracted = archive.extractfile(key + suffix)
                if extracted is None:
                    errors.append(f"missing_member={key}{suffix}")
                    continue
                data = extracted.read()
                actual[suffix] = data
                checked_members += 1
                if data != expected[key][suffix]:
                    errors.append(f"post_write_mismatch={key}{suffix}")
            if set(actual) != set(MEMBER_SUFFIXES):
                continue
            try:
                for suffix in (".png", ".text.png", ".bond.png", ".semantic.png"):
                    _image_contract(actual[suffix], suffix=suffix)
                primitives = json.loads(actual[".primitives.json"])
                primitive_validator(primitives)
                parsed_v01 = parse_visual_svg_v0_1(
                    actual[".visualsvg.txt"].decode("utf-8")
                )
                parsed_v02 = parse_visual_svg_v0_2(
                    actual[".visualsvg-v02.json"].decode("utf-8")
                )
                if flatten_visual_svg_v0_2_to_v0_1(parsed_v02) != parsed_v01:
                    raise ValueError("v02_flatten_mismatch")
                _, sdf_key = parse_single_sdf(
                    actual[".sdf"], molecule_id=row["molecule_id"]
                )
                meta = json.loads(actual[".meta.json"])
                if meta.get("fresh_audit_eligible") is not False:
                    raise ValueError("smoke_marked_fresh_audit_eligible")
                if meta.get("sdf_inchikey") != sdf_key:
                    raise ValueError("meta_sdf_inchikey_mismatch")
                for suffix in HASHED_SUFFIXES:
                    name = suffix.removeprefix(".")
                    if meta["sha256"].get(name) != sha256_bytes(actual[suffix]):
                        raise ValueError(f"meta_hash_mismatch={name}")
                    checked_meta_hashes += 1
            except Exception as error:
                errors.append(f"sample_validation={key}:{type(error).__name__}:{error}")
            variant_by_molecule[row["molecule_id"]].add(row["variant"])
            sdf_hashes_by_molecule[row["molecule_id"]].add(
                sha256_bytes(actual[".sdf"])
            )
    for molecule_id, variants in variant_by_molecule.items():
        if variants != set(VARIANTS):
            errors.append(f"variant_closure={molecule_id}:{sorted(variants)}")
        if len(sdf_hashes_by_molecule[molecule_id]) != 1:
            errors.append(f"sdf_variant_hash_mismatch={molecule_id}")
    return {
        "schema_version": "visual-svg-layers-c-smoke-validation-0.1",
        "status": "passed" if not errors else "failed",
        "fresh_audit_eligible": False,
        "molecules": len(variant_by_molecule),
        "samples": len(samples),
        "members": len(expected_names),
        "checked_members": checked_members,
        "checked_meta_hashes": checked_meta_hashes,
        "variant_closure_passed": all(
            variants == set(VARIANTS) for variants in variant_by_molecule.values()
        ),
        "sdf_hash_shared_across_variants": all(
            len(values) == 1 for values in sdf_hashes_by_molecule.values()
        ),
        "content_sha256": _content_sha256(samples),
        "shard_sha256": file_sha256(shard),
        "errors": errors,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    pilot = Path(args.pilot).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve(strict=False)
    for path, context in ((pilot, "pilot"), (source, "source")):
        _assert_train_only(path.name, context=context)
    generated = (
        output / "shards",
        output / "samples.csv",
        output / "molecules.csv",
        output / "paired_manifest.csv",
        output / "paired_manifest.smoke.csv",
        output / "commitment.smoke.json",
        output / "summary.json",
        output / "validation.json",
        output / "independent_validation.json",
    )
    existing = [path for path in generated if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Smoke output exists; pass --overwrite: {existing[0]}")
    if args.overwrite:
        for path in generated:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(parents=True, exist_ok=True)

    pilot_summary = json.loads(
        (pilot / "summary.json").read_text(encoding="utf-8")
    )
    font_profile = str(pilot_summary.get("font_profile", "v01-original"))
    if font_profile not in layer_builder.FONT_PROFILES:
        raise ValueError(f"Unsupported pilot font profile: {font_profile}")
    layer_builder.FONT_SPECS = layer_builder.FONT_PROFILES[font_profile]
    train_rows = load_exposed_train_rows(pilot)
    selected_rows = select_complete_molecules(train_rows, args.molecules)
    selected_by_key = {row["sample_key"]: row for row in selected_rows}
    sdf_by_molecule = load_train_sdfs(
        source, [row["molecule_id"] for row in selected_rows]
    )
    schema_path = (
        Path(__file__).resolve().parent
        / "visual_svg_v0_1/schemas/visual-primitives-0.1.schema.json"
    )
    primitive_validator = fastjsonschema.compile(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )
    tokenizer = VisualSVGTokenizerV01()
    archives = PilotArchives(pilot, selected_by_key)
    samples = []
    try:
        for row in selected_rows:
            key = row["sample_key"]
            members, result = prepare_smoke_sample(
                key,
                row,
                archives,
                tokenizer,
                primitive_validator,
                sdf_by_molecule[row["molecule_id"]],
            )
            samples.append((key, members, result))
    finally:
        archives.close()

    shard = output / "shards/smoke-train-000000.tar"
    write_shard(shard, samples)
    rows = [item[2] for item in sorted(samples, key=lambda item: item[0])]
    _write_csv(
        output / "samples.csv",
        rows,
        (
            "sample_key",
            "molecule_id",
            "split",
            "variant",
            "source_sample_key",
            "source_shard",
            "source_output_shard",
            "sdf_ref",
            "sdf_inchikey",
            "sequence_tokens",
            "primitive_count",
            "visual_svg_v02_group_count",
        ),
    )
    molecule_rows = []
    for molecule_id in sorted(sdf_by_molecule):
        sdf = sdf_by_molecule[molecule_id]
        molecule_rows.append(
            {
                "molecule_id": molecule_id,
                "split": "smoke_train",
                "variants": 4,
                "sdf_ref": sdf["relative_path"],
                "sdf_sha256": sdf["sha256"],
                "inchikey": sdf["inchikey"],
                "atom_count": sdf["atom_count"],
                "bond_count": sdf["bond_count"],
            }
        )
    _write_csv(
        output / "molecules.csv",
        molecule_rows,
        (
            "molecule_id",
            "split",
            "variants",
            "sdf_ref",
            "sdf_sha256",
            "inchikey",
            "atom_count",
            "bond_count",
        ),
    )
    paired_rows = []
    for key, members, row in sorted(samples, key=lambda item: item[0]):
        paired_rows.append(
            {
                "sample_key": key,
                "molecule_id": row["molecule_id"],
                "inchikey": row["sdf_inchikey"],
                "split": "smoke_train",
                "variant": row["variant"],
                "shard": "shards/smoke-train-000000.tar",
                "image_sha256": sha256_bytes(members[".png"]),
                "sdf_sha256": sha256_bytes(members[".sdf"]),
                "primitives_sha256": sha256_bytes(members[".primitives.json"]),
                "visualsvg_v01_sha256": sha256_bytes(members[".visualsvg.txt"]),
                "visualsvg_v02_sha256": sha256_bytes(
                    members[".visualsvg-v02.json"]
                ),
                "layers_sha256": sha256_bytes(members[".layers.json"]),
                "render_sha256": sha256_bytes(members[".render.json"]),
                "chemmol_sha256": sha256_bytes(members[".chemmol.json"]),
                "meta_sha256": sha256_bytes(members[".meta.json"]),
            }
        )
    paired_fields = (
        "sample_key",
        "molecule_id",
        "inchikey",
        "split",
        "variant",
        "shard",
        "image_sha256",
        "sdf_sha256",
        "primitives_sha256",
        "visualsvg_v01_sha256",
        "visualsvg_v02_sha256",
        "layers_sha256",
        "render_sha256",
        "chemmol_sha256",
        "meta_sha256",
    )
    _write_csv(output / "paired_manifest.csv", paired_rows, paired_fields)
    _write_csv(output / "paired_manifest.smoke.csv", paired_rows, paired_fields)
    validation = validate_package(shard, samples, primitive_validator)
    builder_path = Path(__file__).resolve()
    assembler_path = Path(assemble_c_sample.__code__.co_filename).resolve()
    layer_builder_path = Path(layer_builder.__file__).resolve()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": validation["status"],
        "purpose": "data-contract smoke test only",
        "fresh_audit_eligible": False,
        "seed": SEED,
        "rdkit_version": rdBase.rdkitVersion,
        "font_profile": font_profile,
        "source_pilot": str(pilot),
        "source_full_dataset": str(source),
        "molecules": len(molecule_rows),
        "samples": len(samples),
        "variants": dict(Counter(row["variant"] for row in rows)),
        "members_per_sample": len(MEMBER_SUFFIXES),
        "tar_members": len(samples) * len(MEMBER_SUFFIXES),
        "bytes": shard.stat().st_size,
        "shard_sha256": validation["shard_sha256"],
        "content_sha256": validation["content_sha256"],
        "builder_path": str(builder_path),
        "builder_sha256": file_sha256(builder_path),
        "assembler_path": str(assembler_path),
        "assembler_sha256": file_sha256(assembler_path),
        "layer_builder_path": str(layer_builder_path),
        "layer_builder_sha256": file_sha256(layer_builder_path),
        "primitive_schema_sha256": file_sha256(schema_path),
    }
    (output / "summary.json").write_bytes(json_bytes(summary))
    (output / "validation.json").write_bytes(json_bytes(validation))
    if validation["status"] != "passed":
        raise RuntimeError(f"Smoke validation failed: {validation['errors'][:3]}")
    commitment = {
        "schema_version": "visual-svg-layers-c-input-commitment-0.1",
        "role": "smoke",
        "fresh_audit_eligible": False,
        "sample_count": len(samples),
        "molecule_count": len(molecule_rows),
        "variants": list(VARIANTS),
        "members_per_sample": len(MEMBER_SUFFIXES),
        "paired_manifest": {
            "path": "paired_manifest.smoke.csv",
            "sha256": file_sha256(output / "paired_manifest.smoke.csv"),
        },
        "shards": [
            {
                "path": "shards/smoke-train-000000.tar",
                "sha256": validation["shard_sha256"],
                "bytes": shard.stat().st_size,
                "samples": len(samples),
                "members": len(samples) * len(MEMBER_SUFFIXES),
            }
        ],
        "validation": {
            "path": "validation.json",
            "sha256": file_sha256(output / "validation.json"),
            "status": validation["status"],
        },
        "implementation": {
            "builder_sha256": summary["builder_sha256"],
            "assembler_sha256": summary["assembler_sha256"],
            "layer_builder_sha256": summary["layer_builder_sha256"],
            "primitive_schema_sha256": summary["primitive_schema_sha256"],
        },
    }
    (output / "commitment.smoke.json").write_bytes(json_bytes(commitment))
    return {"output": str(output), **summary}


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot",
        default=str(root / "chemvision_layers_v0_1/pilot_10k"),
    )
    parser.add_argument(
        "--source",
        default=str(root / "webdataset_full_v0"),
    )
    parser.add_argument(
        "--output",
        default=str(root / "visual_svg_layers_c_v0_1/smoke_train_32"),
    )
    parser.add_argument("--molecules", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    report = build(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "MEMBER_SUFFIXES",
    "SCHEMA_VERSION",
    "build",
    "build_parser",
    "find_train_sdf",
    "load_exposed_train_rows",
    "main",
    "select_complete_molecules",
    "validate_package",
]
