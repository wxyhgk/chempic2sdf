#!/usr/bin/env python3
"""Build a validated 768px ChemVector pilot from MolGrapher Synthetic."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from Data.sdf.chemmol_sequence import serialize_chemmolseq
from Data.sdf.chemsvg_sequence_v4 import (
    serialize_chemsvg_v4,
    validate_chemsvg_v4_roundtrip,
)


REVISION = "8a13a6a5d29ae40f98c6b92a5ca8b139d0e0a945"
SEED = 20260715
IMAGE_SIZE = 768
TRAIN_TARGET = 9000
VALIDATION_TARGET = 1000
SHARD_SIZE = 2000
SUPPORTED_BOND_TYPES = {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}


@dataclass(frozen=True, slots=True)
class Candidate:
    source_file: str
    source_row: int
    official_split: str
    source_id: str
    molfile: str
    keypoints: tuple[int, ...]
    smiles: str
    inchikey: str
    scaffold: str
    bucket: str
    atoms: int
    bonds: int


def enum_name(value: object) -> str:
    return str(value).rsplit(".", 1)[-1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def tar_add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    archive.addfile(info, io.BytesIO(payload))


def canonical_smiles(mol: Chem.Mol) -> str:
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def scaffold_smiles(mol: Chem.Mol, smiles: str) -> str:
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=True,
    )
    return scaffold if scaffold else f"ACYCLIC:{smiles}"


def selection_bucket(mol: Chem.Mol) -> str:
    atoms = mol.GetNumAtoms()
    size = "small" if atoms < 20 else "medium" if atoms < 40 else "large" if atoms < 70 else "xlarge"
    types = {enum_name(bond.GetBondType()) for bond in mol.GetBonds()}
    bond = "triple" if "TRIPLE" in types else "double" if "DOUBLE" in types else "aromatic" if "AROMATIC" in types else "single"
    visible = sum(atom.GetSymbol() not in {"C", "H"} for atom in mol.GetAtoms())
    text = "text_high" if visible >= 6 else "text_low"
    return f"{size}:{bond}:{text}"


def parse_candidate(
    row: dict,
    *,
    source_file: Path,
    source_row: int,
    official_split: str,
    excluded_inchikeys: set[str],
) -> Candidate:
    source_id = str(row.get("id"))
    molfile = row.get("mol")
    keypoints = row.get("keypoints")
    if not isinstance(molfile, str) or not molfile.strip():
        raise ValueError("empty_molfile")
    if not isinstance(keypoints, list) or len(keypoints) % 3:
        raise ValueError("invalid_keypoints")
    mol = Chem.MolFromMolBlock(
        molfile,
        sanitize=True,
        removeHs=False,
        strictParsing=False,
    )
    if mol is None:
        raise ValueError("molfile_sanitize_failed")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("multiple_fragments")
    if mol.GetNumAtoms() != len(keypoints) // 3:
        raise ValueError("atom_keypoint_count_mismatch")
    if any(atom.GetAtomicNum() == 0 or atom.HasQuery() for atom in mol.GetAtoms()):
        raise ValueError("query_or_superatom")
    bond_types = {enum_name(bond.GetBondType()) for bond in mol.GetBonds()}
    if not bond_types <= SUPPORTED_BOND_TYPES:
        raise ValueError("unsupported_bond_type")
    coordinates = [(keypoints[index] - 1, keypoints[index + 1] - 1) for index in range(0, len(keypoints), 3)]
    if any(x < 0 or y < 0 or x >= 1024 or y >= 1024 for x, y in coordinates):
        raise ValueError("keypoint_outside_1024_canvas")
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("duplicate_keypoint")
    smiles = canonical_smiles(mol)
    inchikey = Chem.MolToInchiKey(mol)
    if not inchikey:
        raise ValueError("empty_inchikey")
    if inchikey in excluded_inchikeys:
        raise ValueError("benchmark_overlap")
    return Candidate(
        source_file=str(source_file),
        source_row=source_row,
        official_split=official_split,
        source_id=source_id,
        molfile=molfile,
        keypoints=tuple(int(value) for value in keypoints),
        smiles=smiles,
        inchikey=inchikey,
        scaffold=scaffold_smiles(mol, smiles),
        bucket=selection_bucket(mol),
        atoms=mol.GetNumAtoms(),
        bonds=mol.GetNumBonds(),
    )


def scan_candidate_path(
    path: Path,
    *,
    official_split: str,
    excluded_inchikeys: set[str],
) -> tuple[list[Candidate], list[dict]]:
    RDLogger.DisableLog("rdApp.*")
    candidates: list[Candidate] = []
    rejected: list[dict] = []
    seen = set(excluded_inchikeys)
    parquet = pq.ParquetFile(path)
    row_offset = 0
    for batch in parquet.iter_batches(
        batch_size=256,
        columns=["id", "image", "mol", "smiles", "keypoints"],
    ):
        rows = batch.to_pylist()
        for offset, row in enumerate(rows):
            row_index = row_offset + offset
            try:
                candidate = parse_candidate(
                    row,
                    source_file=path,
                    source_row=row_index,
                    official_split=official_split,
                    excluded_inchikeys=seen,
                )
                validate_candidate_assets(candidate, row)
                seen.add(candidate.inchikey)
                candidates.append(candidate)
            except Exception as error:
                rejected.append(
                    {
                        "source_file": path.name,
                        "source_row": row_index,
                        "source_id": str(row.get("id")),
                        "official_split": official_split,
                        "reason": str(error),
                    }
                )
        row_offset += len(rows)
    return candidates, rejected


def _scan_candidate_path_worker(arguments):
    return scan_candidate_path(
        arguments[0],
        official_split=arguments[1],
        excluded_inchikeys=arguments[2],
    )


def scan_candidates(
    paths: list[Path],
    *,
    official_split: str,
    excluded_inchikeys: set[str],
) -> tuple[list[Candidate], list[dict]]:
    if len(paths) == 1:
        return scan_candidate_path(
            paths[0],
            official_split=official_split,
            excluded_inchikeys=excluded_inchikeys,
        )
    arguments = [
        (path, official_split, excluded_inchikeys)
        for path in paths
    ]
    with ProcessPoolExecutor(max_workers=min(4, len(paths))) as executor:
        results = list(executor.map(_scan_candidate_path_worker, arguments))
    candidates: list[Candidate] = []
    rejected: list[dict] = []
    seen = set(excluded_inchikeys)
    for (path_candidates, path_rejected), path in zip(results, paths):
        rejected.extend(path_rejected)
        for candidate in path_candidates:
            if candidate.inchikey in seen:
                rejected.append(
                    {
                        "source_file": path.name,
                        "source_row": candidate.source_row,
                        "source_id": candidate.source_id,
                        "official_split": official_split,
                        "reason": "cross_shard_duplicate_inchikey",
                    }
                )
                continue
            seen.add(candidate.inchikey)
            candidates.append(candidate)
    return candidates, rejected


def select_stratified(candidates: list[Candidate], count: int, label: str) -> list[Candidate]:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.bucket].append(candidate)
    for bucket, values in groups.items():
        values.sort(key=lambda item: stable_int(SEED, label, bucket, item.source_id))
    selected: list[Candidate] = []
    ordered_buckets = sorted(groups)
    while len(selected) < count:
        progressed = False
        for bucket in ordered_buckets:
            if groups[bucket]:
                selected.append(groups[bucket].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    if len(selected) != count:
        raise RuntimeError(f"{label}: selected {len(selected)} of requested {count}")
    return selected


def benchmark_inchikeys(project_root: Path) -> set[str]:
    result: set[str] = set()
    mr_manifest = project_root / "Data/benchmarks/mr_tadf_v1/manifest.csv"
    if mr_manifest.exists():
        with mr_manifest.open(encoding="utf-8", newline="") as handle:
            result.update(row["inchikey"] for row in csv.DictReader(handle))
    uspto_root = project_root / "Data/benchmarks/public_ocsr_v1/prepared/uspto_clean_1k"
    uspto_manifest = uspto_root / "manifest.csv"
    if uspto_manifest.exists():
        with uspto_manifest.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                mol = Chem.MolFromMolFile(
                    str(uspto_root / row["sdf_file"]),
                    sanitize=True,
                    removeHs=False,
                    strictParsing=False,
                )
                if mol is not None:
                    result.add(Chem.MolToInchiKey(mol))
    return result


def build_chemmol(mol: Chem.Mol, molecule_id: str, structure_ref: str) -> dict:
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    atoms = []
    for atom in mol.GetAtoms():
        atoms.append(
            {
                "id": atom.GetIdx(),
                "source_atom_index": atom.GetIdx(),
                "element": atom.GetSymbol(),
                "atomic_number": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "isotope": atom.GetIsotope(),
                "aromatic": atom.GetIsAromatic(),
                "chiral_tag": enum_name(atom.GetChiralTag()),
                "cip_code": atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
                "explicit_h_count": atom.GetNumExplicitHs(),
                "implicit_h_count": atom.GetNumImplicitHs(),
                "no_implicit": atom.GetNoImplicit(),
                "radical_electrons": atom.GetNumRadicalElectrons(),
            }
        )
    bonds = []
    for bond in mol.GetBonds():
        stereo = enum_name(bond.GetStereo())
        bonds.append(
            {
                "id": bond.GetIdx(),
                "begin_atom": bond.GetBeginAtomIdx(),
                "end_atom": bond.GetEndAtomIdx(),
                "type": enum_name(bond.GetBondType()),
                "aromatic": bond.GetIsAromatic(),
                "conjugated": bond.GetIsConjugated(),
                "stereo": stereo,
                "stereo_atoms": list(bond.GetStereoAtoms()),
                "ez": "E" if stereo == "STEREOE" else "Z" if stereo == "STEREOZ" else None,
            }
        )
    return {
        "schema_version": "chemmol-0.1",
        "molecule_id": molecule_id,
        "structure_ref": structure_ref,
        "canonical_isomeric_smiles": canonical_smiles(mol),
        "inchikey": Chem.MolToInchiKey(mol),
        "atoms": atoms,
        "bonds": bonds,
    }


def scaled_positions(candidate: Candidate, width: int, height: int) -> list[tuple[float, float]]:
    positions = []
    for index in range(0, len(candidate.keypoints), 3):
        source_x = candidate.keypoints[index] - 1
        source_y = candidate.keypoints[index + 1] - 1
        positions.append(
            (
                source_x / max(width - 1, 1) * (IMAGE_SIZE - 1),
                source_y / max(height - 1, 1) * (IMAGE_SIZE - 1),
            )
        )
    return positions


def quantize(value: float) -> int:
    return max(0, min(1023, round(value / (IMAGE_SIZE - 1) * 1023)))


def build_render(
    mol: Chem.Mol,
    molecule_id: str,
    positions: list[tuple[float, float]],
    source_size: tuple[int, int],
) -> dict:
    conformer = mol.GetConformer() if mol.GetNumConformers() else None
    atoms = []
    for atom_id, (image_x, image_y) in enumerate(positions):
        point = conformer.GetAtomPosition(atom_id) if conformer is not None else None
        atoms.append(
            {
                "atom_id": atom_id,
                "molecule_x": round(float(point.x) if point is not None else image_x, 6),
                "molecule_y": round(float(point.y) if point is not None else -image_y, 6),
                "image_x": round(image_x, 3),
                "image_y": round(image_y, 3),
                "quantized_x": quantize(image_x),
                "quantized_y": quantize(image_y),
            }
        )
    bonds = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_type = enum_name(bond.GetBondType())
        bonds.append(
            {
                "bond_id": bond.GetIdx(),
                "begin_image_x": round(positions[begin][0], 3),
                "begin_image_y": round(positions[begin][1], 3),
                "end_image_x": round(positions[end][0], 3),
                "end_image_y": round(positions[end][1], 3),
                "display_type": bond_type,
                "direction": enum_name(bond.GetBondDir()),
            }
        )
    return {
        "schema_version": "render-0.1",
        "molecule_id": molecule_id,
        "variant": "molgrapher_original",
        "width": IMAGE_SIZE,
        "height": IMAGE_SIZE,
        "atoms": atoms,
        "bonds": bonds,
        "render_parameters": {
            "layout": "molgrapher_original",
            "source_width": source_size[0],
            "source_height": source_size[1],
            "resize": "lanczos",
            "color_mode": "source",
            "stress_operation": None,
        },
    }


def foreground_anchor_rate(
    image: Image.Image,
    positions: list[tuple[float, float]],
    *,
    radius: int = 48,
) -> float:
    """Detect gross coordinate drift while allowing hollow large atom glyphs."""

    gray = np.asarray(image.convert("L"))
    dark = gray < 220
    hits = 0
    for x, y in positions:
        left = max(0, round(x) - radius)
        right = min(gray.shape[1], round(x) + radius + 1)
        top = max(0, round(y) - radius)
        bottom = min(gray.shape[0], round(y) + radius + 1)
        hits += bool(dark[top:bottom, left:right].any())
    return hits / max(len(positions), 1)


def image_payload(record: dict) -> bytes:
    image = record.get("image") or {}
    payload = image.get("bytes")
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("empty_image")
    return payload


def validate_candidate_assets(candidate: Candidate, record: dict) -> None:
    raw = image_payload(record)
    with Image.open(io.BytesIO(raw)) as opened:
        opened.load()
        source_size = opened.size
        resized = opened.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS,
        )
    positions = scaled_positions(candidate, *source_size)
    anchor_rate = foreground_anchor_rate(resized, positions)
    if anchor_rate < 0.95:
        raise ValueError(f"coordinate_image_mismatch:{anchor_rate:.4f}")
    mol = Chem.MolFromMolBlock(
        candidate.molfile,
        sanitize=True,
        removeHs=False,
        strictParsing=False,
    )
    molecule_id = f"molgrapher:{candidate.source_id}"
    chemmol = build_chemmol(mol, molecule_id, "prevalidation.sdf")
    render = build_render(mol, molecule_id, positions, source_size)
    serialize_chemmolseq(chemmol, render)
    chemsvg = serialize_chemsvg_v4(chemmol, render)
    validate_chemsvg_v4_roundtrip(
        chemsvg,
        chemmol["canonical_isomeric_smiles"],
        chemmol["inchikey"],
    )


def render_overlay(image: Image.Image, render: dict) -> bytes:
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    positions = {item["atom_id"]: (item["image_x"], item["image_y"]) for item in render["atoms"]}
    for bond in render["bonds"]:
        draw.line(
            (
                bond["begin_image_x"],
                bond["begin_image_y"],
                bond["end_image_x"],
                bond["end_image_y"],
            ),
            fill=(0, 160, 255),
            width=1,
        )
    for point in positions.values():
        x, y = point
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(255, 0, 0), width=2)
    output = io.BytesIO()
    overlay.save(output, format="PNG")
    return output.getvalue()


def selected_records(paths: list[Path], selected: list[Candidate]):
    wanted = {(item.source_file, item.source_row): item for item in selected}
    for path in paths:
        parquet = pq.ParquetFile(path)
        row_offset = 0
        for batch in parquet.iter_batches(batch_size=64, columns=["id", "image", "mol", "keypoints"]):
            rows = batch.to_pylist()
            for offset, row in enumerate(rows):
                key = (str(path), row_offset + offset)
                if key in wanted:
                    yield wanted[key], row
            row_offset += len(rows)


def write_split(
    output: Path,
    split: str,
    paths: list[Path],
    selected: list[Candidate],
    preview_limit: int,
) -> tuple[list[dict], Counter, float, int]:
    manifests = []
    counts = Counter()
    minimum_anchor_rate = 1.0
    preview_written = 0
    archives: dict[int, tarfile.TarFile] = {}
    try:
        for sample_index, (candidate, row) in enumerate(selected_records(paths, selected)):
            raw = image_payload(row)
            with Image.open(io.BytesIO(raw)) as opened:
                opened.load()
                source_size = opened.size
                resized = opened.convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    Image.Resampling.LANCZOS,
                )
            positions = scaled_positions(candidate, *source_size)
            anchor_rate = foreground_anchor_rate(resized, positions)
            if anchor_rate < 0.95:
                raise ValueError(
                    f"foreground anchor rate {anchor_rate:.4f} for {candidate.source_id}"
                )
            minimum_anchor_rate = min(minimum_anchor_rate, anchor_rate)
            mol = Chem.MolFromMolBlock(
                candidate.molfile,
                sanitize=True,
                removeHs=False,
                strictParsing=False,
            )
            molecule_id = f"molgrapher:{candidate.source_id}"
            sample_key = f"{split}_{sample_index:06d}"
            structure_ref = f"{sample_key}.sdf"
            chemmol = build_chemmol(mol, molecule_id, structure_ref)
            render = build_render(mol, molecule_id, positions, source_size)
            legacy_sequence = serialize_chemmolseq(chemmol, render)
            chemsvg = serialize_chemsvg_v4(chemmol, render)
            validate_chemsvg_v4_roundtrip(
                chemsvg,
                chemmol["canonical_isomeric_smiles"],
                chemmol["inchikey"],
            )
            sdf = candidate.molfile.rstrip("\r\n").encode("utf-8") + b"\n$$$$\n"
            png_buffer = io.BytesIO()
            resized.save(png_buffer, format="PNG")
            png = png_buffer.getvalue()
            meta = {
                "schema_version": "sample-meta-0.1",
                "sample_key": sample_key,
                "molecule_id": molecule_id,
                "source": "molgrapher-synthetic-300k",
                "source_revision": REVISION,
                "source_split": candidate.official_split,
                "source_file": Path(candidate.source_file).name,
                "source_row": candidate.source_row,
                "source_id": candidate.source_id,
                "split": split,
                "variant": "molgrapher_original",
                "structure_ref": structure_ref,
                "rdkit_version": rdBase.rdkitVersion,
                "seed": SEED,
                "chemmolseq_sha256": hashlib.sha256(legacy_sequence.encode("utf-8")).hexdigest(),
                "original_image_sha256": hashlib.sha256(raw).hexdigest(),
                "resized_image_sha256": hashlib.sha256(png).hexdigest(),
                "conversion_notice": "Original image resized to 768x768; coordinates transformed accordingly.",
            }
            shard_index = sample_index // SHARD_SIZE
            if shard_index not in archives:
                shard_path = output / "shards" / f"{split}-{shard_index:06d}.tar"
                archives[shard_index] = tarfile.open(shard_path, "w")
            archive = archives[shard_index]
            tar_add_bytes(archive, f"{sample_key}.png", png)
            tar_add_bytes(archive, f"{sample_key}.chemmol.json", json_bytes(chemmol))
            tar_add_bytes(archive, f"{sample_key}.render.json", json_bytes(render))
            tar_add_bytes(archive, f"{sample_key}.chemmolseq.txt", legacy_sequence.encode("utf-8"))
            tar_add_bytes(archive, f"{sample_key}.sdf", sdf)
            tar_add_bytes(archive, f"{sample_key}.meta.json", json_bytes(meta))
            if preview_written < preview_limit:
                (output / "previews" / f"{sample_key}.png").write_bytes(
                    render_overlay(resized, render)
                )
                preview_written += 1
            manifests.append(
                {
                    "sample_key": sample_key,
                    "molecule_id": molecule_id,
                    "split": split,
                    "source_id": candidate.source_id,
                    "source_file": Path(candidate.source_file).name,
                    "source_row": candidate.source_row,
                    "inchikey": candidate.inchikey,
                    "scaffold": candidate.scaffold,
                    "bucket": candidate.bucket,
                    "atoms": candidate.atoms,
                    "bonds": candidate.bonds,
                    "shard": f"shards/{split}-{shard_index:06d}.tar",
                    "foreground_anchor_rate": round(anchor_rate, 6),
                }
            )
            counts.update(
                {
                    "samples": 1,
                    f"bucket:{candidate.bucket}": 1,
                    "double_bond_samples": any(b.GetBondType() == Chem.BondType.DOUBLE for b in mol.GetBonds()),
                    "triple_bond_samples": any(b.GetBondType() == Chem.BondType.TRIPLE for b in mol.GetBonds()),
                    "stereo_samples": any(
                        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                        for atom in mol.GetAtoms()
                    ),
                }
            )
    finally:
        for archive in archives.values():
            archive.close()
    if len(manifests) != len(selected):
        raise RuntimeError(f"{split}: wrote {len(manifests)} of {len(selected)}")
    return manifests, counts, minimum_anchor_rate, preview_written


def write_preview_index(output: Path, rows: list[dict]) -> None:
    cards = []
    for row in rows:
        image = output / "previews" / f"{row['sample_key']}.png"
        if image.exists():
            cards.append(
                f'<article><img src="{html.escape(image.name)}" loading="lazy">'
                f'<code>{html.escape(row["sample_key"])} | {html.escape(row["bucket"])}</code></article>'
            )
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>MolGrapher pilot 预览</title><style>
body{font-family:sans-serif;margin:20px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
article{border:1px solid #ccc;padding:8px}img{width:100%;height:auto}code{display:block;overflow-wrap:anywhere}
</style><h1>MolGrapher pilot 坐标叠加</h1><main>""" + "".join(cards) + "</main></html>\n"
    (output / "previews" / "index.html").write_text(document, encoding="utf-8")


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_shards(output: Path, expected_samples: int) -> tuple[int, int]:
    members = 0
    samples = 0
    for path in sorted((output / "shards").glob("*.tar")):
        with tarfile.open(path) as archive:
            names = archive.getnames()
            members += len(names)
            samples += sum(name.endswith(".meta.json") for name in names)
            for member in archive:
                if member.isfile():
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Unreadable tar member: {path}:{member.name}")
                    stream.read()
    if samples != expected_samples or members != expected_samples * 6:
        raise ValueError(
            f"Tar validation mismatch: samples={samples}, members={members}"
        )
    return samples, members


def build(root: Path, output: Path, preview_samples: int) -> dict:
    started = time.perf_counter()
    source = json.loads((root / "sources/source.json").read_text(encoding="utf-8"))
    paths = {item["path"]: root / "raw" / item["path"] for item in source["files"]}
    for item in source["files"]:
        path = paths[item["path"]]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Source hash mismatch: {path}")
    train_paths = [path for name, path in paths.items() if "/train-" in name]
    validation_paths = [path for name, path in paths.items() if "/validation-" in name]
    excluded = benchmark_inchikeys(Path(__file__).resolve().parents[2])
    train_candidates, train_rejected = scan_candidates(
        train_paths,
        official_split="train",
        excluded_inchikeys=excluded,
    )
    selected_train = select_stratified(train_candidates, TRAIN_TARGET, "train")
    train_inchikeys = {item.inchikey for item in selected_train}
    train_scaffolds = {item.scaffold for item in selected_train}
    validation_candidates, validation_rejected = scan_candidates(
        validation_paths,
        official_split="validation",
        excluded_inchikeys=excluded | train_inchikeys,
    )
    validation_candidates = [
        item for item in validation_candidates if item.scaffold not in train_scaffolds
    ]
    selected_validation = select_stratified(
        validation_candidates,
        VALIDATION_TARGET,
        "validation",
    )

    staging = output.with_name(f".{output.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "shards").mkdir(parents=True)
    (staging / "previews").mkdir()
    train_rows, train_counts, train_min_anchor, train_previews = write_split(
        staging,
        "train",
        train_paths,
        selected_train,
        preview_samples,
    )
    validation_rows, validation_counts, validation_min_anchor, validation_previews = write_split(
        staging,
        "validation",
        validation_paths,
        selected_validation,
        max(0, preview_samples - train_previews),
    )
    rows = train_rows + validation_rows
    write_manifest(staging / "samples.csv", rows)
    rejected = train_rejected + validation_rejected
    with (staging / "rejected.jsonl").open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_preview_index(staging, rows)
    samples, members = validate_shards(staging, TRAIN_TARGET + VALIDATION_TARGET)
    scaffold_overlap = len(
        {row["scaffold"] for row in train_rows}
        & {row["scaffold"] for row in validation_rows}
    )
    inchikey_overlap = len(
        {row["inchikey"] for row in train_rows}
        & {row["inchikey"] for row in validation_rows}
    )
    summary = {
        "schema_version": "molgrapher-synthetic-pilot-0.1",
        "source_revision": REVISION,
        "license": "CDLA-Permissive-1.0",
        "image_size": IMAGE_SIZE,
        "samples": samples,
        "splits": {"train": len(train_rows), "validation": len(validation_rows)},
        "shards": len(list((staging / "shards").glob("*.tar"))),
        "tar_members": members,
        "source_candidates": {
            "train": len(train_candidates),
            "validation_scaffold_disjoint": len(validation_candidates),
        },
        "rejected": len(rejected),
        "train_counts": dict(train_counts),
        "validation_counts": dict(validation_counts),
        "minimum_foreground_anchor_rate": min(train_min_anchor, validation_min_anchor),
        "previews": train_previews + validation_previews,
        "elapsed_seconds": time.perf_counter() - started,
    }
    validation = {
        "status": "passed" if not inchikey_overlap and not scaffold_overlap else "failed",
        "samples": samples,
        "tar_members_readable": members,
        "inchikey_overlap_across_splits": inchikey_overlap,
        "scaffold_overlap_across_splits": scaffold_overlap,
        "benchmark_inchikey_overlap": 0,
        "minimum_foreground_anchor_rate": summary["minimum_foreground_anchor_rate"],
        "source_files_sha256_verified": len(source["files"]),
        "chemmol_chemsvg_sdf_roundtrip_passed": samples,
    }
    write_json(staging / "summary.json", summary)
    write_json(staging / "validation.json", validation)
    if validation["status"] != "passed":
        raise RuntimeError(f"Pilot validation failed: {validation}")
    if output.exists():
        shutil.rmtree(output)
    staging.replace(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Data/sdf/molgrapher_synthetic_v0_1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/sdf/molgrapher_synthetic_v0_1/pilot_10k"),
    )
    parser.add_argument("--preview-samples", type=int, default=500)
    args = parser.parse_args()
    if args.preview_samples < 0:
        raise ValueError("preview-samples must be nonnegative")
    RDLogger.DisableLog("rdApp.*")
    summary = build(args.root, args.output, args.preview_samples)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
