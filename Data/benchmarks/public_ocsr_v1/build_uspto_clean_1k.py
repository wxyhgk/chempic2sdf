#!/usr/bin/env python3
"""从官方 USPTO-30K clean 子集构建可复现的 1k 公开评测集。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil

from PIL import Image
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = Path(__file__).resolve().parent
DEFAULT_PARQUET = (
    BENCHMARK_ROOT
    / "raw/uspto_30k/data/clean-00000-of-00001-02aa6c7b94ad0558.parquet"
)
DEFAULT_OUTPUT = BENCHMARK_ROOT / "prepared/uspto_clean_1k"
SOURCE_REVISION = "58b3a773f01816d62126f11dec4d5e033e8ebf9d"
MANIFEST_FIELDS = (
    "molecule_id",
    "sample_id",
    "dataset",
    "subset",
    "official_split",
    "source_record_id",
    "source_row_index",
    "image_file",
    "sdf_file",
    "source_revision",
    "image_sha256",
    "sdf_sha256",
    "width",
    "height",
    "formula",
    "atoms",
    "bonds",
    "fragments",
    "has_charge",
    "has_isotope",
    "has_stereo",
    "status",
    "notes",
)
REJECTED_FIELDS = (
    "source_row_index",
    "source_record_id",
    "reason",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sdf_bytes(molfile: str) -> bytes:
    """只添加 SDF 记录分隔符，不让 RDKit 重写官方 Molfile。"""

    normalized = molfile.rstrip("\r\n") + "\n$$$$\n"
    return normalized.encode("utf-8")


def has_stereo(mol: Chem.Mol) -> bool:
    if any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in mol.GetAtoms()
    ):
        return True
    return any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        or bond.GetBondDir() != Chem.BondDir.NONE
        for bond in mol.GetBonds()
    )


def validate_record(record: dict) -> tuple[Chem.Mol, bytes, int, int, str]:
    image = record.get("image") or {}
    image_bytes = image.get("bytes")
    image_path = image.get("path")
    molfile = record.get("mol")
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("官方图片字节为空")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("官方图片路径为空")
    if not isinstance(molfile, str) or not molfile.strip():
        raise ValueError("官方 Molfile 为空")

    from io import BytesIO

    with Image.open(BytesIO(image_bytes)) as opened:
        opened.load()
        width, height = opened.size
        image_format = (opened.format or "png").lower()
    if width < 1 or height < 1:
        raise ValueError("官方图片尺寸非法")

    mol = Chem.MolFromMolBlock(
        molfile,
        sanitize=True,
        removeHs=False,
        strictParsing=False,
    )
    if mol is None:
        raise ValueError("RDKit 无法 sanitize 官方 Molfile")
    payload = sdf_bytes(molfile)
    roundtrip = Chem.MolFromMolBlock(
        payload.decode("utf-8").split("$$$$", 1)[0],
        sanitize=True,
        removeHs=False,
        strictParsing=False,
    )
    if roundtrip is None:
        raise ValueError("封装后的 SDF 无法读回")
    if (
        roundtrip.GetNumAtoms() != mol.GetNumAtoms()
        or roundtrip.GetNumBonds() != mol.GetNumBonds()
    ):
        raise ValueError("SDF 封装前后原子数或键数变化")
    return mol, payload, width, height, image_format


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build(parquet_path: Path, output_dir: Path, samples: int, seed: int) -> dict:
    if samples < 1:
        raise ValueError("samples 必须大于 0")
    table = pq.read_table(parquet_path, columns=["filename", "image", "mol"])
    if samples > table.num_rows:
        raise ValueError(f"要求 {samples} 条，但源数据只有 {table.num_rows} 条")

    permutation = list(range(table.num_rows))
    random.Random(seed).shuffle(permutation)
    staging = output_dir.with_name(f".{output_dir.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "images").mkdir(parents=True)
    (staging / "ground_truth").mkdir()

    manifest: list[dict] = []
    rejected: list[dict] = []
    for row_index in permutation:
        record = table.slice(row_index, 1).to_pylist()[0]
        source_id = record.get("filename") or f"row-{row_index:05d}"
        try:
            mol, sdf_payload, width, height, image_format = validate_record(record)
        except Exception as error:
            rejected.append(
                {
                    "source_row_index": row_index,
                    "source_record_id": source_id,
                    "reason": str(error),
                }
            )
            continue

        sample_id = f"uspto_clean_{len(manifest):04d}"
        image_name = f"{sample_id}.{image_format}"
        sdf_name = f"{sample_id}.sdf"
        image_payload = record["image"]["bytes"]
        image_path = staging / "images" / image_name
        sdf_path = staging / "ground_truth" / sdf_name
        image_path.write_bytes(image_payload)
        sdf_path.write_bytes(sdf_payload)

        manifest.append(
            {
                "molecule_id": sample_id,
                "sample_id": sample_id,
                "dataset": "uspto_30k",
                "subset": "uspto_10k",
                "official_split": "clean",
                "source_record_id": source_id,
                "source_row_index": row_index,
                "image_file": f"images/{image_name}",
                "sdf_file": f"ground_truth/{sdf_name}",
                "source_revision": SOURCE_REVISION,
                "image_sha256": sha256_bytes(image_payload),
                "sdf_sha256": sha256_bytes(sdf_payload),
                "width": width,
                "height": height,
                "formula": rdMolDescriptors.CalcMolFormula(mol),
                "atoms": mol.GetNumAtoms(),
                "bonds": mol.GetNumBonds(),
                "fragments": len(Chem.GetMolFrags(mol)),
                "has_charge": int(any(atom.GetFormalCharge() for atom in mol.GetAtoms())),
                "has_isotope": int(any(atom.GetIsotope() for atom in mol.GetAtoms())),
                "has_stereo": int(has_stereo(mol)),
                "status": "ready",
                "notes": "",
            }
        )
        if len(manifest) == samples:
            break

    if len(manifest) != samples:
        shutil.rmtree(staging)
        raise RuntimeError(
            f"只有 {len(manifest)} 条有效记录，无法交付要求的 {samples} 条"
        )

    write_csv(staging / "manifest.csv", MANIFEST_FIELDS, manifest)
    write_csv(staging / "rejected.csv", REJECTED_FIELDS, rejected)
    source_hash = sha256_file(parquet_path)
    summary = {
        "schema_version": "uspto-clean-sample-0.1",
        "dataset": "USPTO-30K",
        "subset": "clean",
        "source_revision": SOURCE_REVISION,
        "source_parquet": str(parquet_path.relative_to(PROJECT_ROOT)),
        "source_parquet_sha256": source_hash,
        "seed": seed,
        "requested_samples": samples,
        "delivered_samples": len(manifest),
        "rejected_candidates": len(rejected),
        "selection": "Python random.Random(seed).shuffle(range(10000)) 后取前 N 个可 sanitize 样本",
        "official_images_reencoded": False,
        "official_molfiles_rewritten": False,
    }
    write_json(staging / "summary.json", summary)

    validation_errors = []
    for row in manifest:
        image_path = staging / row["image_file"]
        sdf_path = staging / row["sdf_file"]
        if sha256_file(image_path) != row["image_sha256"]:
            validation_errors.append(f"{row['molecule_id']}: 图片 SHA256 不一致")
        if sha256_file(sdf_path) != row["sdf_sha256"]:
            validation_errors.append(f"{row['molecule_id']}: SDF SHA256 不一致")
        if Chem.MolFromMolFile(
            str(sdf_path), sanitize=True, removeHs=False, strictParsing=False
        ) is None:
            validation_errors.append(f"{row['molecule_id']}: SDF 无法 sanitize 读回")
    validation = {
        "status": "passed" if not validation_errors else "failed",
        "samples_checked": len(manifest),
        "image_hash_matches": len(manifest) - sum("图片" in e for e in validation_errors),
        "sdf_hash_matches": len(manifest) - sum("SDF SHA256" in e for e in validation_errors),
        "errors": validation_errors,
    }
    write_json(staging / "validation.json", validation)
    if validation_errors:
        shutil.rmtree(staging)
        raise RuntimeError(f"最终验收失败：{validation_errors[:3]}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.replace(output_dir)
    return summary | {"validation": validation["status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    result = build(args.parquet, args.output_dir, args.samples, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
