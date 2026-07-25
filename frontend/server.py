#!/usr/bin/env python3
"""Serve benchmark results and controlled single-image ChemVector inference."""

from __future__ import annotations

import argparse
from io import BytesIO
import csv
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import secrets
import subprocess
import time
from urllib.parse import unquote, urlparse

from PIL import Image
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D


FRONTEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_ROOT.parent
UPLOAD_ROOT = PROJECT_ROOT / "outputs" / "uploads"
PROFILE_REGISTRY_PATH = FRONTEND_ROOT / "inference_profiles.json"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
SHADOW_V0928_PROFILE = json.loads(
    (PROJECT_ROOT / "chemvector" / "shadow_v0928.json").read_text(encoding="utf-8")
)
INFERENCE_PROFILES = {
    "shadow-v0928": SHADOW_V0928_PROFILE,
    "development-v0823": {
        "id": "development-v0823",
        "version": "v0.8.23",
        "label": "实验结构化原子解码 v0.8.23",
        "checkpoint": "outputs/checkpoints/chemvector-v0821-junction-aware-bond-selector-9k/best.pt",
        "protocol": "v0.8.22 固定骨架 + 双 crop CTC top-k + 失败驱动原子状态重排",
        "available": True,
        "recommended": False,
        "module": "chemvector.infer_chemscene_hybrid",
        "arguments": [
            "--selector-checkpoint",
            "outputs/checkpoints/chemvector-v0739-boundary-group-selector-9k-15k/best.pt",
            "--bond-checkpoint",
            "outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
            "--bond-set-selector-checkpoint",
            "outputs/checkpoints/chemvector-v0821-junction-aware-bond-selector-9k/best.pt",
            "--endpoint-grouper-checkpoint",
            "outputs/checkpoints/chemvector-v0820-visual-endpoint-junction-grouper-capacity500/best.pt",
            "--line-band-checkpoint",
            "outputs/checkpoints/chemvector-v0738-node-conditioned-bond-jitter-9k-15k/end-to-end-best.pt",
            "--graph-corrector-checkpoint",
            "outputs/checkpoints/chemvector-v085-chemscene-graph-corrector-2k/best.pt",
            "--graph-correction-mode",
            "type_only",
            "--ctc-ocr-checkpoint",
            "outputs/checkpoints/chemvector-v088-pretrained-bbox-ctc-10k/best.pt",
            "--text-bbox-checkpoint",
            "outputs/checkpoints/chemvector-v089-text-bbox-offset-9k-45k/best.pt",
            "--ctc-anchor-fallback-margin",
            "0.25",
            "--junction-source",
            "junction_aware_bonds",
            "--text-attach-radius-pixels",
            "24",
            "--atom-decode-mode",
            "structured_invalid_only",
        ],
    },
    "development-v0822": {
        "id": "development-v0822",
        "version": "v0.8.22",
        "label": "实验 Junction-Aware Bond v0.8.22",
        "checkpoint": "outputs/checkpoints/chemvector-v0821-junction-aware-bond-selector-9k/best.pt",
        "protocol": "top-128 Bond keep/drop + 方向对齐视觉 Junction + OCR 标签附着",
        "available": True,
        "recommended": False,
        "module": "chemvector.infer_chemscene_hybrid",
        "arguments": [
            "--selector-checkpoint",
            "outputs/checkpoints/chemvector-v0739-boundary-group-selector-9k-15k/best.pt",
            "--bond-checkpoint",
            "outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
            "--bond-set-selector-checkpoint",
            "outputs/checkpoints/chemvector-v0821-junction-aware-bond-selector-9k/best.pt",
            "--endpoint-grouper-checkpoint",
            "outputs/checkpoints/chemvector-v0820-visual-endpoint-junction-grouper-capacity500/best.pt",
            "--line-band-checkpoint",
            "outputs/checkpoints/chemvector-v0738-node-conditioned-bond-jitter-9k-15k/end-to-end-best.pt",
            "--graph-corrector-checkpoint",
            "outputs/checkpoints/chemvector-v085-chemscene-graph-corrector-2k/best.pt",
            "--graph-correction-mode",
            "type_only",
            "--ctc-ocr-checkpoint",
            "outputs/checkpoints/chemvector-v088-pretrained-bbox-ctc-10k/best.pt",
            "--text-bbox-checkpoint",
            "outputs/checkpoints/chemvector-v089-text-bbox-offset-9k-45k/best.pt",
            "--ctc-anchor-fallback-margin",
            "0.25",
            "--junction-source",
            "junction_aware_bonds",
            "--text-attach-radius-pixels",
            "24",
        ],
    },
    "development-v0816": {
        "id": "development-v0816",
        "version": "v0.8.16",
        "label": "实验语义 Bond 端点 v0.8.16",
        "checkpoint": "outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
        "protocol": "BondObject 两端点聚合 Junction + OCR 仅附着标签",
        "available": True,
        "recommended": False,
        "module": "chemvector.infer_chemscene_hybrid",
        "arguments": [
            "--selector-checkpoint",
            "outputs/checkpoints/chemvector-v0739-boundary-group-selector-9k-15k/best.pt",
            "--bond-checkpoint",
            "outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
            "--graph-corrector-checkpoint",
            "outputs/checkpoints/chemvector-v085-chemscene-graph-corrector-2k/best.pt",
            "--graph-correction-mode",
            "type_only",
            "--ctc-ocr-checkpoint",
            "outputs/checkpoints/chemvector-v088-pretrained-bbox-ctc-10k/best.pt",
            "--text-bbox-checkpoint",
            "outputs/checkpoints/chemvector-v089-text-bbox-offset-9k-45k/best.pt",
            "--ctc-anchor-fallback-margin",
            "0.25",
            "--junction-source",
            "bond_endpoints",
            "--endpoint-cluster-radius-pixels",
            "12",
            "--text-attach-radius-pixels",
            "24",
        ],
    },
    "development-v089": {
        "id": "development-v089",
        "version": "v0.8.9",
        "label": "实验 ChemScene v0.8.9",
        "checkpoint": "outputs/checkpoints/chemvector-v089-text-bbox-offset-9k-45k/best.pt",
        "protocol": "非对称文字框 + 保守双 crop CTC OCR",
        "available": True,
        "recommended": False,
        "module": "chemvector.infer_chemscene_hybrid",
        "arguments": [
            "--selector-checkpoint",
            "outputs/checkpoints/chemvector-v0739-boundary-group-selector-9k-15k/best.pt",
            "--bond-checkpoint",
            "outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
            "--graph-corrector-checkpoint",
            "outputs/checkpoints/chemvector-v085-chemscene-graph-corrector-2k/best.pt",
            "--graph-correction-mode",
            "type_only",
            "--ctc-ocr-checkpoint",
            "outputs/checkpoints/chemvector-v088-pretrained-bbox-ctc-10k/best.pt",
            "--text-bbox-checkpoint",
            "outputs/checkpoints/chemvector-v089-text-bbox-offset-9k-45k/best.pt",
            "--ctc-anchor-fallback-margin",
            "0.25",
            "--assignment-mode",
            "nearest",
        ],
    },
    "development-v088": {
        "id": "development-v088",
        "version": "v0.8.8",
        "label": "实验 ChemScene v0.8.8",
        "checkpoint": "outputs/checkpoints/chemvector-v088-pretrained-bbox-ctc-10k/best.pt",
        "protocol": "v0.8.5 graph type-only + bbox pretrained CTC OCR",
        "available": True,
        "recommended": False,
        "module": "chemvector.infer_chemscene_hybrid",
        "arguments": [
            "--selector-checkpoint",
            "outputs/checkpoints/chemvector-v0739-boundary-group-selector-9k-15k/best.pt",
            "--bond-checkpoint",
            "outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
            "--graph-corrector-checkpoint",
            "outputs/checkpoints/chemvector-v085-chemscene-graph-corrector-2k/best.pt",
            "--graph-correction-mode",
            "type_only",
            "--ctc-ocr-checkpoint",
            "outputs/checkpoints/chemvector-v088-pretrained-bbox-ctc-10k/best.pt",
            "--assignment-mode",
            "nearest",
        ],
    },
    "development-v086": {
        "id": "development-v086",
        "version": "v0.8.6",
        "label": "实验 ChemScene v0.8.6",
        "checkpoint": "outputs/checkpoints/chemvector-v086-ocr-modifier-color-500/best.pt",
        "protocol": "v0.8.5 graph type-only + modifier/color OCR adapter",
        "available": True,
        "recommended": False,
        "module": "chemvector.infer_chemscene_hybrid",
        "arguments": [
            "--selector-checkpoint",
            "outputs/checkpoints/chemvector-v0739-boundary-group-selector-9k-15k/best.pt",
            "--bond-checkpoint",
            "outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
            "--graph-corrector-checkpoint",
            "outputs/checkpoints/chemvector-v085-chemscene-graph-corrector-2k/best.pt",
            "--graph-correction-mode",
            "type_only",
            "--ocr-checkpoint",
            "outputs/checkpoints/chemvector-v086-ocr-modifier-color-500/best.pt",
            "--assignment-mode",
            "nearest",
        ],
    },
    "development-v085": {
        "id": "development-v085",
        "version": "v0.8.5",
        "label": "实验 ChemScene v0.8.5",
        "checkpoint": "outputs/checkpoints/chemvector-v085-chemscene-graph-corrector-2k/best.pt",
        "protocol": "predicted Junction + multiscale BondObject + graph type-only correction",
        "available": True,
        "recommended": False,
        "module": "chemvector.infer_chemscene_hybrid",
        "arguments": [
            "--selector-checkpoint",
            "outputs/checkpoints/chemvector-v0739-boundary-group-selector-9k-15k/best.pt",
            "--bond-checkpoint",
            "outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
            "--graph-corrector-checkpoint",
            "outputs/checkpoints/chemvector-v085-chemscene-graph-corrector-2k/best.pt",
            "--graph-correction-mode",
            "type_only",
            "--assignment-mode",
            "nearest",
        ],
    },
    "development-v083": {
        "id": "development-v083",
        "version": "v0.8.3",
        "label": "新版 ChemScene v0.8.3",
        "checkpoint": "outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
        "protocol": "predicted Junction + Pixel OCR + multiscale BondObject + ChemScene",
        "available": True,
        "recommended": False,
        "module": "chemvector.infer_chemscene_hybrid",
        "arguments": [
            "--selector-checkpoint",
            "outputs/checkpoints/chemvector-v0739-boundary-group-selector-9k-15k/best.pt",
            "--bond-checkpoint",
            "outputs/checkpoints/chemvector-v082-multiscale-joint-9k-1epoch/best.pt",
            "--assignment-mode",
            "nearest",
        ],
    },
    "stable-v0710": {
        "id": "stable-v0710",
        "version": "v0.7.10",
        "label": "稳定端到端 v0.7.10",
        "checkpoint": "outputs/checkpoints/chemvector-v079-relation-all-pairs-9k/last.pt",
        "protocol": "parallel-v6 + probability bridge",
        "available": True,
        "recommended": True,
        "module": "chemvector.infer_parallel_v6",
        "arguments": [
            "--edge-threshold",
            "0.4",
            "--repair-candidate-mode",
            "model_connect",
        ],
    },
}


def infer_model_version(run_id: str, config: dict) -> str | None:
    """Return an explicit version, with a conservative run-name fallback."""

    explicit = config.get("model_version") or config.get("version")
    if isinstance(explicit, str) and explicit.strip():
        value = explicit.strip()
        return value if value.startswith("v") else f"v{value}"
    for value in (run_id, str(config.get("checkpoint", ""))):
        match = re.search(r"(?:^|[-_/])v(0?\d{2,4})(?:[-_/]|$)", value)
        if not match:
            continue
        digits = match.group(1)
        if digits.startswith("0"):
            digits = digits[1:]
        if len(digits) >= 2:
            return f"v0.{digits[0]}.{int(digits[1:])}"
    return None


def model_metadata(run_id: str, config: dict) -> dict:
    checkpoint = config.get("checkpoint")
    version = infer_model_version(run_id, config)
    protocol_parts = [str(config["inference_protocol"])] if config.get("inference_protocol") else []
    if not protocol_parts and version == "v0.7.10":
        protocol_parts.append("parallel-v6 + probability bridge")
    if config.get("repair_candidate_mode"):
        protocol_parts.append(str(config["repair_candidate_mode"]))
    if config.get("benchmark_type"):
        protocol_parts.append(str(config["benchmark_type"]))
    return {
        "version": version,
        "checkpoint": checkpoint,
        "checkpoint_sha256": config.get("checkpoint_sha256"),
        "protocol": " · ".join(protocol_parts) or None,
    }


def load_inference_profiles() -> dict[str, dict]:
    """Merge the runtime profile registry without restarting the web server."""

    profiles = {profile_id: dict(profile) for profile_id, profile in INFERENCE_PROFILES.items()}
    if not PROFILE_REGISTRY_PATH.is_file():
        return profiles
    payload = json.loads(PROFILE_REGISTRY_PATH.read_text(encoding="utf-8"))
    runtime = payload.get("profiles", payload)
    if not isinstance(runtime, dict):
        raise ValueError("inference_profiles.json 的 profiles 必须是对象")
    for profile_id, override in runtime.items():
        if not isinstance(profile_id, str) or not isinstance(override, dict):
            raise ValueError("inference_profiles.json 包含无效 profile")
        profile = {**profiles.get(profile_id, {}), **override, "id": profile_id}
        required = {"version", "label", "checkpoint", "protocol", "module", "arguments"}
        missing = sorted(required - profile.keys())
        if missing:
            raise ValueError(f"Profile {profile_id} 缺少字段：{', '.join(missing)}")
        profile.setdefault("available", True)
        profile.setdefault("recommended", False)
        profiles[profile_id] = profile
    return profiles


def public_inference_profiles() -> list[dict]:
    return [
        {
            key: value
            for key, value in profile.items()
            if key not in {"arguments", "module"}
        }
        for profile in load_inference_profiles().values()
    ]


def humanize_inference_error(message: str) -> str:
    valence = re.search(
        r"Explicit valence for atom #\s*(\d+)\s+([A-Z][a-z]?),\s*(\d+)",
        message,
    )
    if valence:
        atom_id, element, value = valence.groups()
        return (
            f"模型生成了无效价态：原子 #{atom_id} 被识别为 {element}，"
            f"连接价态为 {value}，RDKit 未生成有效 SDF。"
        )
    disconnected = re.search(
        r"Ink graph is not connected:\s*(\d+) components for (\d+) vertices",
        message,
    )
    if disconnected:
        components, vertices = disconnected.groups()
        return f"模型预测图不连通：{vertices} 个顶点形成了 {components} 个分量，未生成有效 SDF。"
    if "kekul" in message.lower():
        return "模型生成的芳香环无法完成 Kekulé/价态校验，未生成有效 SDF。"
    return "模型未能生成有效 SDF。完整错误已保存在本次上传目录的 inference.log。"


def sample_status(record: dict) -> str:
    if record.get("chemical_match"):
        return "exact"
    if not record.get("sdf_valid"):
        return "invalid"
    if record.get("atom_count_match") and record.get("bond_count_match"):
        return "similar"
    return "mismatch"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_uploaded_image(body: bytes) -> Image.Image:
    """Decode an upload without discarding transparency before preprocessing."""

    try:
        with Image.open(BytesIO(body)) as opened:
            opened.load()
            return opened.convert("RGBA")
    except Exception as error:
        raise ValueError("无法读取上传图片") from error


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Path is outside the project: {value}")
    return resolved


def load_sdf(path: Path, *, sanitize: bool = True) -> Chem.Mol:
    mol = Chem.MolFromMolFile(
        str(path),
        sanitize=sanitize,
        removeHs=False,
        strictParsing=False,
    )
    if mol is None:
        raise ValueError(f"Cannot read SDF: {path}")
    if not sanitize:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    return mol


def fingerprint_similarity(expected: Chem.Mol, predicted: Chem.Mol) -> float:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return float(
        DataStructs.TanimotoSimilarity(
            generator.GetFingerprint(expected),
            generator.GetFingerprint(predicted),
        )
    )


def sdf_comparison(expected_path: Path, predicted_path: Path) -> dict:
    expected = load_sdf(expected_path)
    predicted = load_sdf(predicted_path)
    result = rdFMCS.FindMCS(
        [expected, predicted],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        timeout=2,
    )
    return {
        "fingerprint_similarity": fingerprint_similarity(expected, predicted),
        "mcs_atoms": result.numAtoms,
        "mcs_bonds": result.numBonds,
        "mcs_timed_out": bool(result.canceled),
        "expected_atom_coverage": result.numAtoms / max(expected.GetNumAtoms(), 1),
        "predicted_atom_coverage": result.numAtoms / max(predicted.GetNumAtoms(), 1),
        "expected_bond_coverage": result.numBonds / max(expected.GetNumBonds(), 1),
        "predicted_bond_coverage": result.numBonds / max(predicted.GetNumBonds(), 1),
    }


def render_sdf(sdf_path: Path, output_path: Path) -> None:
    try:
        mol = load_sdf(sdf_path)
        sanitized = True
    except ValueError:
        mol = load_sdf(sdf_path, sanitize=False)
        sanitized = False
    drawer = rdMolDraw2D.MolDraw2DCairo(768, 768)
    options = drawer.drawOptions()
    options.prepareMolsBeforeDrawing = sanitized
    options.bondLineWidth = 2.0
    options.fixedFontSize = 16
    options.padding = 0.08
    options.addStereoAnnotation = False
    options.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
    options.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(drawer.GetDrawingText())
    temporary.replace(output_path)


class BenchmarkRepository:
    def __init__(self, evaluations_root: Path) -> None:
        self.evaluations_root = evaluations_root.resolve()

    def run_directories(self) -> list[Path]:
        if not self.evaluations_root.is_dir():
            return []
        runs = []
        for path in self.evaluations_root.iterdir():
            if not path.is_dir():
                continue
            required = ("summary.json", "benchmark_config.json", "records.jsonl")
            if not all((path / name).is_file() for name in required):
                continue
            try:
                config = load_json(path / "benchmark_config.json")
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(config.get("benchmark_manifest"), str):
                continue
            runs.append(path)
        return sorted(
            runs,
            key=lambda path: (path / "records.jsonl").stat().st_mtime,
            reverse=True,
        )

    def run_directory(self, run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id:
            raise FileNotFoundError(run_id)
        path = (self.evaluations_root / run_id).resolve()
        if not path.is_relative_to(self.evaluations_root) or not path.is_dir():
            raise FileNotFoundError(run_id)
        return path

    def load_manifest(self, config: dict) -> tuple[Path, dict[str, dict]]:
        manifest_path = resolve_project_path(config["benchmark_manifest"])
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return manifest_path.parent, {row["molecule_id"]: row for row in rows}

    def load_run(self, run_id: str) -> dict:
        run_dir = self.run_directory(run_id)
        config = load_json(run_dir / "benchmark_config.json")
        summary = load_json(run_dir / "summary.json")
        records = load_jsonl(run_dir / "records.jsonl")
        object_decomposition_path = run_dir / "object_decomposition.json"
        object_decomposition = (
            load_json(object_decomposition_path)
            if object_decomposition_path.is_file()
            else None
        )
        benchmark_root, manifest = self.load_manifest(config)
        return {
            "id": run_id,
            "directory": run_dir,
            "config": config,
            "summary": summary,
            "object_decomposition": object_decomposition,
            "records": records,
            "record_index": {row["molecule_id"]: row for row in records},
            "benchmark_root": benchmark_root,
            "manifest": manifest,
            "updated_at": datetime.fromtimestamp(
                (run_dir / "records.jsonl").stat().st_mtime
            ).astimezone().isoformat(timespec="seconds"),
        }

    def list_runs(self) -> list[dict]:
        payload = []
        for path in self.run_directories():
            run = self.load_run(path.name)
            statuses = {"exact": 0, "similar": 0, "mismatch": 0, "invalid": 0}
            for record in run["records"]:
                statuses[sample_status(record)] += 1
            payload.append(
                {
                    "id": run["id"],
                    "updated_at": run["updated_at"],
                    "samples": len(run["records"]),
                    "statuses": statuses,
                    "summary": run["summary"],
                    "object_decomposition": run["object_decomposition"],
                    "checkpoint": run["config"].get("checkpoint"),
                    "model": model_metadata(run["id"], run["config"]),
                }
            )
        return payload

    def run_payload(self, run_id: str) -> dict:
        run = self.load_run(run_id)
        samples = []
        for record in run["records"]:
            manifest = run["manifest"].get(record["molecule_id"], {})
            samples.append(
                {
                    "molecule_id": record["molecule_id"],
                    "status": sample_status(record),
                    "difficulty": record.get("difficulty"),
                    "sdf_valid": bool(record.get("sdf_valid")),
                    "chemical_match": bool(record.get("chemical_match")),
                    "fingerprint_similarity": record.get("fingerprint_similarity"),
                    "expected_formula": manifest.get("formula"),
                    "actual_formula": record.get("actual_formula"),
                    "expected_atoms": record.get("expected_atoms"),
                    "predicted_atoms": record.get("predicted_atoms"),
                    "expected_bonds": record.get("expected_bonds"),
                    "predicted_bonds": record.get("predicted_bonds"),
                    "failure_category": record.get("failure_category"),
                    "primary_error": record.get("primary_error"),
                    "error_flags": record.get("error_flags", []),
                    "diagnostics": record.get("diagnostics"),
                    "error": record.get("error"),
                    "seconds": record.get("seconds"),
                }
            )
        return {
            "id": run["id"],
            "updated_at": run["updated_at"],
            "config": run["config"],
            "model": model_metadata(run["id"], run["config"]),
            "summary": run["summary"],
            "object_decomposition": run["object_decomposition"],
            "samples": samples,
        }

    def prediction_sdf(self, run: dict, molecule_id: str) -> Path | None:
        record = run["record_index"].get(molecule_id, {})
        relative = record.get("prediction_sdf_file")
        if relative:
            path = (run["directory"] / relative).resolve()
            if path.is_relative_to(run["directory"]) and path.is_file():
                return path
        legacy = run["directory"] / "sdf" / f"{molecule_id}.sdf"
        return legacy if legacy.is_file() else None

    def prediction_render(self, run: dict, molecule_id: str) -> Path | None:
        record = run["record_index"].get(molecule_id, {})
        relative = record.get("prediction_render_file")
        if relative:
            path = (run["directory"] / relative).resolve()
            if path.is_relative_to(run["directory"]) and path.is_file():
                return path
        return None

    def detail_payload(self, run_id: str, molecule_id: str) -> dict:
        run = self.load_run(run_id)
        record = run["record_index"].get(molecule_id)
        manifest = run["manifest"].get(molecule_id)
        if record is None or manifest is None:
            raise FileNotFoundError(molecule_id)
        predicted_sdf = self.prediction_sdf(run, molecule_id)
        comparison = None
        if predicted_sdf is not None and record.get("sdf_valid"):
            comparison = sdf_comparison(
                run["benchmark_root"] / manifest["sdf_file"],
                predicted_sdf,
            )
        base = f"/api/artifacts/{run_id}/{molecule_id}"
        public_manifest = {
            key: value
            for key, value in manifest.items()
            if key != "canonical_isomeric_smiles"
        }
        return {
            "record": record,
            "manifest": public_manifest,
            "status": sample_status(record),
            "comparison": comparison,
            "prediction_kind": (
                predicted_sdf.name.split("_", 1)[0]
                if predicted_sdf is not None and "_" in predicted_sdf.name
                else None
            ),
            "artifacts": {
                "ground_truth_image": f"{base}/ground-truth.png",
                "ground_truth_sdf": f"{base}/ground-truth.sdf",
                "prediction_image": (
                    f"{base}/prediction.png" if predicted_sdf is not None else None
                ),
                "prediction_sdf": (
                    f"{base}/prediction.sdf" if predicted_sdf is not None else None
                ),
                "raw_chemsvg": (
                    f"{base}/raw.chemsvg.txt"
                    if (
                        run["directory"]
                        / "raw_chemsvg"
                        / f"{molecule_id}.chemsvg.txt"
                    ).is_file()
                    else None
                ),
                "repaired_chemsvg": (
                    f"{base}/repaired.chemsvg.txt"
                    if (
                        run["directory"]
                        / "repaired_chemsvg"
                        / f"{molecule_id}.chemsvg.txt"
                    ).is_file()
                    else None
                ),
            },
        }

    def artifact_path(self, run_id: str, molecule_id: str, artifact: str) -> Path:
        run = self.load_run(run_id)
        manifest = run["manifest"].get(molecule_id)
        if manifest is None or molecule_id not in run["record_index"]:
            raise FileNotFoundError(molecule_id)
        if artifact == "ground-truth.png":
            path = run["benchmark_root"] / manifest["image_file"]
        elif artifact == "ground-truth.sdf":
            path = run["benchmark_root"] / manifest["sdf_file"]
        elif artifact == "prediction.sdf":
            path = self.prediction_sdf(run, molecule_id)
        elif artifact == "prediction.png":
            path = self.prediction_render(run, molecule_id)
            if path is None:
                sdf_path = self.prediction_sdf(run, molecule_id)
                if sdf_path is None:
                    path = None
                else:
                    path = (
                        run["directory"]
                        / "visualizations"
                        / "predicted_images"
                        / f"{molecule_id}.png"
                    )
                    if not path.is_file():
                        render_sdf(sdf_path, path)
        elif artifact == "raw.chemsvg.txt":
            path = (
                run["directory"]
                / "raw_chemsvg"
                / f"{molecule_id}.chemsvg.txt"
            )
        elif artifact == "repaired.chemsvg.txt":
            path = (
                run["directory"]
                / "repaired_chemsvg"
                / f"{molecule_id}.chemsvg.txt"
            )
        else:
            path = None
        if path is None or not path.is_file():
            raise FileNotFoundError(artifact)
        return path


class UploadRepository:
    def __init__(self, root: Path, *, device: str = "cuda") -> None:
        self.root = root.resolve()
        self.device = device

    def run_directory(self, run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id:
            raise FileNotFoundError(run_id)
        path = (self.root / run_id).resolve()
        if not path.is_relative_to(self.root) or not path.is_dir():
            raise FileNotFoundError(run_id)
        return path

    def profile(self, profile_id: str) -> dict:
        profile = load_inference_profiles().get(profile_id)
        if profile is None:
            raise FileNotFoundError(profile_id)
        if not profile["available"]:
            raise ValueError(profile["unavailable_reason"])
        checkpoint = resolve_project_path(profile["checkpoint"])
        if not checkpoint.is_file():
            raise ValueError(f"Checkpoint does not exist: {profile['checkpoint']}")
        return profile

    def infer(self, profile_id: str, body: bytes, content_type: str) -> dict:
        if not body:
            raise ValueError("上传图片为空")
        if len(body) > MAX_UPLOAD_BYTES:
            raise ValueError("上传图片超过 20 MB")
        if not content_type.startswith("image/"):
            raise ValueError("只接受图片文件")
        profile = self.profile(profile_id)
        image = decode_uploaded_image(body)
        if max(image.size) > 12000:
            raise ValueError("图片边长不能超过 12000 像素")

        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(4)
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        source_path = run_dir / "source.png"
        image.save(source_path)
        command = [
            str(PROJECT_ROOT / ".venv-train" / "bin" / "python"),
            "-m",
            profile["module"],
        ]
        if profile["module"] == "chemvector.infer_parallel_v6":
            command.extend(("--checkpoint", profile["checkpoint"]))
        elif profile["module"] == "chemvector.infer_chemscene_hybrid":
            command.extend(("--model-version", profile["version"]))
        command.extend(
            [
            "--image",
            str(source_path),
            "--output-dir",
            str(run_dir),
            "--device",
            self.device,
            *profile["arguments"],
            ]
        )
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        result = {
            "id": run_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model": {
                key: value
                for key, value in profile.items()
                if key not in {"arguments", "available", "module"}
            },
            "device": self.device,
            "wall_seconds": time.perf_counter() - started,
            "success": completed.returncode == 0,
            "error": None,
        }
        if completed.returncode == 0 and (run_dir / "metrics.json").is_file():
            result["metrics"] = load_json(run_dir / "metrics.json")
        else:
            message = completed.stderr.strip() or completed.stdout.strip()
            technical_error = message[-12000:] or "推理进程失败"
            (run_dir / "inference.log").write_text(
                technical_error + "\n", encoding="utf-8"
            )
            result["error"] = humanize_inference_error(technical_error)
        (run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return self.payload(run_id)

    def payload(self, run_id: str) -> dict:
        run_dir = self.run_directory(run_id)
        result = load_json(run_dir / "result.json")
        base = f"/api/uploads/{run_id}"
        result["artifacts"] = {
            "source_image": f"{base}/source.png",
            "prepared_image": (
                f"{base}/input-prepared.png"
                if (run_dir / "input-prepared.png").is_file()
                else None
            ),
            "prediction_image": (
                f"{base}/prediction.png"
                if (run_dir / "prediction-repaired.png").is_file()
                else None
            ),
            "prediction_sdf": (
                f"{base}/prediction.sdf"
                if (run_dir / "prediction-repaired.sdf").is_file()
                else None
            ),
            "raw_chemsvg": (
                f"{base}/raw.chemsvg.txt"
                if (run_dir / "prediction-parallel.chemsvg.txt").is_file()
                else None
            ),
            "repaired_chemsvg": (
                f"{base}/repaired.chemsvg.txt"
                if (run_dir / "prediction-repaired.chemsvg.txt").is_file()
                else None
            ),
            "chemscene": (
                f"{base}/scene.json"
                if (run_dir / "prediction.chemscene.json").is_file()
                else None
            ),
        }
        return result

    def artifact_path(self, run_id: str, artifact: str) -> Path:
        names = {
            "source.png": "source.png",
            "input-prepared.png": "input-prepared.png",
            "prediction.png": "prediction-repaired.png",
            "prediction.sdf": "prediction-repaired.sdf",
            "raw.chemsvg.txt": "prediction-parallel.chemsvg.txt",
            "repaired.chemsvg.txt": "prediction-repaired.chemsvg.txt",
            "scene.json": "prediction.chemscene.json",
        }
        filename = names.get(artifact)
        if filename is None:
            raise FileNotFoundError(artifact)
        path = self.run_directory(run_id) / filename
        if not path.is_file():
            raise FileNotFoundError(artifact)
        return path


class BenchmarkHandler(SimpleHTTPRequestHandler):
    repository: BenchmarkRepository
    uploads: UploadRepository

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
        try:
            if parts == ["api", "runs"]:
                self.send_json({"runs": self.repository.list_runs()})
                return
            if parts == ["api", "inference", "profiles"]:
                self.send_json({"profiles": public_inference_profiles()})
                return
            if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                self.send_json(self.repository.run_payload(parts[2]))
                return
            if (
                len(parts) == 5
                and parts[:2] == ["api", "runs"]
                and parts[3] == "samples"
            ):
                self.send_json(self.repository.detail_payload(parts[2], parts[4]))
                return
            if len(parts) == 5 and parts[:2] == ["api", "artifacts"]:
                self.send_file(
                    self.repository.artifact_path(parts[2], parts[3], parts[4])
                )
                return
            if len(parts) == 4 and parts[:2] == ["api", "uploads"]:
                self.send_file(self.uploads.artifact_path(parts[2], parts[3]))
                return
            self.serve_frontend(parsed.path)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as error:
            self.send_json({"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
        try:
            if len(parts) == 3 and parts[:2] == ["api", "infer"]:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > MAX_UPLOAD_BYTES:
                    raise ValueError("上传图片大小必须位于 1 字节到 20 MB 之间")
                payload = self.uploads.infer(
                    parts[2],
                    self.rfile.read(length),
                    self.headers.get("Content-Type", ""),
                )
                status = HTTPStatus.OK if payload["success"] else HTTPStatus.UNPROCESSABLE_ENTITY
                self.send_json(payload, status=status)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
        except ValueError as error:
            self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except subprocess.TimeoutExpired:
            self.send_json({"error": "推理超过 300 秒"}, status=HTTPStatus.GATEWAY_TIMEOUT)
        except Exception as error:
            self.send_json({"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def serve_frontend(self, request_path: str) -> None:
        relative = request_path.lstrip("/") or "index.html"
        path = (FRONTEND_ROOT / relative).resolve()
        if not path.is_relative_to(FRONTEND_ROOT) or not path.is_file():
            path = FRONTEND_ROOT / "index.html"
        self.send_file(path)

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in {".sdf", ".txt"}:
            content_type = "text/plain; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--evaluations-root", default="outputs/evaluations")
    parser.add_argument("--uploads-root", default="outputs/uploads")
    parser.add_argument("--inference-device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    BenchmarkHandler.repository = BenchmarkRepository(
        resolve_project_path(args.evaluations_root)
    )
    BenchmarkHandler.uploads = UploadRepository(
        resolve_project_path(args.uploads_root), device=args.inference_device
    )
    server = ThreadingHTTPServer((args.host, args.port), BenchmarkHandler)
    print(f"ChemVector benchmark browser: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
