"""Experiment provenance and content hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from chemvector.experiment_loop.state import atomic_write_json


IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_paths(project_root: Path, values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    files: list[Path] = []
    for value in values:
        path = (project_root / value).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            files.append(path)
        else:
            files.extend(
                item
                for item in path.rglob("*")
                if item.is_file()
                and not any(part in IGNORED_PARTS for part in item.parts)
                and item.suffix != ".pyc"
            )
    for path in sorted(set(files)):
        relative = path.relative_to(project_root)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def command_output(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or result.stderr.strip() or None


def environment_manifest(project_root: Path) -> dict:
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"], project_root),
        "git_status": command_output(["git", "status", "--short"], project_root),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            project_root,
        ),
    }
    try:
        import torch

        manifest.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            }
        )
    except ImportError:
        manifest["torch"] = None
    try:
        from rdkit import rdBase

        manifest["rdkit"] = rdBase.rdkitVersion
    except ImportError:
        manifest["rdkit"] = None
    return manifest


def write_manifest(
    path: Path,
    *,
    project_root: Path,
    campaign_path: Path,
    source_paths: tuple[str, ...],
    input_artifacts: tuple[str, ...],
) -> dict:
    artifact_hashes = {}
    for value in input_artifacts:
        artifact = (project_root / value).resolve()
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        artifact_hashes[value] = sha256_file(artifact)
    manifest = {
        "campaign_file": str(campaign_path),
        "campaign_sha256": sha256_file(campaign_path),
        "source_paths": list(source_paths),
        "source_sha256": hash_paths(project_root, source_paths),
        "input_artifacts": artifact_hashes,
        "environment": environment_manifest(project_root),
    }
    atomic_write_json(path, manifest)
    return manifest


def verify_manifest(path: Path, *, project_root: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    source_paths = tuple(manifest["source_paths"])
    current_source = hash_paths(project_root, source_paths)
    if current_source != manifest["source_sha256"]:
        raise ValueError("Campaign source files changed after initialization")
    for value, expected in manifest.get("input_artifacts", {}).items():
        artifact = (project_root / value).resolve()
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError(f"Campaign input artifact changed: {value}")
    return manifest
