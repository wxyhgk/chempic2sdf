#!/usr/bin/env python3
"""Run the frozen v0.9.28 complete-object ChemScene shadow profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from chemvector.paths import PACKAGE_ROOT, PROJECT_ROOT

PROFILE_PATH = PACKAGE_ROOT / "configs/shadow_v0928.json"


def load_profile() -> dict[str, Any]:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if profile.get("id") != "shadow-v0928" or profile.get("version") != "v0.9.28":
        raise ValueError("v0.9.28 shadow profile identity differs")
    if profile.get("module") != "chemvector.infer_chemscene_hybrid":
        raise ValueError("v0.9.28 shadow module differs")
    return profile


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_profile(profile: dict[str, Any], *, verify_hashes: bool = True) -> None:
    manifest = profile.get("checkpoint_manifest")
    if not isinstance(manifest, list) or len(manifest) != 8:
        raise ValueError("v0.9.28 requires exactly eight frozen checkpoints")
    seen = set()
    for item in manifest:
        path = PROJECT_ROOT / str(item["path"])
        if path in seen or not path.is_file():
            raise FileNotFoundError(path)
        seen.add(path)
        if verify_hashes and file_sha256(path) != str(item["sha256"]):
            raise ValueError(f"Checkpoint SHA differs: {path}")


def inference_command(
    image: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
) -> list[str]:
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    profile = load_profile()
    return [
        sys.executable,
        "-m",
        str(profile["module"]),
        *[str(value) for value in profile["arguments"]],
        "--model-version",
        str(profile["version"]),
        "--image",
        str(Path(image).expanduser().resolve()),
        "--output-dir",
        str(Path(output_dir).expanduser().resolve()),
        "--device",
        device,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--skip-checkpoint-hashes", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    profile = load_profile()
    validate_profile(profile, verify_hashes=not args.skip_checkpoint_hashes)
    command = inference_command(args.image, args.output_dir, device=args.device)
    if args.print_command:
        print(json.dumps(command, ensure_ascii=False))
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()


__all__ = ["inference_command", "load_profile", "validate_profile"]
