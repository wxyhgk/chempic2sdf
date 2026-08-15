"""Build byte-replayable ownership non-identifiability artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .ownership_worlds import (
    build_bridge_support_input,
    build_bridge_witness,
    build_manifest,
    canonical_bytes,
)


OUTPUT_ROOT = Path(__file__).resolve().parent / "counterexamples"


def build_values() -> dict[str, Any]:
    support_input = build_bridge_support_input()
    observation, world_a, world_b = build_bridge_witness()
    manifest = build_manifest(support_input, observation, world_a, world_b)
    return {
        "support_input.json": support_input,
        "observation.json": observation,
        "world_a.json": world_a,
        "world_b.json": world_b,
        "manifest.json": manifest,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for filename, value in build_values().items():
        (OUTPUT_ROOT / filename).write_bytes(canonical_bytes(value))


if __name__ == "__main__":
    main()
