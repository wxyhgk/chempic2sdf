"""Build the frozen R0 matrix and representative witness."""

from __future__ import annotations

from pathlib import Path

from .ownership_realization import build_matrix, canonical_bytes, representative_witness


ROOT = Path(__file__).resolve().parent


def build() -> None:
    matrix = build_matrix()
    witness, png = representative_witness()
    ROOT.joinpath("results.json").write_bytes(canonical_bytes(matrix))
    ROOT.joinpath("representative_witness.json").write_bytes(canonical_bytes(witness))
    ROOT.joinpath("representative_edge_w5_round_box4.png").write_bytes(png)


if __name__ == "__main__":
    build()
