"""Write the annotation-free observation and factorized R0 artifact."""

from __future__ import annotations

from pathlib import Path

from .build_observation import build_observation
from .candidate_domain import build_candidate_domain, ensure_constructor_has_no_oracle_literal
from .common import canonical_bytes


ROOT = Path(__file__).resolve().parent


def build() -> None:
    ensure_constructor_has_no_oracle_literal()
    observation = build_observation()
    candidate = build_candidate_domain(observation)
    ROOT.joinpath("r0_observation.json").write_bytes(canonical_bytes(observation))
    ROOT.joinpath("r0_candidate_domain.json").write_bytes(canonical_bytes(candidate))


if __name__ == "__main__":
    build()
