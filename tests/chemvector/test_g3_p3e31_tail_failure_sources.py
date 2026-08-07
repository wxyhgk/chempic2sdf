from __future__ import annotations

import pytest

from chemvector.evaluation.audits.atoms.audit_g3_p3e31_tail_failure_sources import (
    atom_visual_role,
    crowd_bucket,
    direction_from_cases,
    rank_bucket,
)


def test_atom_visual_role_separates_text_and_implicit_degree() -> None:
    assert atom_visual_role(visible_text=True, degree=3) == "visible_text"
    assert atom_visual_role(visible_text=False, degree=1) == "implicit_endpoint"
    assert atom_visual_role(visible_text=False, degree=2) == "implicit_chain"
    assert atom_visual_role(visible_text=False, degree=3) == "implicit_junction"


def test_rank_and_crowd_buckets_have_fixed_boundaries() -> None:
    assert rank_bucket(129) == "129-144"
    assert rank_bucket(192) == "177-192"
    with pytest.raises(ValueError):
        rank_bucket(128)
    assert crowd_bucket(0.49) == "lt0.50"
    assert crowd_bucket(0.5) == "0.50-0.75"
    assert crowd_bucket(1.0) == "ge1.00"


def test_direction_prefers_text_majority() -> None:
    cases = [
        {"role": "visible_text", "variant": "rdkit"},
        {"role": "visible_text", "variant": "stress"},
        {"role": "implicit_chain", "variant": "rdkit"},
    ]
    result = direction_from_cases(cases)
    assert result["direction"] == "text_aware_candidate_generator"


def test_direction_falls_back_to_mixed_generator() -> None:
    cases = [
        {"role": "visible_text", "variant": "rdkit"},
        {"role": "implicit_chain", "variant": "stress"},
        {"role": "implicit_junction", "variant": "original"},
        {"role": "implicit_endpoint", "variant": "coordgen"},
    ]
    result = direction_from_cases(cases)
    assert result["direction"] == "mixed_candidate_generator_ranking"
