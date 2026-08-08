import torch

from chemvector.components.recovery.spatial_secondary_peak import spatial_offset_grid
from chemvector.evaluation.audits.atoms.audit_g3_p6e2_candidate_peak_reachability import (
    ranked_regional_maximum_offsets,
)


def test_p6e2_regional_maxima_are_score_ranked_and_target_free() -> None:
    logits = torch.full((1, 32, 32), -10.0)
    logits[0, 8, 12] = 3.0
    logits[0, 21, 18] = 5.0
    outputs = ranked_regional_maximum_offsets(logits, maximum_candidates=2)[0]
    grid = spatial_offset_grid()
    torch.testing.assert_close(outputs[0], grid[21, 18] / 768.0)
    torch.testing.assert_close(outputs[1], grid[8, 12] / 768.0)


def test_p6e2_regional_maxima_reject_invalid_budget() -> None:
    try:
        ranked_regional_maximum_offsets(torch.zeros(1, 32, 32), maximum_candidates=0)
    except ValueError as error:
        assert "budget" in str(error)
    else:
        raise AssertionError("invalid candidate budget was accepted")
