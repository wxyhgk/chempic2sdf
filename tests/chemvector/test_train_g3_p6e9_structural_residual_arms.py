import torch

from chemvector.training.atoms.train_g3_p6e9_structural_residual_arms import (
    ARMS, CAP, EXPOSURES, HARD_ACTIONS, STEPS, matched_easy_rows,
)


def test_p6e9_protocol_is_frozen() -> None:
    assert ARMS == ("global", "local17", "multiscale")
    assert CAP == 2.0 and HARD_ACTIONS == 5171 and EXPOSURES == 192 and STEPS == 31026


def test_p6e9_easy_matching_reuses_only_same_stratum() -> None:
    hard={"offset_targets":torch.zeros(2,2),"variant":["stress"]*2,"target_role":["IMPLICIT"]*2,"degree_bucket":["degree1"]*2}
    easy={"offset_targets":torch.zeros(1,2),"variant":["stress"],"target_role":["IMPLICIT"],"degree_bucket":["degree1"],"sample_key":["a"],"target_index":[0]}
    assert matched_easy_rows(hard,easy,1).tolist()==[0,0]
