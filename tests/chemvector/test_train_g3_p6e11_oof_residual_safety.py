from chemvector.training.atoms.train_g3_p6e11_oof_residual_safety import FOLDS,ResidualSafetyMLP,scaffold_fold


def test_p6e11_protocol_and_fold_are_fixed() -> None:
    assert FOLDS==5 and scaffold_fold("c1ccccc1")==scaffold_fold("c1ccccc1")
    assert sum(value.numel() for value in ResidualSafetyMLP(26).parameters())<2000
