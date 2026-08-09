from chemvector.training.atoms.run_g3_p6_residual_long_cycle import cycle_decision


def test_p6_long_cycle_stops_without_safe_winner() -> None:
    assert cycle_decision({"decision":"residual_cycle_data_go"},{"decision":"residual_calibration_no_winner"})=="stopped_p6e9_no_safe_winner"
    assert cycle_decision({"decision":"residual_cycle_data_go"},{"decision":"residual_calibration_winner_selected"})=="ready_for_p6e10_holdout"
