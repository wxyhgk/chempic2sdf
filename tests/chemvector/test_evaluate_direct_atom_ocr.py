from chemvector.evaluate_direct_atom_ocr import atom_state, summarize


def test_atom_state_parses_visible_labels_and_implicit_carbon() -> None:
    assert atom_state("") == ("C", 0, 0, 0, 0)
    assert atom_state("NH2+") == ("N", 1, 0, 2, 0)
    assert atom_state("Ph") is None


def test_ocr_summary_requires_complete_atom_set_for_exact() -> None:
    records = [
        {
            "expected": ["", "N"],
            "predicted": {"radius_10": [("", 0.0), ("N", -0.1)]},
            "target_atoms": 2,
            "selected_atoms": 3,
            "matched_atoms": 2,
        }
    ]
    result = summarize(records, "radius_10")
    assert result["state_accuracy"] == 1.0
    assert result["state_exact_rate"] == 0.0
