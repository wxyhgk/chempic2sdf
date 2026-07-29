from chemvector.data import chemscene_atom_labels
from chemvector.evaluate_direct_atom_ocr import atom_state, deployment_gate, summarize


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


def test_chemscene_atom_labels_follow_atom_object_order() -> None:
    scene = {
        "atom_objects": [
            {"text": {"value": "O"}},
            {"text": None},
            {"text": {"value": "NH2+"}},
        ]
    }
    assert chemscene_atom_labels(scene) == {
        0: ("O", True),
        1: ("", False),
        2: ("NH2+", True),
    }


def test_bbox_component_gate_keeps_combined_target_separate() -> None:
    result = {
        "state_accuracy": 0.952,
        "state_exact_rate": 0.189,
        "text_raw_accuracy": 0.829,
    }
    assert deployment_gate(result, learned_bbox=True)["passed"]
    assert not deployment_gate(result, learned_bbox=False)["passed"]
