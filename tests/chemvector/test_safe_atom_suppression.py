import torch

from chemvector.safe_atom_suppression import (
    label_single_delete_actions,
    lexicographic_node_match,
    strict_winner_removed_nodes,
    unique_delete_nodes,
)


def _sample(
    anchors,
    target_anchors,
    pairs,
    probabilities,
    target_bonds,
    chemical=None,
):
    return {
        "anchors": torch.tensor(anchors, dtype=torch.float64) / 768.0,
        "scores": torch.linspace(0.9, 0.5, len(anchors)),
        "pairs": torch.tensor(pairs, dtype=torch.long),
        "probabilities": torch.tensor(probabilities),
        "chemical": torch.tensor(chemical or [1] * len(pairs)),
        "target_anchors": torch.tensor(target_anchors, dtype=torch.float64) / 768.0,
        "target_bonds": target_bonds,
    }


def test_lexicographic_match_maximizes_in_radius_cardinality() -> None:
    anchors = torch.tensor([[0.0, 0.0], [5.5, 0.0]]) / 768.0
    targets = torch.tensor([[5.0, 0.0], [10.5, 0.0]]) / 768.0
    mapping = lexicographic_node_match(
        anchors,
        targets,
        torch.ones(2, dtype=torch.bool),
        radius_px=6.0,
    )
    assert mapping == {0: 0, 1: 1}


def test_delete_actions_are_unique_per_removed_node() -> None:
    anchors = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]) / 768.0
    assert unique_delete_nodes(anchors, maximum_distance_px=16.0) == [0, 1, 2]


def test_unmatched_redundant_node_is_safe_rescue() -> None:
    sample = _sample(
        anchors=[[0.0, 0.0], [7.0, 0.0], [100.0, 0.0]],
        target_anchors=[[0.0, 0.0], [100.0, 0.0]],
        pairs=[[0, 2], [1, 2], [0, 1]],
        probabilities=[0.99, 0.99, 0.01],
        target_bonds={(0, 1): 1},
    )
    labels = label_single_delete_actions(sample, presence_threshold=0.5)
    by_removed = {action["removed"]: action for action in labels["actions"]}
    assert by_removed[1]["safe"]
    assert by_removed[1]["strict_gain"]
    assert labels["winner_removed_nodes"] == [1]


def test_distinct_close_atoms_cannot_be_deleted() -> None:
    sample = _sample(
        anchors=[[0.0, 0.0], [5.0, 0.0]],
        target_anchors=[[0.0, 0.0], [5.0, 0.0]],
        pairs=[[0, 1]],
        probabilities=[0.01],
        target_bonds={},
    )
    labels = label_single_delete_actions(sample, presence_threshold=0.5)
    assert not any(action["safe"] for action in labels["actions"])
    assert labels["winner_removed_nodes"] == []


def test_unique_bond_owner_selects_the_other_duplicate() -> None:
    sample = _sample(
        anchors=[[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]],
        target_anchors=[[0.0, 0.0], [100.0, 0.0]],
        pairs=[[0, 2], [1, 2], [0, 1]],
        probabilities=[0.99, 0.01, 0.01],
        target_bonds={(0, 1): 1},
    )
    labels = label_single_delete_actions(sample, presence_threshold=0.5)
    by_removed = {action["removed"]: action for action in labels["actions"]}
    assert not by_removed[0]["safe"]
    assert by_removed[1]["safe"]
    assert labels["winner_removed_nodes"] == [1]


def test_safe_neutral_does_not_beat_keep() -> None:
    baseline = (0, 0, -2, -3)
    actions = [
        {"removed": 1, "safe": True, "utility": baseline},
        {"removed": 2, "safe": False, "utility": (1, 1, 0, 0)},
    ]
    assert strict_winner_removed_nodes(baseline, actions) == []
