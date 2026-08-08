import pytest
import torch

from chemvector.evaluation.audits.atoms.build_g3_p4e9_recovery_oracle import (
    ACTION_ID_STRIDE,
    allocate_scaffold_splits,
    deficient_components,
    recovery_actions,
)
from chemvector.evaluation.audits.atoms.audit_g3_p4e8_local_multiplicity_components import (
    deficient_bipartite_components,
)


def test_scaffold_split_is_deterministic_and_group_disjoint() -> None:
    rows = []
    for index in range(30):
        for variant in ("coordgen", "original", "rdkit", "stress"):
            rows.append(
                {
                    "sample_key": f"sample-{index}-{variant}",
                    "molecule_id": f"mol-{index}",
                    "inchikey": f"key-{index}",
                    "scaffold": f"scaffold-{index // 2}",
                    "variant": variant,
                }
            )
    first, first_audit = allocate_scaffold_splits(rows)
    second, second_audit = allocate_scaffold_splits(list(reversed(rows)))
    assert first == second
    assert first_audit == second_audit
    for left in range(0, 30, 2):
        assert first[f"mol-{left}"] == first[f"mol-{left + 1}"]
    assert not any(first_audit["cross_split_scaffold_overlap"].values())


def test_recovery_oracle_separates_split_and_refine_actions() -> None:
    # Candidate 0 is shared by GT 0/1; candidate 1 is 10px from GT 2.
    predicted = torch.tensor([[100.0, 100.0], [210.0, 200.0], [500.0, 500.0]]) / 768.0
    expected = torch.tensor([[96.0, 100.0], [104.0, 100.0], [200.0, 200.0]]) / 768.0
    source_ids = torch.tensor([7, 11, 19])
    actions = recovery_actions(predicted, source_ids, expected)
    assert [item["route"] for item in actions] == ["ADD_REFINE", "ADD_SPLIT"]
    by_route = {item["route"]: item for item in actions}
    assert by_route["ADD_SPLIT"]["base_source_id"] == 7
    assert by_route["ADD_SPLIT"]["added_source_id"] == ACTION_ID_STRIDE + 7
    assert by_route["ADD_REFINE"]["base_source_id"] == 11
    assert by_route["ADD_REFINE"]["added_source_id"] == 2 * ACTION_ID_STRIDE + 11
    assert by_route["ADD_REFINE"]["source_distance_pixels"] == pytest.approx(10.0)
    assert all(item["oracle_matched_gain"] == 1 for item in actions)


def test_refine_requires_strictly_unreachable_gt_within_twelve_pixels() -> None:
    expected = torch.tensor([[100.0, 100.0]]) / 768.0
    source_ids = torch.tensor([1, 2])
    too_far = torch.tensor([[87.0, 100.0], [500.0, 500.0]]) / 768.0
    assert recovery_actions(too_far, source_ids, expected) == []
    already_reachable = torch.tensor([[95.0, 100.0], [500.0, 500.0]]) / 768.0
    assert recovery_actions(already_reachable, source_ids, expected) == []


def test_split_source_is_not_reused_by_refine_route() -> None:
    predicted = torch.tensor([[100.0, 100.0], [500.0, 500.0]]) / 768.0
    expected = torch.tensor(
        [[96.0, 100.0], [104.0, 100.0], [100.0, 111.0]]
    ) / 768.0
    actions = recovery_actions(predicted, torch.tensor([7, 9]), expected)
    assert len(actions) == 1
    assert actions[0]["route"] == "ADD_SPLIT"
    assert actions[0]["base_source_id"] == 7


def test_ambiguous_shared_component_is_explicitly_unsupported() -> None:
    predicted = torch.tensor([[100.0, 100.0], [112.0, 100.0]]) / 768.0
    expected = torch.tensor(
        [[96.0, 100.0], [106.0, 100.0], [116.0, 100.0]]
    ) / 768.0
    actions, unsupported = recovery_actions(
        predicted,
        torch.tensor([7, 9]),
        expected,
        return_details=True,
    )
    assert actions == []
    assert unsupported == [7, 9]


def test_fast_component_matcher_replays_p4e8_cardinality() -> None:
    generator = torch.Generator().manual_seed(8)
    for _ in range(20):
        predicted = torch.rand(16, 2, generator=generator)
        expected = torch.rand(12, 2, generator=generator)
        fast = deficient_components(predicted, expected)
        reference = deficient_bipartite_components(
            predicted, expected, radius_pixels=6.0
        )
        project = lambda rows: [
            (
                row["gt_indices"],
                row["candidate_indices"],
                row["matched_count"],
                row["deficiency"],
            )
            for row in rows
        ]
        assert project(fast) == project(reference)
