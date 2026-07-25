from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from chemvector.bond_group_proposal_membership import (
    PARALLEL_2,
    SINGLE_LINE,
)
from chemvector.bond_group_proposal_pointer_pipeline import (
    adapt_proposal_pointer_targets,
    compare_latent_root_assignments,
    decode_proposal_pointer_partition,
    latent_root_assignment_map,
    pointer_root_selection_diagnostics,
    prepare_proposal_pointer_pipeline,
    resolve_pointer_group_counts,
)


def _targets() -> dict[str, torch.Tensor]:
    return {
        "group_types": torch.tensor([[PARALLEL_2, SINGLE_LINE, 0]]),
        "group_membership": torch.tensor(
            [
                [
                    [True, True, False, False, False],
                    [False, False, True, False, False],
                    [False, False, False, False, False],
                ]
            ]
        ),
        "group_mask": torch.tensor([[True, True, False]]),
        "member_counts": torch.tensor([[2, 1, 0]]),
        "group_weights": torch.tensor([[0.75, 1.0, 0.0]]),
        "proposal_keep": torch.tensor([[True, True, True, False, False]]),
        "source_group_ids": torch.tensor([[10, 20, -1]]),
        "known_counts": torch.tensor([2]),
    }


def _output(proposals: int = 5, count_classes: int = 7) -> dict[str, torch.Tensor]:
    return {
        "keep_logits": torch.zeros((1, proposals)),
        "group_type_logits": torch.zeros((1, proposals, 7)),
        "member_count_logits": torch.zeros((1, proposals, 9)),
        "pointer_logits": torch.full((1, proposals, proposals), -8.0),
        "group_count_logits": torch.zeros((1, count_classes)),
    }


class BondGroupProposalPointerPipelineTest(unittest.TestCase):
    def test_target_adapter_preserves_exact_oracle_membership_union(self) -> None:
        proposal_mask = torch.tensor([[True, True, True, True, False]])
        targets = _targets()

        adapted = adapt_proposal_pointer_targets(
            targets,
            proposal_mask,
            max_group_count=3,
        )

        self.assertEqual(set(adapted), set(targets))
        self.assertIs(adapted["group_membership"], targets["group_membership"])
        self.assertEqual(adapted["known_counts"].tolist(), [2])

        broken = dict(targets)
        broken["proposal_keep"] = torch.tensor(
            [[True, True, True, True, False]]
        )
        with self.assertRaisesRegex(ValueError, "membership union"):
            adapt_proposal_pointer_targets(
                broken,
                proposal_mask,
                max_group_count=3,
            )

    def test_prepare_reuses_membership_pipeline_and_exposes_usable_mask(self) -> None:
        proposal_mask = torch.tensor([[True, True, True, True, False]])
        prepared = {
            "model_inputs": {
                "proposal_features": torch.zeros(1, 5, 4),
                "proposal_descriptors": torch.zeros(1, 5, 23),
                "flat_type_logits": torch.zeros(1, 5, 5),
                "proposal_scores": torch.zeros(1, 5),
                "proposal_mask": proposal_mask,
            },
            "targets": _targets(),
            "known_counts": torch.tensor([2]),
            "oracle_coverage": {"oracle_keep": torch.tensor([3])},
        }
        batch = {"proposal_query_index": torch.tensor([[3, 1, 4, 2, -1]])}

        with patch(
            "chemvector.bond_group_proposal_pointer_pipeline."
            "prepare_proposal_membership_pipeline",
            return_value=prepared,
        ) as reused:
            result = prepare_proposal_pointer_pipeline(
                batch,
                {"type_logits": torch.zeros(1, 5, 5)},
                torch.zeros(5, 2),
                max_group_count=3,
            )

        self.assertEqual(reused.call_args.kwargs["group_slots"], 3)
        self.assertEqual(
            result["oracle_usable_mask"].tolist(),
            [[True, True, True, False, False]],
        )
        self.assertEqual(
            result["proposal_query_indices"].tolist(),
            [[3, 1, 4, 2, -1]],
        )

    def test_known_count_oracle_mask_excludes_ignored_and_padding(self) -> None:
        output = _output()
        output["keep_logits"][0] = torch.tensor([9.0, -2.0, 8.0, 30.0, 40.0])
        output["group_type_logits"][0, 0, PARALLEL_2] = 10.0
        output["group_type_logits"][0, 2, SINGLE_LINE] = 10.0
        output["member_count_logits"][0, 0, 2] = 10.0
        output["member_count_logits"][0, 2, 1] = 10.0
        pointer = output["pointer_logits"][0]
        pointer[0, 0] = pointer[0, 1] = pointer[1, 0] = 8.0
        pointer[2, 2] = 8.0
        proposal_mask = torch.tensor([[True, True, True, True, False]])
        usable = torch.tensor([[True, True, True, False, False]])

        decoded, resolution = decode_proposal_pointer_partition(
            output,
            proposal_mask,
            root_candidate_mask=usable,
            member_candidate_mask=usable,
            mode="known_count",
            known_counts=torch.tensor([2]),
        )

        self.assertEqual(resolution.feasible_counts.tolist(), [2])
        assigned = [
            proposal
            for row in decoded[0]
            for proposal in row["proposal_indices"]
        ]
        self.assertEqual(sorted(assigned), [0, 1, 2])
        self.assertNotIn(3, assigned)
        self.assertNotIn(4, assigned)

    def test_root_and_member_candidate_masks_have_distinct_semantics(self) -> None:
        output = _output(proposals=4)
        output["keep_logits"][0] = torch.tensor([8.0, 30.0, 7.0, -2.0])
        output["group_type_logits"][0, 0, PARALLEL_2] = 10.0
        output["group_type_logits"][0, 2, PARALLEL_2] = 10.0
        pointer = output["pointer_logits"][0]
        pointer[0, 0] = pointer[0, 1] = pointer[1, 0] = 8.0
        pointer[2, 2] = pointer[2, 3] = pointer[3, 2] = 8.0
        valid = torch.ones((1, 4), dtype=torch.bool)
        roots = torch.tensor([[True, False, True, False]])
        members = torch.ones((1, 4), dtype=torch.bool)

        decoded, _ = decode_proposal_pointer_partition(
            output,
            valid,
            root_candidate_mask=roots,
            member_candidate_mask=members,
            mode="known_count",
            known_counts=torch.tensor([2]),
        )

        self.assertEqual([row["root_index"] for row in decoded[0]], [0, 2])
        assigned = [
            proposal
            for row in decoded[0]
            for proposal in row["proposal_indices"]
        ]
        self.assertEqual(sorted(assigned), [0, 1, 2, 3])
        self.assertEqual(len(assigned), len(set(assigned)))

        with self.assertRaisesRegex(ValueError, "must be member"):
            decode_proposal_pointer_partition(
                output,
                valid,
                root_candidate_mask=torch.tensor(
                    [[True, False, True, True]]
                ),
                member_candidate_mask=torch.tensor(
                    [[True, True, True, False]]
                ),
                mode="known_count",
                known_counts=torch.tensor([2]),
            )

    def test_learned_count_is_selected_only_from_the_feasible_domain(self) -> None:
        output = _output(proposals=4, count_classes=7)
        output["group_count_logits"][0, 5] = 20.0
        output["group_count_logits"][0, 2] = 10.0
        output["keep_logits"][0, :2] = torch.tensor([8.0, 7.0])
        output["group_type_logits"][0, :2, SINGLE_LINE] = 10.0
        output["pointer_logits"][0].fill_(0.0)
        valid = torch.ones((1, 4), dtype=torch.bool)
        roots = torch.tensor([[True, True, False, False]])
        members = torch.tensor([[True, True, False, False]])

        resolution = resolve_pointer_group_counts(
            output,
            roots,
            members,
            mode="learned_count",
        )
        self.assertEqual(resolution.raw_counts.tolist(), [5])
        self.assertEqual(resolution.feasible_counts.tolist(), [2])
        self.assertEqual(resolution.infeasible.tolist(), [True])
        decoded, decoded_resolution = decode_proposal_pointer_partition(
            output,
            valid,
            root_candidate_mask=roots,
            member_candidate_mask=members,
            mode="learned_count",
        )
        self.assertEqual(len(decoded[0]), 2)
        self.assertEqual(decoded_resolution.feasible_counts.tolist(), [2])

    def test_root_diagnostics_expose_same_group_collisions_and_affinity(self) -> None:
        output = _output(proposals=4)
        output["pointer_logits"][0, 0, 1] = 10.0
        output["pointer_logits"][0, 0, 0] = 1.0
        output["pointer_logits"][0, 1, 1] = 10.0
        targets = {
            "group_mask": torch.tensor([[True, True]]),
            "group_membership": torch.tensor(
                [[[True, True, False, False], [False, False, True, True]]]
            ),
        }
        decoded = [
            [
                {"root_index": 0, "proposal_indices": [0, 2]},
                {"root_index": 1, "proposal_indices": [1, 3]},
            ]
        ]
        diagnostics = pointer_root_selection_diagnostics(
            output,
            decoded,
            targets,
            torch.ones((1, 4), dtype=torch.bool),
        )

        self.assertEqual(diagnostics["selected_roots"], 2)
        self.assertEqual(diagnostics["distinct_target_groups"], 1)
        self.assertEqual(diagnostics["root_collisions"], 1)
        self.assertEqual(diagnostics["affinity_self_preferred_roots"], 1)
        self.assertEqual(diagnostics["assigned_members_same_as_root_group"], 2)

    def test_latent_root_stability_uses_flat_query_identity(self) -> None:
        targets = _targets()
        selected = torch.tensor([[1, 2, -1]])
        query_indices = torch.tensor([[8, 3, 5, 7, -1]])
        current = latent_root_assignment_map(
            ["sample-a"],
            targets,
            selected,
            query_indices,
        )
        self.assertEqual(current, {"sample-a::10": 3, "sample-a::20": 5})
        first = compare_latent_root_assignments(None, current)
        self.assertIsNone(first["rate"])
        compared = compare_latent_root_assignments(
            {"sample-a::10": 3, "sample-a::20": 9},
            current,
        )
        self.assertEqual(compared["compared"], 2)
        self.assertEqual(compared["stable"], 1)
        self.assertEqual(compared["rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
