import unittest

import torch

from chemvector.pretrain_counterfactual_ctc import (
    PairAwareBatchSampler,
    character_presence_loss,
    character_presence_probabilities,
    character_presence_targets,
    counterfactual_pair_loss,
)
from chemvector.pretrain_open_text_ctc import OpenTextEntry


class CounterfactualCTCTest(unittest.TestCase):
    def test_presence_targets_encode_suffix_characters(self) -> None:
        targets = character_presence_targets(["N", "NH2+"])
        self.assertLess(int(targets[0].sum()), int(targets[1].sum()))

    def test_presence_probabilities_find_confident_character(self) -> None:
        logits = torch.full((1, 4, 257), -10.0)
        logits[..., 0] = 5.0
        logits[0, 1, ord("H") + 1] = 15.0
        probabilities = character_presence_probabilities(logits)
        targets = character_presence_targets(["H"])
        index = int(torch.where(targets[0] > 0)[0][0])
        self.assertGreater(float(probabilities[0, index]), 0.99)

    def test_presence_loss_penalizes_missing_positive(self) -> None:
        targets = character_presence_targets(["H"])
        good = targets * 0.98 + (1.0 - targets) * 0.02
        bad = 1.0 - good
        self.assertLess(
            float(character_presence_loss(good, targets, positive_weight=4.0)),
            float(character_presence_loss(bad, targets, positive_weight=4.0)),
        )

    def test_pair_loss_rewards_same_style_suffix_separation(self) -> None:
        targets = character_presence_targets(["N", "NH"])
        good = targets * 0.9 + (1.0 - targets) * 0.1
        bad = torch.full_like(good, 0.5)
        pair_ids = ["pair", "pair"]
        self.assertLess(
            float(counterfactual_pair_loss(good, targets, pair_ids, margin=0.25)),
            float(counterfactual_pair_loss(bad, targets, pair_ids, margin=0.25)),
        )

    def test_pair_sampler_keeps_complete_pair_in_batch(self) -> None:
        class Dataset:
            entries = [
                OpenTextEntry(
                    key=str(index),
                    shard=None,
                    text=text,
                    split="train",
                    source_kind="synthetic_compositional",
                    bbox_protocol="gt",
                    string_group="nitrogen_modifier",
                    color_mode="black",
                    font_family_id="font",
                    counterfactual_pair_id="pair" if index < 2 else None,
                    counterfactual_pair_group=(
                        "nitrogen_suffix" if index < 2 else None
                    ),
                )
                for index, text in enumerate(("N", "NH", "O", "F"))
            ]

        sampler = PairAwareBatchSampler(
            Dataset(),
            torch.ones(4, dtype=torch.double),
            batch_size=4,
            samples=4,
            pair_fraction=0.5,
            seed=1,
        )
        batch = next(iter(sampler))
        self.assertIn(0, batch)
        self.assertIn(1, batch)


if __name__ == "__main__":
    unittest.main()
