"""Protocol tests for P1-X selector capacity training."""

from __future__ import annotations

import unittest

import torch

from chemvector.training.atoms.train_g3_p1x_multiscale_peak_selector_capacity import (
    OUTPUT,
    build_parser,
    cached_epoch_batches,
    capacity_gate,
    collate_cached,
    validate_args,
)


def record(key: str) -> dict:
    return {
        "sample_key": key,
        "candidate_ids": torch.arange(128),
        "candidate_features": torch.zeros(128, 32),
        "candidate_anchors": torch.zeros(128, 2),
        "raw_scores": torch.zeros(128),
        "normalized_scores": torch.zeros(128),
        "scale_ids": torch.tensor([0] * 96 + [1] * 32),
        "scale_ranks": torch.zeros(128),
        "atom_anchors": torch.zeros(128, 2),
        "atom_mask": torch.tensor([True] + [False] * 127),
    }


class G3P1XTrainingTests(unittest.TestCase):
    def test_cli_protocol_is_fixed(self) -> None:
        args = build_parser().parse_args([])
        validate_args(args)
        self.assertEqual(args.output_dir, OUTPUT)
        with self.assertRaisesRegex(ValueError, "fixed batch_size"):
            validate_args(build_parser().parse_args(["--batch-size", "2"]))

    def test_epoch_batches_alternate_two_domains(self) -> None:
        cache = {
            "overfit32": [record(f"o-{index}") for index in range(32)],
            "adjacent32": [record(f"a-{index}") for index in range(32)],
        }
        batches = list(cached_epoch_batches(cache))
        self.assertEqual(len(batches), 16)
        self.assertEqual([name for name, _ in batches], ["overfit32", "adjacent32"] * 8)
        batch = collate_cached(batches[0][1], torch.device("cpu"))
        self.assertEqual(batch["candidate_features"].shape, (4, 128, 32))

    def test_capacity_gate_requires_both_domains_exact(self) -> None:
        complete = {
            "matched_atoms": 100,
            "target_atoms": 100,
            "whole_set_exact_images": 32,
        }
        decision, checks = capacity_gate(
            {"overfit32": complete, "adjacent32": complete}
        )
        self.assertEqual(decision, "dual_domain_selector_capacity_go")
        self.assertTrue(all(checks.values()))
        failed = dict(complete)
        failed["whole_set_exact_images"] = 31
        decision, _ = capacity_gate(
            {"overfit32": complete, "adjacent32": failed}
        )
        self.assertEqual(decision, "dual_domain_selector_capacity_no_go")


if __name__ == "__main__":
    unittest.main()
