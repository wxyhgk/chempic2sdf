import unittest
from pathlib import Path

from PIL import Image

from chemvector.pretrain_open_text_ctc import (
    OpenTextEntry,
    balance_key,
    edit_distance,
    gate_results,
    letterbox_crop,
    load_initial_model,
    metric_score,
    source_bucket,
    training_weights,
)
from chemvector.audit_open_text_ctc_capacity import balanced_subset_indices


class OpenTextCTCTest(unittest.TestCase):
    def test_letterbox_preserves_aspect_ratio(self) -> None:
        image = Image.new("RGB", (100, 100), "black")
        output = letterbox_crop(image)
        self.assertEqual(output.size, (192, 48))
        self.assertEqual(output.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(output.getpixel((96, 24)), (0, 0, 0))

    def test_source_buckets_keep_real_predictions_separate(self) -> None:
        entry = OpenTextEntry(
            key="1",
            shard=Path("unused.tar"),
            text="NH2",
            split="train",
            source_kind="molecule_bbox",
            bbox_protocol="real_predicted",
            string_group="nitrogen_modifier",
            color_mode="source_black_white",
            font_family_id="font",
        )
        self.assertEqual(source_bucket(entry), "molecule_real")
        self.assertEqual(balance_key(entry), "NH2")

    def test_random_strings_balance_by_structure_not_exact_text(self) -> None:
        common = dict(
            shard=Path("unused.tar"),
            split="train",
            source_kind="synthetic_compositional",
            bbox_protocol="gt",
            string_group="random_composition",
            color_mode="black",
            font_family_id="font",
        )
        left = OpenTextEntry(key="1", text="Ab1", **common)
        right = OpenTextEntry(key="2", text="Zx9", **common)
        self.assertEqual(balance_key(left), balance_key(right))

    def test_training_weights_reproduce_requested_bucket_mass(self) -> None:
        entries = []
        protocols = (
            ("synthetic_compositional", "gt", "random_composition"),
            ("synthetic_compositional", "gt", "element_symbol"),
            ("molecule_bbox", "gt", "element_symbol"),
            ("molecule_bbox", "sampled_predicted", "element_symbol"),
            ("molecule_bbox", "real_predicted", "element_symbol"),
        )
        for index, (source, protocol, string_group) in enumerate(protocols):
            entries.append(
                OpenTextEntry(
                    key=str(index),
                    shard=Path("unused.tar"),
                    text="N",
                    split="train",
                    source_kind=source,
                    bbox_protocol=protocol,
                    string_group=string_group,
                    color_mode="black",
                    font_family_id="font",
                )
            )
        _, audit = training_weights(entries)
        self.assertEqual(
            audit["observed_probability"],
            {
                "molecule_gt": 0.1,
                "molecule_real": 0.3,
                "molecule_sampled": 0.1,
                "synthetic_curated": 0.2,
                "synthetic_random": 0.3,
            },
        )

    def test_edit_distance(self) -> None:
        self.assertEqual(edit_distance("NH2", "NH"), 1)
        self.assertEqual(edit_distance("B", "Br"), 1)

    def test_selection_score_penalizes_false_modifiers(self) -> None:
        def metrics(false_modifier_rate: float) -> dict:
            return {
                "curated_macro_exact_rate": 0.5,
                "segments": {
                    "custom": {
                        "real_predicted": {"exact_rate": 0.5},
                        "molecule_modifier": {"exact_rate": 0.5},
                        "molecule_common_single": {"exact_rate": 0.5},
                        "molecule_simple": {
                            "false_modifier_rate": false_modifier_rate
                        },
                        "open_random": {"edit_similarity": 0.5},
                    }
                },
            }

        self.assertGreater(metric_score(metrics(0.0)), metric_score(metrics(0.5)))

    def test_gates_compare_matching_internal_buckets(self) -> None:
        custom = {
            "real_predicted": {"exact_rate": 0.85},
            "molecule_all": {"exact_rate": 0.80},
            "molecule_modifier": {"exact_rate": 0.75},
            "molecule_common_single": {"exact_rate": 0.96},
            "molecule_simple": {"false_modifier_rate": 0.01},
        }
        validation = {
            "exact_rate": 0.90,
            "supported_macro_exact_rate": 0.80,
            "segments": {"custom": custom},
        }
        internal = {
            "exact_rate": 0.20,
            "supported_macro_exact_rate": 0.70,
            "segments": {
                "custom": {
                    **custom,
                    "real_predicted": {"exact_rate": 0.78},
                    "molecule_all": {"exact_rate": 0.73},
                }
            },
        }
        self.assertEqual(gate_results(validation, internal)["status"], "passed")

    def test_capacity_subset_is_bucket_balanced(self) -> None:
        class Dataset:
            entries = []

        dataset = Dataset()
        for index, (source, protocol, string_group) in enumerate(
            (
                ("synthetic_compositional", "gt", "random_composition"),
                ("synthetic_compositional", "gt", "element_symbol"),
                ("molecule_bbox", "gt", "element_symbol"),
                ("molecule_bbox", "sampled_predicted", "element_symbol"),
                ("molecule_bbox", "real_predicted", "element_symbol"),
            )
        ):
            for repeat in range(2):
                dataset.entries.append(
                    OpenTextEntry(
                        key=f"{index}-{repeat}",
                        shard=Path("unused.tar"),
                        text="N",
                        split="train",
                        source_kind=source,
                        bbox_protocol=protocol,
                        string_group=string_group,
                        color_mode="black",
                        font_family_id="font",
                    )
                )
        indices, counts = balanced_subset_indices(dataset, samples=10, seed=1)
        self.assertEqual(len(indices), 10)
        self.assertEqual(set(counts.values()), {2})

    def test_partial_arm_only_unfreezes_last_two_convolutions(self) -> None:
        checkpoint = Path(
            "outputs/checkpoints/chemvector-v088-pretrained-bbox-ctc-10k/best.pt"
        )
        if not checkpoint.is_file():
            self.skipTest("v0.8.8 checkpoint unavailable")
        model, _ = load_initial_model(checkpoint, arm="partial")
        self.assertFalse(model.pixel_encoder[0].weight.requires_grad)
        self.assertTrue(model.pixel_encoder[2].weight.requires_grad)
        self.assertTrue(model.pixel_encoder[4].weight.requires_grad)

    def test_full_arm_unfreezes_pixel_encoder(self) -> None:
        checkpoint = Path(
            "outputs/checkpoints/chemvector-v088-pretrained-bbox-ctc-10k/best.pt"
        )
        if not checkpoint.is_file():
            self.skipTest("v0.8.8 checkpoint unavailable")
        model, _ = load_initial_model(checkpoint, arm="full")
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.pixel_encoder.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
