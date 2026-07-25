import unittest

from PIL import Image

from chemvector.pretrain_oracle_mask_ctc import (
    add_macro_metrics,
    balanced_label_repeats,
    crop_layer,
)


class OracleMaskCTCProbeTest(unittest.TestCase):
    def test_balancing_repeats_rare_labels_more(self) -> None:
        repeats = balanced_label_repeats({"O": 100, "Se": 4}, maximum_repeat=8)
        self.assertEqual(repeats["O"], 1)
        self.assertGreater(repeats["Se"], repeats["O"])

    def test_layer_crop_uses_stable_canvas(self) -> None:
        image = Image.new("L", (100, 100), 0)
        crop = crop_layer(
            image,
            [40, 40, 60, 60],
            context=4,
            jitter_x=0,
            jitter_y=0,
        )
        self.assertEqual(crop.size, (192, 48))

    def test_macro_metrics_do_not_follow_micro_frequency(self) -> None:
        metrics = add_macro_metrics(
            {
                "labels": {
                    "O": {"exact_rate": 1.0, "samples": 1000},
                    "Se": {"exact_rate": 0.0, "samples": 20},
                }
            }
        )
        self.assertEqual(metrics["macro_exact_rate"], 0.5)
        self.assertEqual(metrics["supported_macro_exact_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
