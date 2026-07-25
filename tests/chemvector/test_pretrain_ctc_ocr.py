import unittest

from PIL import Image

from chemvector.pretrain_ctc_ocr import bbox_crop, is_modifier_label


class ChemicalTextCropTest(unittest.TestCase):
    def test_bbox_crop_has_stable_canvas(self) -> None:
        image = Image.new("RGB", (100, 100), "white")
        crop = bbox_crop(image, [40, 40, 60, 60], context=4)
        self.assertEqual(crop.size, (192, 48))

    def test_modifier_labels(self) -> None:
        self.assertTrue(is_modifier_label("NH2+"))
        self.assertTrue(is_modifier_label("CH3"))
        self.assertFalse(is_modifier_label("N"))


if __name__ == "__main__":
    unittest.main()
