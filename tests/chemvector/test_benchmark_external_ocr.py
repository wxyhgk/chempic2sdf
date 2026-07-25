import unittest

from chemvector.benchmark_external_ocr import has_modifier, label_group


class ExternalOCRBenchmarkTest(unittest.TestCase):
    def test_label_groups(self) -> None:
        self.assertEqual(label_group("N"), "simple_element")
        self.assertEqual(label_group("Cl"), "simple_element")
        self.assertEqual(label_group("NH"), "modifier")
        self.assertEqual(label_group("CH3"), "modifier")
        self.assertEqual(label_group("Ph"), "open_string")

    def test_modifier_detection(self) -> None:
        self.assertTrue(has_modifier("N+"))
        self.assertTrue(has_modifier("CH3"))
        self.assertFalse(has_modifier("N"))


if __name__ == "__main__":
    unittest.main()
