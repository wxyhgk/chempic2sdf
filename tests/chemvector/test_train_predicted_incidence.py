import unittest

from chemvector.train_predicted_incidence import _sample_split


class TrainPredictedIncidenceTests(unittest.TestCase):
    def test_sample_split_accepts_both_dataset_metadata_conventions(self) -> None:
        self.assertEqual(
            _sample_split({"meta": {"source_split": "train"}}), "train"
        )
        self.assertEqual(
            _sample_split({"meta": {"split": "validation"}}), "validation"
        )
        self.assertIsNone(_sample_split({"meta": {}}))


if __name__ == "__main__":
    unittest.main()
