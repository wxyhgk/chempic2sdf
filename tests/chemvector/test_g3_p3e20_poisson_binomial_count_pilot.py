"""Protocol tests for the P3-E20 additive Count pilot."""

import unittest

from chemvector.training.atoms.train_g3_p3e20_poisson_binomial_count_pilot import (
    BUCKETS,
    DEV_IMAGES,
    DEV_MOLECULES_PER_BUCKET,
    EPOCHS,
    EXPOSURE,
    STEPS,
    TRAIN_IMAGES,
    TRAIN_MOLECULES_PER_BUCKET,
    balanced_molecule_splits,
    capability_gate,
    training_batches,
)


def records_per_bucket(molecules: int) -> list[dict]:
    representatives = {
        "1-20": 10,
        "21-40": 30,
        "41-60": 50,
        "61-80": 70,
        "81-100": 90,
    }
    records = []
    variants = ("coordgen", "original", "rdkit", "stress")
    for bucket in BUCKETS:
        for molecule in range(molecules):
            for variant in variants:
                records.append({
                    "sample_key": f"{bucket}-{molecule}_{variant}",
                    "label": representatives[bucket],
                })
    return records


class G3P3E20PoissonBinomialPilotTests(unittest.TestCase):
    def test_balanced_split_is_molecule_disjoint(self) -> None:
        required = TRAIN_MOLECULES_PER_BUCKET + DEV_MOLECULES_PER_BUCKET
        splits, audit = balanced_molecule_splits(records_per_bucket(required))
        self.assertEqual(len(splits["train"]), TRAIN_IMAGES)
        self.assertEqual(len(splits["dev"]), DEV_IMAGES)
        self.assertEqual(audit["molecule_overlap"], 0)
        self.assertEqual(set(audit["train_bucket_images"]), set(BUCKETS))

    def test_training_schedule_repeats_only_across_epochs(self) -> None:
        batches, audit = training_batches(TRAIN_IMAGES)
        self.assertEqual(len(batches), STEPS)
        self.assertEqual(audit["epochs"], EPOCHS)
        self.assertEqual(audit["exposure"], EXPOSURE)

    def test_gate_requires_high_count_gain_and_low_count_protection(self) -> None:
        def metrics(value):
            return {
                "count_accuracy": value,
                "count_mae": 1.0,
                "by_count": {
                    bucket: {"count_accuracy": value} for bucket in BUCKETS
                },
            }

        baseline = metrics(0.20)
        dev = metrics(0.30)
        dev["count_mae"] = 0.70
        train = metrics(0.60)
        decision, checks = capability_gate(train, dev, baseline)
        self.assertEqual(decision, "poisson_binomial_count_pilot_go")
        self.assertTrue(all(checks.values()))
        dev["by_count"]["81-100"]["count_accuracy"] = 0.24
        decision, checks = capability_gate(train, dev, baseline)
        self.assertEqual(decision, "poisson_binomial_count_pilot_no_go")
        self.assertFalse(checks["dev_81_100_gain5pp"])


if __name__ == "__main__":
    unittest.main()
