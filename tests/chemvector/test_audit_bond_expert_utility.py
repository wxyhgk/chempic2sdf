import unittest

from chemvector.audit_bond_expert_utility import compare_records


def record(topology: bool, chemical: bool, sdf: bool) -> dict:
    return {
        "predicted": {
            "topology_exact": topology,
            "chemical_exact": chemical,
        },
        "compiled_sdf_exact": sdf,
    }


class BondExpertUtilityTest(unittest.TestCase):
    def test_reports_rescue_break_and_oracle_union(self) -> None:
        base = {
            "a": record(True, False, False),
            "b": record(False, False, False),
        }
        expert = {
            "a": record(False, False, False),
            "b": record(True, True, True),
        }

        summary, labels = compare_records(base, expert)

        topology = summary["metrics"]["topology_exact"]
        self.assertEqual(topology["base_exact"], 1)
        self.assertEqual(topology["expert_exact"], 1)
        self.assertEqual(topology["oracle_union_exact"], 2)
        self.assertEqual(topology["expert_rescue"], 1)
        self.assertEqual(topology["expert_break"], 1)
        self.assertEqual(labels[0]["topology_exact_utility"], "expert_break")
        self.assertEqual(labels[1]["topology_exact_utility"], "expert_rescue")

    def test_rejects_mismatched_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "keys differ"):
            compare_records({"a": record(False, False, False)}, {})


if __name__ == "__main__":
    unittest.main()
