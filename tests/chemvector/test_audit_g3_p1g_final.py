import unittest

from chemvector.evaluation.audits.atoms.audit_g3_p1g_final import _assert_equal


class G3P1GFinalAuditTests(unittest.TestCase):
    def test_recursive_metric_comparison_accepts_exact_tree(self) -> None:
        value = {"a": [1, 2.0, {"b": True}]}
        _assert_equal(value, {"a": [1, 2.0, {"b": True}]})

    def test_recursive_metric_comparison_rejects_difference(self) -> None:
        with self.assertRaisesRegex(AssertionError, "root.a"):
            _assert_equal({"a": 1.0}, {"a": 1.1})


if __name__ == "__main__":
    unittest.main()
