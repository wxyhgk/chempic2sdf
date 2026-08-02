import unittest

from chemvector.evaluation.audits.atoms.audit_g3_p1k_final import _assert_equal


class G3P1KFinalAuditTests(unittest.TestCase):
    def test_exact_tree_passes(self) -> None:
        _assert_equal({"a": [1, 2.0]}, {"a": [1, 2.0]})

    def test_difference_fails(self) -> None:
        with self.assertRaisesRegex(AssertionError, "root.a"):
            _assert_equal({"a": 1.0}, {"a": 1.1})


if __name__ == "__main__":
    unittest.main()
