import unittest

from chemvector.audit_atom_spacing_distribution import distance_bucket


class AtomSpacingDistributionTests(unittest.TestCase):
    def test_distance_buckets_cover_close_vertex_cases(self) -> None:
        self.assertEqual(distance_bucket(7.9), "lt_8")
        self.assertEqual(distance_bucket(13.7), "12_16")
        self.assertEqual(distance_bucket(48.0), "ge_48")


if __name__ == "__main__":
    unittest.main()
