from __future__ import annotations

import copy
import unittest

from .symbolic_family import (
    FamilyError,
    SymbolicFamily,
    independent_binary_family,
)


class SymbolicCapacityTests(unittest.TestCase):
    def test_seven_binary_variables_fit_materialized_capacity(self) -> None:
        family = SymbolicFamily.from_document(independent_binary_family(7))
        values, overflow = family.materialize(capacity=128)
        self.assertFalse(overflow)
        self.assertEqual(len(values), 128)

    def test_eight_binary_variables_have_256_solutions(self) -> None:
        family = SymbolicFamily.from_document(independent_binary_family(8))
        self.assertEqual(family.solution_count(), 256)

    def test_eight_binary_variables_overflow_materialized_but_not_symbolic_form(self) -> None:
        document = independent_binary_family(8)
        family = SymbolicFamily.from_document(document)
        values, overflow = family.materialize(capacity=128)
        self.assertTrue(overflow)
        self.assertEqual(values, ())
        self.assertEqual(len(document["variables"]), 8)
        self.assertEqual(document["constraints"], [])

    def test_table_constraint_reduces_solution_set_exactly(self) -> None:
        document = independent_binary_family(2)
        document["constraints"] = [{
            "allowed_tuples": [[0, 1], [1, 0]],
            "id": 0,
            "scope": [0, 1],
        }]
        family = SymbolicFamily.from_document(document)
        self.assertEqual(tuple(family.solutions()), ((0, 1), (1, 0)))

    def test_non_integer_assignment_values_return_false(self) -> None:
        family = SymbolicFamily.from_document(independent_binary_family(1))
        self.assertFalse(family.is_solution((False,)))
        self.assertFalse(family.is_solution((0.0,)))
        self.assertFalse(family.is_solution(("0",)))


class ClosedDocumentTests(unittest.TestCase):
    def test_extra_root_field_rejects(self) -> None:
        document = independent_binary_family(1)
        document["winner"] = 0
        with self.assertRaises(FamilyError):
            SymbolicFamily.from_document(document)

    def test_training_authorization_rejects(self) -> None:
        document = independent_binary_family(1)
        document["training_authorized"] = True
        with self.assertRaises(FamilyError):
            SymbolicFamily.from_document(document)

    def test_noncontiguous_variable_ids_reject(self) -> None:
        document = independent_binary_family(2)
        document["variables"][1]["id"] = 2
        with self.assertRaises(FamilyError):
            SymbolicFamily.from_document(document)

    def test_extra_variable_field_rejects(self) -> None:
        document = independent_binary_family(1)
        document["variables"][0]["score"] = 1
        with self.assertRaises(FamilyError):
            SymbolicFamily.from_document(document)

    def test_boolean_cannot_impersonate_integer_id(self) -> None:
        document = independent_binary_family(1)
        document["variables"][0]["id"] = False
        with self.assertRaises(FamilyError):
            SymbolicFamily.from_document(document)

    def test_duplicate_allowed_tuple_rejects(self) -> None:
        document = independent_binary_family(2)
        document["constraints"] = [{
            "allowed_tuples": [[0, 1], [0, 1]],
            "id": 0,
            "scope": [0, 1],
        }]
        with self.assertRaises(FamilyError):
            SymbolicFamily.from_document(document)

    def test_out_of_domain_tuple_rejects(self) -> None:
        document = independent_binary_family(2)
        document["constraints"] = [{
            "allowed_tuples": [[0, 2]],
            "id": 0,
            "scope": [0, 1],
        }]
        with self.assertRaises(FamilyError):
            SymbolicFamily.from_document(document)

    def test_unsorted_scope_rejects(self) -> None:
        document = independent_binary_family(2)
        document["constraints"] = [{
            "allowed_tuples": [[0, 1]],
            "id": 0,
            "scope": [1, 0],
        }]
        with self.assertRaises(FamilyError):
            SymbolicFamily.from_document(document)


if __name__ == "__main__":
    unittest.main()
