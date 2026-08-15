#!/usr/bin/env python3
"""A small closed finite-domain CSP used to test symbolic candidate capacity."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any, Iterator


class FamilyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Variable:
    id: int
    domain_size: int


@dataclass(frozen=True, slots=True)
class TableConstraint:
    id: int
    scope: tuple[int, ...]
    allowed_tuples: frozenset[tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class SymbolicFamily:
    variables: tuple[Variable, ...]
    constraints: tuple[TableConstraint, ...]

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "SymbolicFamily":
        if not isinstance(document, dict):
            raise FamilyError("document must be an object")
        if set(document) != {"constraints", "schema_version", "training_authorized", "variables"}:
            raise FamilyError("document fields are not closed")
        if document["schema_version"] != "symbolic-candidate-family-probe-v0.1":
            raise FamilyError("unknown schema version")
        if document["training_authorized"] is not False:
            raise FamilyError("training must remain unauthorized")
        if not isinstance(document["variables"], list) or not isinstance(document["constraints"], list):
            raise FamilyError("variables and constraints must be arrays")
        parsed_variables: list[Variable] = []
        for item in document["variables"]:
            if not isinstance(item, dict) or set(item) != {"domain_size", "id"}:
                raise FamilyError("variable fields are not closed")
            if type(item["id"]) is not int or type(item["domain_size"]) is not int:
                raise FamilyError("variable id and domain_size must be integers")
            parsed_variables.append(Variable(item["id"], item["domain_size"]))
        variables = tuple(parsed_variables)
        if tuple(variable.id for variable in variables) != tuple(range(len(variables))):
            raise FamilyError("variable IDs must be contiguous")
        if any(not 1 <= variable.domain_size <= 16 for variable in variables):
            raise FamilyError("domain_size must be between 1 and 16")
        domain_sizes = {variable.id: variable.domain_size for variable in variables}
        constraints: list[TableConstraint] = []
        for expected_id, item in enumerate(document["constraints"]):
            if not isinstance(item, dict) or set(item) != {"allowed_tuples", "id", "scope"}:
                raise FamilyError("constraint fields are not closed")
            if type(item["id"]) is not int:
                raise FamilyError("constraint id must be an integer")
            if item["id"] != expected_id:
                raise FamilyError("constraint IDs must be contiguous")
            if not isinstance(item["scope"], list) or any(type(value) is not int for value in item["scope"]):
                raise FamilyError("constraint scope must be an integer array")
            scope = tuple(item["scope"])
            if scope != tuple(sorted(set(scope))) or any(value not in domain_sizes for value in scope):
                raise FamilyError("constraint scope must be sorted, unique and known")
            if not isinstance(item["allowed_tuples"], list) or any(
                not isinstance(row, list) or any(type(value) is not int for value in row)
                for row in item["allowed_tuples"]
            ):
                raise FamilyError("allowed_tuples must be arrays of integer arrays")
            allowed_rows = [tuple(row) for row in item["allowed_tuples"]]
            if allowed_rows != sorted(set(allowed_rows)):
                raise FamilyError("allowed tuples must be sorted and byte-unique")
            if any(
                len(row) != len(scope)
                or any(not 0 <= value < domain_sizes[variable_id] for value, variable_id in zip(row, scope))
                for row in allowed_rows
            ):
                raise FamilyError("allowed tuple lies outside its scoped domains")
            constraints.append(TableConstraint(expected_id, scope, frozenset(allowed_rows)))
        return cls(variables, tuple(constraints))

    def is_solution(self, assignment: tuple[int, ...]) -> bool:
        if len(assignment) != len(self.variables):
            return False
        if any(type(value) is not int for value in assignment):
            return False
        if any(not 0 <= assignment[item.id] < item.domain_size for item in self.variables):
            return False
        return all(
            tuple(assignment[variable_id] for variable_id in constraint.scope)
            in constraint.allowed_tuples
            for constraint in self.constraints
        )

    def solutions(self) -> Iterator[tuple[int, ...]]:
        domains = (range(variable.domain_size) for variable in self.variables)
        for assignment in itertools.product(*domains):
            if self.is_solution(assignment):
                yield assignment

    def solution_count(self) -> int:
        return sum(1 for _ in self.solutions())

    def materialize(self, *, capacity: int = 128) -> tuple[tuple[tuple[int, ...], ...], bool]:
        values: list[tuple[int, ...]] = []
        for assignment in self.solutions():
            values.append(assignment)
            if len(values) > capacity:
                return (), True
        return tuple(values), False


def independent_binary_family(variable_count: int) -> dict[str, Any]:
    return {
        "constraints": [],
        "schema_version": "symbolic-candidate-family-probe-v0.1",
        "training_authorized": False,
        "variables": [
            {"domain_size": 2, "id": index}
            for index in range(variable_count)
        ],
    }
