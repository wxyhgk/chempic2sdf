"""Audit legitimate degree-4 junctions that contradict a global crossing prior."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
from math import atan2, degrees
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "crossing-prior-risk-audit-v0.2"
PAIRINGS = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)


@dataclass(frozen=True, slots=True)
class Atom2D:
    x: Fraction
    y: Fraction
    element: str


@dataclass(frozen=True, slots=True)
class Bond2D:
    begin: int
    end: int
    order: int


@dataclass(frozen=True, slots=True)
class Mol2D:
    atoms: tuple[Atom2D, ...]
    bonds: tuple[Bond2D, ...]


def parse_v2000(path: Path) -> Mol2D:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    counts_index = next(
        (index for index, line in enumerate(lines) if "V2000" in line),
        None,
    )
    if counts_index is None:
        raise ValueError(f"{path} does not contain a V2000 counts line")
    counts = lines[counts_index]
    atom_count = int(counts[:3])
    bond_count = int(counts[3:6])
    atom_lines = lines[counts_index + 1 : counts_index + 1 + atom_count]
    bond_lines = lines[
        counts_index + 1 + atom_count : counts_index + 1 + atom_count + bond_count
    ]
    if len(atom_lines) != atom_count or len(bond_lines) != bond_count:
        raise ValueError(f"{path} has a truncated V2000 atom or bond block")
    atoms = tuple(
        Atom2D(
            x=Fraction(line[:10].strip()),
            y=Fraction(line[10:20].strip()),
            element=line[31:34].strip(),
        )
        for line in atom_lines
    )
    bonds = tuple(
        Bond2D(
            begin=int(line[:3]) - 1,
            end=int(line[3:6]) - 1,
            order=int(line[6:9]),
        )
        for line in bond_lines
    )
    if any(
        bond.begin < 0
        or bond.end < 0
        or bond.begin >= atom_count
        or bond.end >= atom_count
        for bond in bonds
    ):
        raise ValueError(f"{path} contains an out-of-range bond endpoint")
    return Mol2D(atoms=atoms, bonds=bonds)


def _cross(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> Fraction:
    return left[0] * right[1] - left[1] * right[0]


def _dot(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> Fraction:
    return left[0] * right[0] + left[1] * right[1]


def strict_opposite(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> bool:
    return _cross(left, right) == 0 and _dot(left, right) < 0


def opposite_deviation_degrees(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> float:
    cross = abs(float(_cross(left, right)))
    dot = float(_dot(left, right))
    return degrees(atan2(cross, -dot))


def matching_deviation_degrees(
    vectors: tuple[tuple[Fraction, Fraction], ...]
) -> float:
    if len(vectors) != 4:
        raise ValueError("A degree-4 matching requires exactly four vectors")
    return min(
        max(
            opposite_deviation_degrees(vectors[left], vectors[right])
            for left, right in pairing
        )
        for pairing in PAIRINGS
    )


def is_strict_cross(vectors: tuple[tuple[Fraction, Fraction], ...]) -> bool:
    if len(vectors) != 4:
        return False
    return any(
        all(strict_opposite(vectors[left], vectors[right]) for left, right in pairing)
        for pairing in PAIRINGS
    )


def degree_four_records(path: Path, molecule: Mol2D) -> Iterable[dict]:
    incident: list[list[Bond2D]] = [[] for _ in molecule.atoms]
    for bond in molecule.bonds:
        incident[bond.begin].append(bond)
        incident[bond.end].append(bond)
    for atom_id, bonds in enumerate(incident):
        if len(bonds) != 4:
            continue
        center = molecule.atoms[atom_id]
        neighbor_ids = tuple(
            bond.end if bond.begin == atom_id else bond.begin for bond in bonds
        )
        vectors = tuple(
            (
                molecule.atoms[neighbor_id].x - center.x,
                molecule.atoms[neighbor_id].y - center.y,
            )
            for neighbor_id in neighbor_ids
        )
        if any(vector == (0, 0) for vector in vectors):
            raise ValueError(f"{path} atom {atom_id} has a zero-length incident bond")
        yield {
            "atom_id_zero_based": atom_id,
            "all_single_bonds": all(bond.order == 1 for bond in bonds),
            "element": center.element,
            "neighbor_elements": [
                molecule.atoms[neighbor_id].element for neighbor_id in neighbor_ids
            ],
            "strict_cross": is_strict_cross(vectors),
            "matching_deviation_degrees": matching_deviation_degrees(vectors),
        }


def audit_directory(root: Path) -> dict:
    paths = sorted(root.glob("*.sdf"))
    parse_failures = []
    degree_four = []
    for path in paths:
        try:
            molecule = parse_v2000(path)
            records = list(degree_four_records(path, molecule))
        except (OSError, UnicodeError, ValueError) as error:
            parse_failures.append({"file": path.name, "error": str(error)})
            continue
        for record in records:
            degree_four.append({"file": path.name, **record})

    def bucket(predicate) -> tuple[int, int]:
        selected = [record for record in degree_four if predicate(record)]
        return len(selected), len({record["file"] for record in selected})

    strict_atom_count, strict_molecule_count = bucket(
        lambda record: record["strict_cross"]
    )
    one_atom_count, one_molecule_count = bucket(
        lambda record: record["matching_deviation_degrees"] <= 1.0
    )
    five_atom_count, five_molecule_count = bucket(
        lambda record: record["matching_deviation_degrees"] <= 5.0
    )
    strict_non_single_or_non_carbon = [
        record
        for record in degree_four
        if record["strict_cross"]
        and (record["element"] != "C" or not record["all_single_bonds"])
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "input_root": str(root.resolve()),
        "sdf_file_count": len(paths),
        "parsed_file_count": len(paths) - len(parse_failures),
        "parse_failures": parse_failures,
        "degree_four_atom_count": len(degree_four),
        "degree_four_molecule_count": len({record["file"] for record in degree_four}),
        "strict_cross": {
            "atom_count": strict_atom_count,
            "molecule_count": strict_molecule_count,
            "records": [record for record in degree_four if record["strict_cross"]],
        },
        "within_1_degree": {
            "atom_count": one_atom_count,
            "molecule_count": one_molecule_count,
        },
        "within_5_degrees": {
            "atom_count": five_atom_count,
            "molecule_count": five_molecule_count,
        },
        "strict_non_single_or_non_carbon": strict_non_single_or_non_carbon,
        "prior_id": "unmarked_transversal_crossing_is_disconnected_v1",
        "global_prior_safe": strict_atom_count == 0,
        "training_authorized": False,
        "scope": "external_prior_risk_diagnostic_only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_directory(args.root)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
