#!/usr/bin/env python3
"""ChemTraverse v0.1 deterministic semantic graph traversal protocol."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from typing import Any

from rdkit import Chem

try:
    from .chemmol_sequence import ChemMolSequenceError
    from .chemscene_v0_1 import (
        BondObjectRecord,
        JunctionRecord,
        ParsedChemSceneV01,
        TextObjectRecord,
        mol_from_chemscene_v0_1,
        parse_chemscene_v0_1,
        sdf_from_chemscene_v0_1,
    )
except ImportError:
    from chemmol_sequence import ChemMolSequenceError
    from chemscene_v0_1 import (
        BondObjectRecord,
        JunctionRecord,
        ParsedChemSceneV01,
        TextObjectRecord,
        mol_from_chemscene_v0_1,
        parse_chemscene_v0_1,
        sdf_from_chemscene_v0_1,
    )


SCHEMA_VERSION = "chemtraverse-0.1"
HEADER = "<chemtraverse v=0.1 units=q1023>"
END = "<chemtraverse_end>"
ORIENTATIONS = frozenset({"F", "R"})
CHIRAL_TOKENS = frozenset({"-", "TCW", "TCCW"})
TETRAHEDRAL_TAGS = frozenset(
    {
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
    }
)


def _encode_utf8(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ChemMolSequenceError("ChemTraverse text values must be non-empty strings")
    return "utf8:" + value.encode("utf-8").hex()


def _decode_utf8(value: str) -> str:
    if not value.startswith("utf8:") or len(value) == 5:
        raise ChemMolSequenceError("ChemTraverse text token must use utf8:<hex>")
    try:
        result = bytes.fromhex(value[5:]).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise ChemMolSequenceError("Invalid ChemTraverse UTF-8 token") from error
    if not result:
        raise ChemMolSequenceError("ChemTraverse text cannot be empty")
    return result


def _line(name: str, *values: object) -> str:
    body = " ".join((name, *(str(value) for value in values)))
    return f"<{body}>"


def _tokens(line: str) -> list[str]:
    if not line.startswith("<") or not line.endswith(">"):
        raise ChemMolSequenceError(f"Malformed ChemTraverse record: {line!r}")
    values = line[1:-1].split()
    if not values:
        raise ChemMolSequenceError("Empty ChemTraverse record")
    return values


def _integer(value: str, field: str, minimum: int | None = None) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ChemMolSequenceError(f"{field} must be an integer") from error
    if minimum is not None and result < minimum:
        raise ChemMolSequenceError(f"{field} must be >= {minimum}")
    return result


def _flag(value: str, field: str) -> bool:
    if value not in {"0", "1"}:
        raise ChemMolSequenceError(f"{field} must be 0 or 1")
    return value == "1"


def _bbox_token(value: tuple[int, int, int, int] | None) -> str:
    return "-" if value is None else ",".join(str(item) for item in value)


def _parse_bbox(value: str) -> list[int] | None:
    if value == "-":
        return None
    parts = value.split(",")
    if len(parts) != 4:
        raise ChemMolSequenceError("ChemTraverse bbox must have four values")
    return [_integer(item, "bbox", 0) for item in parts]


def _atom_key(atom: JunctionRecord, text: str, cip: str) -> tuple[Any, ...]:
    return (
        atom.y,
        atom.x,
        atom.element,
        atom.formal_charge,
        atom.isotope,
        int(atom.aromatic),
        atom.explicit_h_count,
        int(atom.no_implicit),
        atom.radical_electrons,
        cip,
        atom.role,
        text,
    )


def _bond_key(bond: BondObjectRecord) -> tuple[str, ...]:
    return bond.display_type, bond.chemical_type, bond.direction, bond.stereo


def _neighbor_key(
    origin: JunctionRecord,
    neighbor: JunctionRecord,
    bond: BondObjectRecord,
    text: str,
    cip: str,
) -> tuple[Any, ...]:
    dx = neighbor.x - origin.x
    dy = neighbor.y - origin.y
    angle = math.atan2(dy, dx) % (2.0 * math.pi)
    return (
        angle,
        dx * dx + dy * dy,
        *_atom_key(neighbor, text, cip),
        *_bond_key(bond),
    )


def _permutation_is_odd(source: list[tuple[int, int]], target: list[tuple[int, int]]) -> bool:
    if set(source) != set(target) or len(source) != len(target):
        raise ChemMolSequenceError("ChemTraverse chiral neighbor sets disagree")
    ranks = {coordinate: index for index, coordinate in enumerate(target)}
    values = [ranks[coordinate] for coordinate in source]
    inversions = sum(
        values[left] > values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return bool(inversions % 2)


def _flip_chiral(value: str) -> str:
    return {
        "CHI_TETRAHEDRAL_CW": "CHI_TETRAHEDRAL_CCW",
        "CHI_TETRAHEDRAL_CCW": "CHI_TETRAHEDRAL_CW",
        "TCW": "TCCW",
        "TCCW": "TCW",
    }.get(value, value)


def _stable_chiral_by_coordinate(
    parsed: ParsedChemSceneV01,
    adjacency: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]],
) -> dict[tuple[int, int], str]:
    """Canonicalize tetrahedral parity against image-coordinate neighbor order."""

    atom_index = {atom.coordinate: index for index, atom in enumerate(parsed.junctions)}
    result: dict[tuple[int, int], str] = {}
    for atom in parsed.junctions:
        tag = atom.canonical_chiral_tag
        if tag == "CHI_UNSPECIFIED":
            result[atom.coordinate] = "-"
            continue
        if tag not in TETRAHEDRAL_TAGS:
            raise ChemMolSequenceError(
                f"ChemTraverse does not support chiral tag {tag!r}"
            )
        neighbors = [coordinate for _, coordinate in adjacency[atom.coordinate]]
        index_order = sorted(neighbors, key=atom_index.__getitem__)
        coordinate_order = sorted(neighbors, key=lambda value: (value[1], value[0]))
        if _permutation_is_odd(index_order, coordinate_order):
            tag = _flip_chiral(tag)
        result[atom.coordinate] = (
            "TCW" if tag == "CHI_TETRAHEDRAL_CW" else "TCCW"
        )
    return result


def _restore_chiral_tags(
    scene: dict[str, Any], target_chiral_by_atom: dict[int, str]
) -> dict[str, Any]:
    """Convert coordinate-canonical parity to ChemScene's atom-index parity."""

    coordinate_to_index = {
        tuple(item["coordinate"]): index
        for index, item in enumerate(scene["junctions"])
    }
    neighbors: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for bond in scene["bond_objects"]:
        left, right = (tuple(value) for value in bond["endpoints"])
        left_index = coordinate_to_index[left]
        right_index = coordinate_to_index[right]
        neighbors[left_index].append(right)
        neighbors[right_index].append(left)
    for atom_index, token in target_chiral_by_atom.items():
        index_order = sorted(
            neighbors[atom_index], key=coordinate_to_index.__getitem__
        )
        coordinate_order = sorted(
            neighbors[atom_index], key=lambda value: (value[1], value[0])
        )
        tag = (
            "CHI_TETRAHEDRAL_CW" if token == "TCW" else "CHI_TETRAHEDRAL_CCW"
        )
        if _permutation_is_odd(index_order, coordinate_order):
            tag = _flip_chiral(tag)
        scene["junctions"][atom_index]["atom"]["canonical_chiral_tag"] = tag
    return scene


def serialize_chemtraverse_v0_1(
    scene: dict[str, Any] | ParsedChemSceneV01,
) -> str:
    """Serialize a ChemScene into one image-coordinate-canonical DFS sequence."""

    parsed = scene if isinstance(scene, ParsedChemSceneV01) else parse_chemscene_v0_1(scene)
    atom_by_coordinate = {item.coordinate: item for item in parsed.junctions}
    text_record_by_anchor = {item.anchor: item for item in parsed.text_objects}
    text_by_anchor = {key: value.text for key, value in text_record_by_anchor.items()}
    bond_by_index = {index: item for index, item in enumerate(parsed.bond_objects)}
    adjacency: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    for bond_index, bond in bond_by_index.items():
        adjacency[bond.begin].append((bond_index, bond.end))
        adjacency[bond.end].append((bond_index, bond.begin))
    chiral_by_coordinate = _stable_chiral_by_coordinate(parsed, adjacency)

    lines = [HEADER, _line("meta", parsed.width, parsed.height, _encode_utf8(parsed.molecule_id))]
    visited: dict[tuple[int, int], int] = {}
    emitted_bonds: set[int] = set()
    bond_reference: dict[int, int] = {}

    def atom_line(coordinate: tuple[int, int]) -> str:
        atom = atom_by_coordinate[coordinate]
        return _line(
            "atom",
            atom.x,
            atom.y,
            atom.role,
            atom.element,
            atom.formal_charge,
            atom.isotope,
            int(atom.aromatic),
            atom.explicit_h_count,
            int(atom.no_implicit),
            atom.radical_electrons,
            chiral_by_coordinate[coordinate],
        )

    def orientation(bond: BondObjectRecord, current: tuple[int, int]) -> str:
        return "F" if bond.begin == current else "R"

    def emit_bond(name: str, bond_index: int, current: tuple[int, int], *prefix: object) -> None:
        bond = bond_by_index[bond_index]
        bond_reference[bond_index] = len(bond_reference)
        lines.append(
            _line(
                name,
                *prefix,
                orientation(bond, current),
                bond.display_type,
                bond.chemical_type,
                bond.direction,
                bond.stereo,
            )
        )

    def walk(coordinate: tuple[int, int]) -> None:
        visited[coordinate] = len(visited)
        lines.append(atom_line(coordinate))
        origin = atom_by_coordinate[coordinate]
        neighbors = sorted(
            adjacency[coordinate],
            key=lambda item: _neighbor_key(
                origin,
                atom_by_coordinate[item[1]],
                bond_by_index[item[0]],
                text_by_anchor.get(item[1], ""),
                chiral_by_coordinate[item[1]],
            ),
        )
        for bond_index, neighbor in neighbors:
            if bond_index in emitted_bonds:
                continue
            emitted_bonds.add(bond_index)
            if neighbor in visited:
                emit_bond("link", bond_index, coordinate, visited[neighbor])
                continue
            emit_bond("down", bond_index, coordinate)
            walk(neighbor)
            lines.append("<up>")

    roots = sorted(
        parsed.junctions,
        key=lambda atom: _atom_key(
            atom,
            text_by_anchor.get(atom.coordinate, ""),
            chiral_by_coordinate[atom.coordinate],
        ),
    )
    for root in roots:
        if root.coordinate in visited:
            continue
        lines.append("<component>")
        walk(root.coordinate)
        lines.append("<component_end>")

    coordinate_by_reference = {reference: coordinate for coordinate, reference in visited.items()}
    for reference in range(len(coordinate_by_reference)):
        coordinate = coordinate_by_reference[reference]
        text = text_record_by_anchor.get(coordinate)
        if text is None:
            continue
        lines.append(
            _line(
                "text",
                reference,
                text.center_x,
                text.center_y,
                _bbox_token(text.bbox),
                _encode_utf8(text.text),
            )
        )
    for source_bond_index, reference in sorted(
        bond_reference.items(), key=lambda item: item[1]
    ):
        bond = bond_by_index[source_bond_index]
        if bond.stereo_reference_0 is None:
            continue
        lines.append(
            _line(
                "stereo",
                reference,
                visited[bond.stereo_reference_0],
                visited[bond.stereo_reference_1],
            )
        )
    lines.append(END)
    return "\n".join(lines) + "\n"


def parse_chemtraverse_v0_1(sequence: str) -> dict[str, Any]:
    """Parse ChemTraverse and return a strictly validated ChemScene v0.1 object."""

    lines = [line.strip() for line in sequence.splitlines() if line.strip()]
    if len(lines) < 4 or lines[0] != HEADER or lines[-1] != END:
        raise ChemMolSequenceError("ChemTraverse header or terminator is missing")
    meta = _tokens(lines[1])
    if len(meta) != 4 or meta[0] != "meta":
        raise ChemMolSequenceError("ChemTraverse meta record is invalid")
    width = _integer(meta[1], "canvas.width", 1)
    height = _integer(meta[2], "canvas.height", 1)
    molecule_id = _decode_utf8(meta[3])

    junctions: list[dict[str, Any]] = []
    bonds: list[dict[str, Any]] = []
    texts: dict[int, dict[str, Any]] = {}
    stereo_records: dict[int, tuple[int, int]] = {}
    current: int | None = None
    parents: list[int] = []
    pending: tuple[int, list[str]] | None = None
    component_open = False
    postamble = False
    coordinates: set[tuple[int, int]] = set()
    target_chiral_by_atom: dict[int, str] = {}

    def add_atom(values: list[str]) -> None:
        nonlocal current, pending
        if len(values) != 12:
            raise ChemMolSequenceError("ChemTraverse atom record has the wrong arity")
        x = _integer(values[1], "atom.x", 0)
        y = _integer(values[2], "atom.y", 0)
        if x > 1023 or y > 1023 or (x, y) in coordinates:
            raise ChemMolSequenceError("ChemTraverse atom coordinate is invalid or duplicated")
        coordinates.add((x, y))
        atom_index = len(junctions)
        chiral = values[11]
        if chiral not in CHIRAL_TOKENS:
            raise ChemMolSequenceError(
                f"Unknown ChemTraverse chiral token: {chiral!r}"
            )
        if chiral != "-":
            target_chiral_by_atom[atom_index] = chiral
        junctions.append(
            {
                "coordinate": [x, y],
                "role": values[3],
                "atom": {
                    "element": values[4],
                    "formal_charge": _integer(values[5], "atom.formal_charge"),
                    "isotope": _integer(values[6], "atom.isotope", 0),
                    "aromatic": _flag(values[7], "atom.aromatic"),
                    "explicit_h_count": _integer(values[8], "atom.explicit_h_count", 0),
                    "no_implicit": _flag(values[9], "atom.no_implicit"),
                    "radical_electrons": _integer(values[10], "atom.radical_electrons", 0),
                    "canonical_chiral_tag": "CHI_UNSPECIFIED",
                },
            }
        )
        if pending is None:
            if current is not None:
                raise ChemMolSequenceError("A component can have only one root atom")
            current = atom_index
            return
        parent, bond_values = pending
        parents.append(parent)
        current = atom_index
        bonds.append(_bond_payload(parent, atom_index, bond_values))
        pending = None

    def _bond_payload(left: int, right: int, values: list[str]) -> dict[str, Any]:
        if len(values) != 5 or values[0] not in ORIENTATIONS:
            raise ChemMolSequenceError("ChemTraverse bond payload is invalid")
        begin, end = (left, right) if values[0] == "F" else (right, left)
        return {
            "endpoints": [junctions[begin]["coordinate"], junctions[end]["coordinate"]],
            "display_type": values[1],
            "chemical_type": values[2],
            "direction": values[3],
            "stereo": values[4],
            "bbox": None,
        }

    for raw in lines[2:-1]:
        values = _tokens(raw)
        name = values[0]
        if name == "component":
            if len(values) != 1 or component_open or pending is not None or parents:
                raise ChemMolSequenceError("ChemTraverse component start is misplaced")
            if postamble:
                raise ChemMolSequenceError("Component records cannot follow the postamble")
            component_open = True
            current = None
        elif name == "atom":
            if not component_open or postamble:
                raise ChemMolSequenceError("ChemTraverse atom is outside a component")
            add_atom(values)
        elif name == "down":
            if len(values) != 6 or current is None or pending is not None:
                raise ChemMolSequenceError("ChemTraverse down record is invalid")
            pending = (current, values[1:])
        elif name == "link":
            if len(values) != 7 or current is None or pending is not None:
                raise ChemMolSequenceError("ChemTraverse link record is invalid")
            reference = _integer(values[1], "link.reference", 0)
            if reference >= len(junctions) - 1 or reference == current:
                raise ChemMolSequenceError("ChemTraverse link must target an earlier atom")
            bonds.append(_bond_payload(current, reference, values[2:]))
        elif name == "up":
            if len(values) != 1 or pending is not None or not parents:
                raise ChemMolSequenceError("ChemTraverse up record is invalid")
            current = parents.pop()
        elif name == "component_end":
            if (
                len(values) != 1
                or not component_open
                or current is None
                or pending is not None
                or parents
            ):
                raise ChemMolSequenceError("ChemTraverse component end is invalid")
            component_open = False
            current = None
        elif name == "text":
            postamble = True
            if len(values) != 6 or component_open:
                raise ChemMolSequenceError("ChemTraverse text record is invalid")
            reference = _integer(values[1], "text.reference", 0)
            if reference >= len(junctions) or reference in texts:
                raise ChemMolSequenceError("ChemTraverse text reference is invalid")
            texts[reference] = {
                "anchor": junctions[reference]["coordinate"],
                "center": [
                    _integer(values[2], "text.center_x", 0),
                    _integer(values[3], "text.center_y", 0),
                ],
                "bbox": _parse_bbox(values[4]),
                "text": _decode_utf8(values[5]),
            }
        elif name == "stereo":
            postamble = True
            if len(values) != 4 or component_open:
                raise ChemMolSequenceError("ChemTraverse stereo record is invalid")
            bond_reference = _integer(values[1], "stereo.bond_reference", 0)
            references = (
                _integer(values[2], "stereo.atom_reference_0", 0),
                _integer(values[3], "stereo.atom_reference_1", 0),
            )
            if (
                bond_reference >= len(bonds)
                or bond_reference in stereo_records
                or max(references) >= len(junctions)
            ):
                raise ChemMolSequenceError("ChemTraverse stereo reference is invalid")
            stereo_records[bond_reference] = references
        else:
            raise ChemMolSequenceError(f"Unknown ChemTraverse record: {name!r}")

    if component_open or pending is not None or parents or not junctions:
        raise ChemMolSequenceError("ChemTraverse sequence ended with an open traversal")
    for atom_index, junction in enumerate(junctions):
        if (junction["role"] == "TEXT") != (atom_index in texts):
            raise ChemMolSequenceError("ChemTraverse atom role and text records disagree")
    for bond_index, bond in enumerate(bonds):
        if bond_index in stereo_records:
            left, right = stereo_records[bond_index]
            bond["stereo_references"] = [
                junctions[left]["coordinate"],
                junctions[right]["coordinate"],
            ]

    scene = {
        "schema_version": "chemscene-0.1",
        "molecule_id": molecule_id,
        "units": "q1023",
        "canvas": {"width": width, "height": height},
        "junctions": junctions,
        "text_objects": [texts[index] for index in sorted(texts)],
        "bond_objects": bonds,
    }
    parse_chemscene_v0_1(scene)
    return _restore_chiral_tags(scene, target_chiral_by_atom)


def mol_from_chemtraverse_v0_1(
    sequence: str, *, apply_display_directions: bool = False
) -> Chem.Mol:
    return mol_from_chemscene_v0_1(
        parse_chemtraverse_v0_1(sequence),
        apply_display_directions=apply_display_directions,
    )


def sdf_from_chemtraverse_v0_1(sequence: str) -> str:
    return sdf_from_chemscene_v0_1(parse_chemtraverse_v0_1(sequence))


def validate_chemtraverse_v0_1_roundtrip(scene: dict[str, Any]) -> str:
    """Require Scene -> Traverse -> Scene -> Traverse byte identity and SDF compilation."""

    sequence = serialize_chemtraverse_v0_1(scene)
    restored = parse_chemtraverse_v0_1(sequence)
    repeated = serialize_chemtraverse_v0_1(restored)
    if repeated != sequence:
        raise ChemMolSequenceError("ChemTraverse serialization is not canonical")
    sdf_from_chemscene_v0_1(restored)
    return sequence
