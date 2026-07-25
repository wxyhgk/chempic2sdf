#!/usr/bin/env python3
"""ChemScene v0.2 joint AtomObject/BondObject protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem

try:
    from .chemmol_sequence import ChemMolSequenceError
    from .chemscene_v0_1 import (
        build_chemscene_v0_1,
        mol_from_chemscene_v0_1,
        parse_chemscene_v0_1,
        sdf_from_chemscene_v0_1,
        validate_chemscene_v0_1_roundtrip,
    )
except ImportError:
    from chemmol_sequence import ChemMolSequenceError
    from chemscene_v0_1 import (
        build_chemscene_v0_1,
        mol_from_chemscene_v0_1,
        parse_chemscene_v0_1,
        sdf_from_chemscene_v0_1,
        validate_chemscene_v0_1_roundtrip,
    )


SCHEMA_VERSION = "chemscene-0.2"
UNITS = "q1023"
QUANTIZATION_MAX = 1023
ATOM_ROLES = frozenset({"IMPLICIT", "TEXT"})
PORT_SOURCES = frozenset({"ANCHOR", "TEXT_BBOX_RAY"})


@dataclass(frozen=True, slots=True)
class AtomPortRecord:
    bond_id: int
    endpoint: int
    x: int
    y: int
    source: str

    @property
    def coordinate(self) -> tuple[int, int]:
        return self.x, self.y


@dataclass(frozen=True, slots=True)
class AtomObjectRecord:
    atom_id: int
    anchor_x: int
    anchor_y: int
    role: str
    element: str
    formal_charge: int
    isotope: int
    aromatic: bool
    explicit_h_count: int
    no_implicit: bool
    radical_electrons: int
    canonical_chiral_tag: str
    text: str | None
    text_center: tuple[int, int] | None
    text_bbox: tuple[int, int, int, int] | None
    incident_ports: tuple[AtomPortRecord, ...]

    @property
    def anchor(self) -> tuple[int, int]:
        return self.anchor_x, self.anchor_y


@dataclass(frozen=True, slots=True)
class BondObjectRecord:
    bond_id: int
    begin_atom: int
    end_atom: int
    begin_port: tuple[int, int]
    end_port: tuple[int, int]
    display_type: str
    chemical_type: str
    direction: str
    stereo: str
    stereo_atom_refs: tuple[int, int] | None
    bbox: tuple[int, int, int, int] | None


@dataclass(frozen=True, slots=True)
class ParsedChemSceneV02:
    molecule_id: str
    width: int
    height: int
    atom_objects: tuple[AtomObjectRecord, ...]
    bond_objects: tuple[BondObjectRecord, ...]
    schema_version: str = SCHEMA_VERSION
    units: str = UNITS


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ChemMolSequenceError(f"{field} must be an object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ChemMolSequenceError(f"{field} must be a list")
    return value


def _require_string(value: Any, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ChemMolSequenceError(f"{field} must be a string")
    return value


def _require_int(value: Any, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ChemMolSequenceError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ChemMolSequenceError(f"{field} must be >= {minimum}")
    return value


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ChemMolSequenceError(f"{field} must be a boolean")
    return value


def _coordinate(value: Any, field: str) -> tuple[int, int]:
    values = _require_list(value, field)
    if len(values) != 2:
        raise ChemMolSequenceError(f"{field} must contain two coordinates")
    result = tuple(_require_int(item, f"{field}[]", 0) for item in values)
    if max(result) > QUANTIZATION_MAX:
        raise ChemMolSequenceError(f"{field} is outside the quantized canvas")
    return result


def _bbox(value: Any, field: str) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    values = _require_list(value, field)
    if len(values) != 4:
        raise ChemMolSequenceError(f"{field} must contain four coordinates")
    result = tuple(_require_int(item, f"{field}[]", 0) for item in values)
    if max(result) > QUANTIZATION_MAX:
        raise ChemMolSequenceError(f"{field} is outside the quantized canvas")
    if result[0] > result[2] or result[1] > result[3]:
        raise ChemMolSequenceError(f"{field} has inverted bounds")
    return result


def _enum_name(enum_type: Any, value: Any, field: str) -> str:
    name = _require_string(value, field)
    if not hasattr(enum_type, name):
        raise ChemMolSequenceError(f"Unknown {field}: {name!r}")
    return name


def _clip_port_to_text_bbox(
    anchor: tuple[int, int],
    target: tuple[int, int],
    bbox: tuple[int, int, int, int] | None,
) -> tuple[tuple[int, int], str]:
    if bbox is None:
        return anchor, "ANCHOR"
    ax, ay = anchor
    tx, ty = target
    x0, y0, x1, y1 = bbox
    if not (x0 <= ax <= x1 and y0 <= ay <= y1):
        return anchor, "ANCHOR"
    dx, dy = tx - ax, ty - ay
    candidates: list[tuple[float, float, float]] = []
    if dx:
        boundary_x = x1 if dx > 0 else x0
        t = (boundary_x - ax) / dx
        y = ay + t * dy
        if 0.0 <= t <= 1.0 and y0 - 1e-6 <= y <= y1 + 1e-6:
            candidates.append((t, float(boundary_x), y))
    if dy:
        boundary_y = y1 if dy > 0 else y0
        t = (boundary_y - ay) / dy
        x = ax + t * dx
        if 0.0 <= t <= 1.0 and x0 - 1e-6 <= x <= x1 + 1e-6:
            candidates.append((t, x, float(boundary_y)))
    if not candidates:
        return anchor, "ANCHOR"
    _, x, y = min(candidates, key=lambda item: item[0])
    return (
        max(0, min(QUANTIZATION_MAX, round(x))),
        max(0, min(QUANTIZATION_MAX, round(y))),
    ), "TEXT_BBOX_RAY"


def parse_chemscene_v0_2(scene: dict[str, Any]) -> ParsedChemSceneV02:
    scene = _require_dict(scene, "scene")
    if scene.get("schema_version") != SCHEMA_VERSION:
        raise ChemMolSequenceError("Unsupported ChemScene schema version")
    if scene.get("units") != UNITS:
        raise ChemMolSequenceError("ChemScene v0.2 coordinates must use q1023")
    molecule_id = _require_string(scene.get("molecule_id"), "molecule_id")
    canvas = _require_dict(scene.get("canvas"), "canvas")
    width = _require_int(canvas.get("width"), "canvas.width", 1)
    height = _require_int(canvas.get("height"), "canvas.height", 1)

    atom_objects: list[AtomObjectRecord] = []
    anchors = set()
    atom_payloads = _require_list(scene.get("atom_objects"), "atom_objects")
    for index, payload in enumerate(atom_payloads):
        item = _require_dict(payload, f"atom_objects[{index}]")
        atom_id = _require_int(item.get("id"), f"atom_objects[{index}].id", 0)
        if atom_id != index:
            raise ChemMolSequenceError("AtomObject IDs must be dense and ordered")
        anchor = _coordinate(item.get("anchor"), f"atom_objects[{index}].anchor")
        if anchor in anchors:
            raise ChemMolSequenceError("ChemScene contains duplicate AtomObject anchors")
        anchors.add(anchor)
        role = _require_string(item.get("role"), f"atom_objects[{index}].role")
        if role not in ATOM_ROLES:
            raise ChemMolSequenceError(f"Unknown AtomObject role: {role!r}")
        chemical = _require_dict(item.get("chemical"), "atom_object.chemical")
        text_payload = item.get("text")
        if text_payload is None:
            text = None
            text_center = None
            text_bbox = None
        else:
            text_item = _require_dict(text_payload, "atom_object.text")
            text = _require_string(text_item.get("value"), "atom_object.text.value")
            text_center = _coordinate(
                text_item.get("center"), "atom_object.text.center"
            )
            text_bbox = _bbox(text_item.get("bbox"), "atom_object.text.bbox")
            if text_bbox is None:
                raise ChemMolSequenceError("TEXT AtomObject must contain a text bbox")
        if (role == "TEXT") != (text is not None):
            raise ChemMolSequenceError("AtomObject role and text payload disagree")
        ports = []
        for port_index, port_payload in enumerate(
            _require_list(item.get("incident_ports"), "atom_object.incident_ports")
        ):
            port = _require_dict(port_payload, f"incident_ports[{port_index}]")
            endpoint = _require_int(port.get("endpoint"), "port.endpoint", 0)
            if endpoint not in {0, 1}:
                raise ChemMolSequenceError("AtomObject port endpoint must be 0 or 1")
            source = _require_string(port.get("source"), "port.source")
            if source not in PORT_SOURCES:
                raise ChemMolSequenceError(f"Unknown AtomObject port source: {source!r}")
            coordinate = _coordinate(port.get("coordinate"), "port.coordinate")
            if source == "ANCHOR" and coordinate != anchor:
                raise ChemMolSequenceError("ANCHOR port must equal its AtomObject anchor")
            if source == "TEXT_BBOX_RAY" and role != "TEXT":
                raise ChemMolSequenceError("TEXT_BBOX_RAY port requires a TEXT AtomObject")
            ports.append(
                AtomPortRecord(
                    bond_id=_require_int(port.get("bond_id"), "port.bond_id", 0),
                    endpoint=endpoint,
                    x=coordinate[0],
                    y=coordinate[1],
                    source=source,
                )
            )
        atom_objects.append(
            AtomObjectRecord(
                atom_id=atom_id,
                anchor_x=anchor[0],
                anchor_y=anchor[1],
                role=role,
                element=_require_string(chemical.get("element"), "chemical.element"),
                formal_charge=_require_int(
                    chemical.get("formal_charge"), "chemical.formal_charge"
                ),
                isotope=_require_int(chemical.get("isotope"), "chemical.isotope", 0),
                aromatic=_require_bool(chemical.get("aromatic"), "chemical.aromatic"),
                explicit_h_count=_require_int(
                    chemical.get("explicit_h_count"), "chemical.explicit_h_count", 0
                ),
                no_implicit=_require_bool(
                    chemical.get("no_implicit"), "chemical.no_implicit"
                ),
                radical_electrons=_require_int(
                    chemical.get("radical_electrons"),
                    "chemical.radical_electrons",
                    0,
                ),
                canonical_chiral_tag=_enum_name(
                    Chem.ChiralType,
                    chemical.get("canonical_chiral_tag"),
                    "chemical.canonical_chiral_tag",
                ),
                text=text,
                text_center=text_center,
                text_bbox=text_bbox,
                incident_ports=tuple(ports),
            )
        )
    if not atom_objects:
        raise ChemMolSequenceError("ChemScene must contain at least one AtomObject")

    bond_objects: list[BondObjectRecord] = []
    expected_ports = {}
    adjacency = {atom.atom_id: set() for atom in atom_objects}
    atom_pairs: set[tuple[int, int]] = set()
    bond_payloads = _require_list(scene.get("bond_objects"), "bond_objects")
    for index, payload in enumerate(bond_payloads):
        item = _require_dict(payload, f"bond_objects[{index}]")
        bond_id = _require_int(item.get("id"), f"bond_objects[{index}].id", 0)
        if bond_id != index:
            raise ChemMolSequenceError("BondObject IDs must be dense and ordered")
        atom_refs = _require_list(item.get("atom_refs"), "bond_object.atom_refs")
        if len(atom_refs) != 2:
            raise ChemMolSequenceError("BondObject must reference two AtomObjects")
        begin_atom = _require_int(atom_refs[0], "bond_object.atom_refs[0]", 0)
        end_atom = _require_int(atom_refs[1], "bond_object.atom_refs[1]", 0)
        if (
            begin_atom == end_atom
            or begin_atom >= len(atom_objects)
            or end_atom >= len(atom_objects)
        ):
            raise ChemMolSequenceError("BondObject AtomObject references are invalid")
        atom_pair = tuple(sorted((begin_atom, end_atom)))
        if atom_pair in atom_pairs:
            raise ChemMolSequenceError("ChemScene contains duplicate BondObjects")
        atom_pairs.add(atom_pair)
        ports = _require_list(item.get("ports"), "bond_object.ports")
        if len(ports) != 2:
            raise ChemMolSequenceError("BondObject must contain two visible ports")
        begin_port = _coordinate(ports[0], "bond_object.ports[0]")
        end_port = _coordinate(ports[1], "bond_object.ports[1]")
        stereo_refs_payload = item.get("stereo_atom_refs")
        stereo = _enum_name(Chem.BondStereo, item.get("stereo"), "bond_object.stereo")
        if stereo_refs_payload is None:
            stereo_refs = None
        else:
            values = _require_list(stereo_refs_payload, "bond_object.stereo_atom_refs")
            if len(values) != 2:
                raise ChemMolSequenceError("stereo_atom_refs must contain two atoms")
            stereo_refs = (
                _require_int(values[0], "stereo_atom_refs[0]", 0),
                _require_int(values[1], "stereo_atom_refs[1]", 0),
            )
            if max(stereo_refs) >= len(atom_objects):
                raise ChemMolSequenceError("Stereo AtomObject reference is invalid")
        needs_refs = stereo not in {"STEREONONE", "STEREOANY"}
        if needs_refs != (stereo_refs is not None):
            raise ChemMolSequenceError("BondObject stereo and atom references disagree")
        bond_objects.append(
            BondObjectRecord(
                bond_id=bond_id,
                begin_atom=begin_atom,
                end_atom=end_atom,
                begin_port=begin_port,
                end_port=end_port,
                display_type=_enum_name(
                    Chem.BondType, item.get("display_type"), "bond_object.display_type"
                ),
                chemical_type=_enum_name(
                    Chem.BondType,
                    item.get("chemical_type"),
                    "bond_object.chemical_type",
                ),
                direction=_enum_name(
                    Chem.BondDir, item.get("direction"), "bond_object.direction"
                ),
                stereo=stereo,
                stereo_atom_refs=stereo_refs,
                bbox=_bbox(item.get("bbox"), "bond_object.bbox"),
            )
        )
        expected_ports[(bond_id, 0)] = (begin_atom, begin_port)
        expected_ports[(bond_id, 1)] = (end_atom, end_port)
        adjacency[begin_atom].add(end_atom)
        adjacency[end_atom].add(begin_atom)

    actual_ports = {}
    for atom in atom_objects:
        for port in atom.incident_ports:
            key = (port.bond_id, port.endpoint)
            if key in actual_ports:
                raise ChemMolSequenceError("A BondObject endpoint has multiple AtomObject owners")
            actual_ports[key] = (atom.atom_id, port.coordinate)
    if actual_ports != expected_ports:
        raise ChemMolSequenceError("AtomObject incident ports and BondObjects disagree")
    for bond in bond_objects:
        if bond.stereo_atom_refs is None:
            continue
        if bond.stereo_atom_refs[0] not in adjacency[bond.begin_atom]:
            raise ChemMolSequenceError("First stereo atom is not adjacent to begin atom")
        if bond.stereo_atom_refs[1] not in adjacency[bond.end_atom]:
            raise ChemMolSequenceError("Second stereo atom is not adjacent to end atom")
    return ParsedChemSceneV02(
        molecule_id=molecule_id,
        width=width,
        height=height,
        atom_objects=tuple(atom_objects),
        bond_objects=tuple(bond_objects),
    )


def build_chemscene_v0_2(
    chemmol: dict[str, Any],
    render: dict[str, Any],
    layers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    v01 = build_chemscene_v0_1(chemmol, render, layers)
    parsed = parse_chemscene_v0_1(v01)
    atom_id_by_anchor = {
        atom.coordinate: atom_id for atom_id, atom in enumerate(parsed.junctions)
    }
    text_by_anchor = {item.anchor: item for item in parsed.text_objects}
    atom_objects = []
    for atom_id, junction in enumerate(parsed.junctions):
        text = text_by_anchor.get(junction.coordinate)
        atom_objects.append(
            {
                "id": atom_id,
                "anchor": list(junction.coordinate),
                "role": junction.role,
                "chemical": {
                    "element": junction.element,
                    "formal_charge": junction.formal_charge,
                    "isotope": junction.isotope,
                    "aromatic": junction.aromatic,
                    "explicit_h_count": junction.explicit_h_count,
                    "no_implicit": junction.no_implicit,
                    "radical_electrons": junction.radical_electrons,
                    "canonical_chiral_tag": junction.canonical_chiral_tag,
                },
                "text": (
                    {
                        "value": text.text,
                        "center": [text.center_x, text.center_y],
                        "bbox": list(text.bbox),
                    }
                    if text is not None
                    else None
                ),
                "incident_ports": [],
            }
        )
    bond_objects = []
    for bond_id, bond in enumerate(parsed.bond_objects):
        begin_atom = atom_id_by_anchor[bond.begin]
        end_atom = atom_id_by_anchor[bond.end]
        begin_bbox = atom_objects[begin_atom]["text"]
        end_bbox = atom_objects[end_atom]["text"]
        begin_port, begin_source = _clip_port_to_text_bbox(
            bond.begin,
            bond.end,
            tuple(begin_bbox["bbox"]) if begin_bbox is not None else None,
        )
        end_port, end_source = _clip_port_to_text_bbox(
            bond.end,
            bond.begin,
            tuple(end_bbox["bbox"]) if end_bbox is not None else None,
        )
        atom_objects[begin_atom]["incident_ports"].append(
            {
                "bond_id": bond_id,
                "endpoint": 0,
                "coordinate": list(begin_port),
                "source": begin_source,
            }
        )
        atom_objects[end_atom]["incident_ports"].append(
            {
                "bond_id": bond_id,
                "endpoint": 1,
                "coordinate": list(end_port),
                "source": end_source,
            }
        )
        item = {
            "id": bond_id,
            "atom_refs": [begin_atom, end_atom],
            "ports": [list(begin_port), list(end_port)],
            "display_type": bond.display_type,
            "chemical_type": bond.chemical_type,
            "direction": bond.direction,
            "stereo": bond.stereo,
            "bbox": list(bond.bbox) if bond.bbox is not None else None,
        }
        if bond.stereo_reference_0 is not None:
            item["stereo_atom_refs"] = [
                atom_id_by_anchor[bond.stereo_reference_0],
                atom_id_by_anchor[bond.stereo_reference_1],
            ]
        bond_objects.append(item)
    scene = {
        "schema_version": SCHEMA_VERSION,
        "molecule_id": parsed.molecule_id,
        "units": UNITS,
        "canvas": {"width": parsed.width, "height": parsed.height},
        "atom_objects": atom_objects,
        "bond_objects": bond_objects,
    }
    parse_chemscene_v0_2(scene)
    return scene


def chemscene_v0_1_from_v0_2(
    scene: dict[str, Any] | ParsedChemSceneV02,
) -> dict[str, Any]:
    parsed = scene if isinstance(scene, ParsedChemSceneV02) else parse_chemscene_v0_2(scene)
    anchors = {atom.atom_id: atom.anchor for atom in parsed.atom_objects}
    junctions = []
    text_objects = []
    for atom in parsed.atom_objects:
        junctions.append(
            {
                "coordinate": list(atom.anchor),
                "role": atom.role,
                "atom": {
                    "element": atom.element,
                    "formal_charge": atom.formal_charge,
                    "isotope": atom.isotope,
                    "aromatic": atom.aromatic,
                    "explicit_h_count": atom.explicit_h_count,
                    "no_implicit": atom.no_implicit,
                    "radical_electrons": atom.radical_electrons,
                    "canonical_chiral_tag": atom.canonical_chiral_tag,
                },
            }
        )
        if atom.text is not None:
            text_objects.append(
                {
                    "anchor": list(atom.anchor),
                    "center": list(atom.text_center),
                    "bbox": list(atom.text_bbox),
                    "text": atom.text,
                }
            )
    bond_objects = []
    for bond in parsed.bond_objects:
        item = {
            "endpoints": [
                list(anchors[bond.begin_atom]),
                list(anchors[bond.end_atom]),
            ],
            "display_type": bond.display_type,
            "chemical_type": bond.chemical_type,
            "direction": bond.direction,
            "stereo": bond.stereo,
            "bbox": list(bond.bbox) if bond.bbox is not None else None,
        }
        if bond.stereo_atom_refs is not None:
            item["stereo_references"] = [
                list(anchors[bond.stereo_atom_refs[0]]),
                list(anchors[bond.stereo_atom_refs[1]]),
            ]
        bond_objects.append(item)
    result = {
        "schema_version": "chemscene-0.1",
        "molecule_id": parsed.molecule_id,
        "units": parsed.units,
        "canvas": {"width": parsed.width, "height": parsed.height},
        "junctions": junctions,
        "text_objects": text_objects,
        "bond_objects": bond_objects,
    }
    parse_chemscene_v0_1(result)
    return result


def mol_from_chemscene_v0_2(scene: dict[str, Any] | ParsedChemSceneV02) -> Chem.Mol:
    return mol_from_chemscene_v0_1(chemscene_v0_1_from_v0_2(scene))


def sdf_from_chemscene_v0_2(scene: dict[str, Any] | ParsedChemSceneV02) -> str:
    return sdf_from_chemscene_v0_1(chemscene_v0_1_from_v0_2(scene))


def validate_chemscene_v0_2_roundtrip(
    scene: dict[str, Any], expected_smiles: str, expected_inchikey: str
) -> Chem.Mol:
    parse_chemscene_v0_2(scene)
    return validate_chemscene_v0_1_roundtrip(
        chemscene_v0_1_from_v0_2(scene), expected_smiles, expected_inchikey
    )
