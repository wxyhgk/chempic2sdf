#!/usr/bin/env python3
"""ChemScene v0.1 object protocol and deterministic SDF compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdDepictor

try:
    from .chemmol_sequence import ChemMolSequenceError
    from .chemsvg_sequence_v4 import (
        ParsedChemSVGSequenceV4,
        StrokeRecordV4,
        VertexRecordV4,
        mol_from_parsed_chemsvg_v4,
        parse_chemsvg_v4,
        serialize_chemsvg_v4,
        serialize_parsed_chemsvg_v4,
    )
except ImportError:
    from chemmol_sequence import ChemMolSequenceError
    from chemsvg_sequence_v4 import (
        ParsedChemSVGSequenceV4,
        StrokeRecordV4,
        VertexRecordV4,
        mol_from_parsed_chemsvg_v4,
        parse_chemsvg_v4,
        serialize_chemsvg_v4,
        serialize_parsed_chemsvg_v4,
    )


SCHEMA_VERSION = "chemscene-0.1"
UNITS = "q1023"
QUANTIZATION_MAX = 1023
JUNCTION_ROLES = frozenset({"IMPLICIT", "TEXT"})


class ChemSceneDepictionConflictError(ChemMolSequenceError):
    """Raised when source visual geometry contradicts the chemical stereo label."""


class ChemSceneSDFRoundtripError(ChemMolSequenceError):
    """Raised when an RDKit molecule cannot be written to a lossless SDF record."""


@dataclass(frozen=True, slots=True)
class JunctionRecord:
    x: int
    y: int
    role: str
    element: str
    formal_charge: int
    isotope: int
    aromatic: bool
    explicit_h_count: int
    no_implicit: bool
    radical_electrons: int
    canonical_chiral_tag: str

    @property
    def coordinate(self) -> tuple[int, int]:
        return self.x, self.y


@dataclass(frozen=True, slots=True)
class TextObjectRecord:
    anchor_x: int
    anchor_y: int
    center_x: int
    center_y: int
    text: str
    bbox: tuple[int, int, int, int] | None = None

    @property
    def anchor(self) -> tuple[int, int]:
        return self.anchor_x, self.anchor_y


@dataclass(frozen=True, slots=True)
class BondObjectRecord:
    begin_x: int
    begin_y: int
    end_x: int
    end_y: int
    display_type: str
    chemical_type: str
    direction: str
    stereo: str
    stereo_reference_0_x: int = -1
    stereo_reference_0_y: int = -1
    stereo_reference_1_x: int = -1
    stereo_reference_1_y: int = -1
    bbox: tuple[int, int, int, int] | None = None

    @property
    def begin(self) -> tuple[int, int]:
        return self.begin_x, self.begin_y

    @property
    def end(self) -> tuple[int, int]:
        return self.end_x, self.end_y

    @property
    def stereo_reference_0(self) -> tuple[int, int] | None:
        if self.stereo_reference_0_x < 0:
            return None
        return self.stereo_reference_0_x, self.stereo_reference_0_y

    @property
    def stereo_reference_1(self) -> tuple[int, int] | None:
        if self.stereo_reference_1_x < 0:
            return None
        return self.stereo_reference_1_x, self.stereo_reference_1_y


@dataclass(frozen=True, slots=True)
class ParsedChemSceneV01:
    molecule_id: str
    width: int
    height: int
    junctions: tuple[JunctionRecord, ...]
    text_objects: tuple[TextObjectRecord, ...]
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
    coordinate = (
        _require_int(values[0], f"{field}[0]", 0),
        _require_int(values[1], f"{field}[1]", 0),
    )
    if max(coordinate) > QUANTIZATION_MAX:
        raise ChemMolSequenceError(f"{field} is outside the quantized canvas")
    return coordinate


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


def _quantize(value: float, maximum: int) -> int:
    return max(
        0,
        min(
            QUANTIZATION_MAX,
            round(float(value) / max(maximum - 1, 1) * QUANTIZATION_MAX),
        ),
    )


def _quantized_bbox(value: Any, width: int, height: int) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ChemMolSequenceError("Layer bbox must contain four values")
    return [
        _quantize(value[0], width),
        _quantize(value[1], height),
        _quantize(value[2], width),
        _quantize(value[3], height),
    ]


def _bond_sort_key(record: BondObjectRecord) -> tuple[Any, ...]:
    endpoints = sorted(((record.begin_y, record.begin_x), (record.end_y, record.end_x)))
    return endpoints[0], endpoints[1], record.display_type, record.chemical_type


def parse_chemscene_v0_1(scene: dict[str, Any]) -> ParsedChemSceneV01:
    """Strictly validate a ChemScene object without inferring missing geometry."""

    scene = _require_dict(scene, "scene")
    if scene.get("schema_version") != SCHEMA_VERSION:
        raise ChemMolSequenceError("Unsupported ChemScene schema version")
    if scene.get("units") != UNITS:
        raise ChemMolSequenceError("ChemScene v0.1 coordinates must use q1023")
    molecule_id = _require_string(scene.get("molecule_id"), "molecule_id")
    canvas = _require_dict(scene.get("canvas"), "canvas")
    width = _require_int(canvas.get("width"), "canvas.width", 1)
    height = _require_int(canvas.get("height"), "canvas.height", 1)

    junctions: list[JunctionRecord] = []
    coordinate_set: set[tuple[int, int]] = set()
    for index, payload in enumerate(_require_list(scene.get("junctions"), "junctions")):
        item = _require_dict(payload, f"junctions[{index}]")
        coordinate = _coordinate(item.get("coordinate"), f"junctions[{index}].coordinate")
        if coordinate in coordinate_set:
            raise ChemMolSequenceError("ChemScene contains duplicate junction coordinates")
        coordinate_set.add(coordinate)
        role = _require_string(item.get("role"), f"junctions[{index}].role")
        if role not in JUNCTION_ROLES:
            raise ChemMolSequenceError(f"Unknown junction role: {role!r}")
        atom = _require_dict(item.get("atom"), f"junctions[{index}].atom")
        junctions.append(
            JunctionRecord(
                x=coordinate[0],
                y=coordinate[1],
                role=role,
                element=_require_string(atom.get("element"), "atom.element"),
                formal_charge=_require_int(atom.get("formal_charge"), "atom.formal_charge"),
                isotope=_require_int(atom.get("isotope"), "atom.isotope", 0),
                aromatic=_require_bool(atom.get("aromatic"), "atom.aromatic"),
                explicit_h_count=_require_int(
                    atom.get("explicit_h_count"), "atom.explicit_h_count", 0
                ),
                no_implicit=_require_bool(atom.get("no_implicit"), "atom.no_implicit"),
                radical_electrons=_require_int(
                    atom.get("radical_electrons"), "atom.radical_electrons", 0
                ),
                canonical_chiral_tag=_enum_name(
                    Chem.ChiralType,
                    atom.get("canonical_chiral_tag"),
                    "atom.canonical_chiral_tag",
                ),
            )
        )
    if not junctions:
        raise ChemMolSequenceError("ChemScene must contain at least one junction")

    text_objects: list[TextObjectRecord] = []
    text_anchors: set[tuple[int, int]] = set()
    for index, payload in enumerate(
        _require_list(scene.get("text_objects"), "text_objects")
    ):
        item = _require_dict(payload, f"text_objects[{index}]")
        anchor = _coordinate(item.get("anchor"), f"text_objects[{index}].anchor")
        center = _coordinate(item.get("center"), f"text_objects[{index}].center")
        if anchor not in coordinate_set:
            raise ChemMolSequenceError("TextObject anchor is not a Junction")
        if anchor in text_anchors:
            raise ChemMolSequenceError("A Junction cannot own multiple TextObjects")
        text_anchors.add(anchor)
        text_objects.append(
            TextObjectRecord(
                anchor_x=anchor[0],
                anchor_y=anchor[1],
                center_x=center[0],
                center_y=center[1],
                text=_require_string(item.get("text"), "text_object.text"),
                bbox=_bbox(item.get("bbox"), "text_object.bbox"),
            )
        )

    role_by_coordinate = {record.coordinate: record.role for record in junctions}
    for coordinate, role in role_by_coordinate.items():
        if (role == "TEXT") != (coordinate in text_anchors):
            raise ChemMolSequenceError("Junction role and TextObject ownership disagree")

    bond_objects: list[BondObjectRecord] = []
    endpoint_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for index, payload in enumerate(
        _require_list(scene.get("bond_objects"), "bond_objects")
    ):
        item = _require_dict(payload, f"bond_objects[{index}]")
        endpoints = _require_list(item.get("endpoints"), f"bond_objects[{index}].endpoints")
        if len(endpoints) != 2:
            raise ChemMolSequenceError("BondObject must have exactly two endpoints")
        begin = _coordinate(endpoints[0], "bond_object.begin")
        end = _coordinate(endpoints[1], "bond_object.end")
        if begin == end or begin not in coordinate_set or end not in coordinate_set:
            raise ChemMolSequenceError("BondObject endpoints must be distinct Junctions")
        pair = tuple(sorted((begin, end)))
        if pair in endpoint_pairs:
            raise ChemMolSequenceError("ChemScene contains duplicate BondObjects")
        endpoint_pairs.add(pair)

        references_payload = item.get("stereo_references")
        if references_payload is None:
            reference_0 = reference_1 = (-1, -1)
        else:
            references = _require_list(references_payload, "bond_object.stereo_references")
            if len(references) != 2:
                raise ChemMolSequenceError("stereo_references must contain two Junctions")
            reference_0 = _coordinate(references[0], "stereo_reference_0")
            reference_1 = _coordinate(references[1], "stereo_reference_1")
            if reference_0 not in coordinate_set or reference_1 not in coordinate_set:
                raise ChemMolSequenceError("Stereo reference is not a Junction")

        stereo = _enum_name(Chem.BondStereo, item.get("stereo"), "bond_object.stereo")
        needs_references = stereo not in {"STEREONONE", "STEREOANY"}
        if needs_references != (references_payload is not None):
            raise ChemMolSequenceError("BondObject stereo and references disagree")
        bond_objects.append(
            BondObjectRecord(
                begin_x=begin[0],
                begin_y=begin[1],
                end_x=end[0],
                end_y=end[1],
                display_type=_enum_name(
                    Chem.BondType,
                    item.get("display_type"),
                    "bond_object.display_type",
                ),
                chemical_type=_enum_name(
                    Chem.BondType,
                    item.get("chemical_type"),
                    "bond_object.chemical_type",
                ),
                direction=_enum_name(
                    Chem.BondDir,
                    item.get("direction"),
                    "bond_object.direction",
                ),
                stereo=stereo,
                stereo_reference_0_x=reference_0[0],
                stereo_reference_0_y=reference_0[1],
                stereo_reference_1_x=reference_1[0],
                stereo_reference_1_y=reference_1[1],
                bbox=_bbox(item.get("bbox"), "bond_object.bbox"),
            )
        )

    adjacency = {coordinate: set() for coordinate in coordinate_set}
    for bond in bond_objects:
        adjacency[bond.begin].add(bond.end)
        adjacency[bond.end].add(bond.begin)
    for bond in bond_objects:
        if bond.stereo_reference_0 is None:
            continue
        if bond.stereo_reference_0 not in adjacency[bond.begin]:
            raise ChemMolSequenceError("First stereo reference is not adjacent to begin")
        if bond.stereo_reference_1 not in adjacency[bond.end]:
            raise ChemMolSequenceError("Second stereo reference is not adjacent to end")

    return ParsedChemSceneV01(
        molecule_id=molecule_id,
        width=width,
        height=height,
        junctions=tuple(junctions),
        text_objects=tuple(text_objects),
        bond_objects=tuple(bond_objects),
    )


def build_chemscene_v0_1(
    chemmol: dict[str, Any],
    render: dict[str, Any],
    layers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an Oracle ChemScene target from existing structure/render labels."""

    molecule_id = chemmol.get("molecule_id")
    if molecule_id != render.get("molecule_id"):
        raise ChemMolSequenceError("ChemMol and Render molecule_id values differ")
    width = _require_int(render.get("width"), "render.width", 1)
    height = _require_int(render.get("height"), "render.height", 1)
    if layers is not None:
        if layers.get("molecule_id") != molecule_id:
            raise ChemMolSequenceError("Layers and ChemMol molecule_id values differ")
        if layers.get("width") != width or layers.get("height") != height:
            raise ChemMolSequenceError("Layers and Render canvas sizes differ")

    parsed_svg = parse_chemsvg_v4(serialize_chemsvg_v4(chemmol, render))
    render_atoms = {item["atom_id"]: item for item in render["atoms"]}
    atom_id_by_coordinate = {
        (item["quantized_x"], item["quantized_y"]): atom_id
        for atom_id, item in render_atoms.items()
    }
    layer_atoms = (
        {item["atom_id"]: item for item in layers.get("atoms", [])}
        if layers is not None
        else {}
    )

    junctions = []
    text_objects = []
    for vertex in parsed_svg.vertices:
        coordinate = (vertex.x, vertex.y)
        atom_id = atom_id_by_coordinate.get(coordinate)
        layer_atom = layer_atoms.get(atom_id)
        visible_text = bool(layer_atom and layer_atom.get("visible_text"))
        junctions.append(
            {
                "coordinate": [vertex.x, vertex.y],
                "role": "TEXT" if visible_text else "IMPLICIT",
                "atom": {
                    "element": vertex.element,
                    "formal_charge": vertex.formal_charge,
                    "isotope": vertex.isotope,
                    "aromatic": vertex.aromatic,
                    "explicit_h_count": vertex.explicit_h_count,
                    "no_implicit": vertex.no_implicit,
                    "radical_electrons": vertex.radical_electrons,
                    "canonical_chiral_tag": vertex.canonical_chiral_tag,
                },
            }
        )
        if visible_text:
            center = layer_atom.get("center")
            if not isinstance(center, list) or len(center) != 2:
                raise ChemMolSequenceError("Visible layer atom has no valid center")
            text_objects.append(
                {
                    "anchor": [vertex.x, vertex.y],
                    "center": [
                        _quantize(center[0], width),
                        _quantize(center[1], height),
                    ],
                    "bbox": _quantized_bbox(layer_atom.get("bbox"), width, height),
                    "text": _require_string(layer_atom.get("text"), "layer.atom.text"),
                }
            )

    layer_bonds_by_pair: dict[tuple[tuple[int, int], tuple[int, int]], dict[str, Any]] = {}
    if layers is not None:
        for item in layers.get("bonds", []):
            pair = tuple(
                sorted(
                    (
                        (
                            _quantize(item["begin"][0], width),
                            _quantize(item["begin"][1], height),
                        ),
                        (
                            _quantize(item["end"][0], width),
                            _quantize(item["end"][1], height),
                        ),
                    )
                )
            )
            layer_bonds_by_pair[pair] = item

    bond_objects = []
    for stroke in parsed_svg.strokes:
        pair = tuple(sorted((stroke.begin, stroke.end)))
        layer_bond = layer_bonds_by_pair.get(pair)
        item: dict[str, Any] = {
            "endpoints": [list(stroke.begin), list(stroke.end)],
            "display_type": stroke.display_type,
            "chemical_type": stroke.bond_type,
            "direction": stroke.direction,
            "stereo": stroke.stereo,
            "bbox": (
                _quantized_bbox(layer_bond.get("bbox"), width, height)
                if layer_bond is not None
                else None
            ),
        }
        if stroke.stereo_atom_0 is not None:
            item["stereo_references"] = [
                list(stroke.stereo_atom_0),
                list(stroke.stereo_atom_1),
            ]
        bond_objects.append(item)

    scene = {
        "schema_version": SCHEMA_VERSION,
        "molecule_id": molecule_id,
        "units": UNITS,
        "canvas": {"width": width, "height": height},
        "junctions": junctions,
        "text_objects": text_objects,
        "bond_objects": bond_objects,
    }
    parse_chemscene_v0_1(scene)
    return scene


def parsed_chemsvg_from_chemscene_v0_1(
    scene: dict[str, Any] | ParsedChemSceneV01,
) -> ParsedChemSVGSequenceV4:
    parsed = scene if isinstance(scene, ParsedChemSceneV01) else parse_chemscene_v0_1(scene)
    vertices = tuple(
        VertexRecordV4(
            x=item.x,
            y=item.y,
            element=item.element,
            formal_charge=item.formal_charge,
            isotope=item.isotope,
            aromatic=item.aromatic,
            explicit_h_count=item.explicit_h_count,
            no_implicit=item.no_implicit,
            radical_electrons=item.radical_electrons,
            canonical_chiral_tag=item.canonical_chiral_tag,
        )
        for item in parsed.junctions
    )
    strokes = tuple(
        StrokeRecordV4(
            begin_x=item.begin_x,
            begin_y=item.begin_y,
            end_x=item.end_x,
            end_y=item.end_y,
            display_type=item.display_type,
            bond_type=item.chemical_type,
            direction=item.direction,
            stereo=item.stereo,
            stereo_atom_0_x=item.stereo_reference_0_x,
            stereo_atom_0_y=item.stereo_reference_0_y,
            stereo_atom_1_x=item.stereo_reference_1_x,
            stereo_atom_1_y=item.stereo_reference_1_y,
        )
        for item in sorted(parsed.bond_objects, key=_bond_sort_key)
    )
    return ParsedChemSVGSequenceV4(vertices=vertices, strokes=strokes)


def chemsvg_from_chemscene_v0_1(
    scene: dict[str, Any] | ParsedChemSceneV01,
) -> str:
    return serialize_parsed_chemsvg_v4(parsed_chemsvg_from_chemscene_v0_1(scene))


def mol_from_chemscene_v0_1(
    scene: dict[str, Any] | ParsedChemSceneV01,
    *,
    apply_display_directions: bool = False,
) -> Chem.Mol:
    parsed = scene if isinstance(scene, ParsedChemSceneV01) else parse_chemscene_v0_1(scene)
    mol = mol_from_parsed_chemsvg_v4(
        parsed_chemsvg_from_chemscene_v0_1(parsed),
        apply_display_directions=apply_display_directions,
    )
    text_by_anchor = {item.anchor: item.text for item in parsed.text_objects}
    for atom_index, junction in enumerate(parsed.junctions):
        if junction.element != "*":
            continue
        alias = text_by_anchor.get(junction.coordinate)
        if alias:
            mol.GetAtomWithIdx(atom_index).SetProp("molFileAlias", alias)
    return mol


def _mol_signature(mol: Chem.Mol) -> tuple[str, str]:
    return (
        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        Chem.MolToInchiKey(mol),
    )


def _same_structure(left: Chem.Mol, right: Chem.Mol) -> bool:
    """Compare SDF structures without requiring identical SMILES traversal text."""

    has_dummy = any(
        atom.GetAtomicNum() == 0 for mol in (left, right) for atom in mol.GetAtoms()
    )
    if not has_dummy:
        return Chem.MolToInchiKey(left) == Chem.MolToInchiKey(right)
    aliases = lambda mol: tuple(
        atom.GetProp("molFileAlias") if atom.HasProp("molFileAlias") else ""
        for atom in mol.GetAtoms()
    )
    return (
        Chem.MolToSmiles(left, canonical=True, isomericSmiles=True)
        == Chem.MolToSmiles(right, canonical=True, isomericSmiles=True)
        and aliases(left) == aliases(right)
    )


def _read_molblock(block: str) -> Chem.Mol | None:
    return Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)


def _lossless_molblock(mol: Chem.Mol) -> str:
    """Write SDF coordinates while preserving the compiled molecular graph.

    Some source depictions place two neighbors of a stereocenter on overlapping
    rays. V2000 then cannot encode a unique wedge in those exact coordinates.
    The scene remains the visual source of truth; only the SDF conformer is
    regenerated when the first MolBlock fails its own structural roundtrip.
    """

    expected = Chem.Mol(mol)
    block = Chem.MolToMolBlock(mol)
    restored = _read_molblock(block)
    if restored is not None and _same_structure(restored, expected):
        return block

    relaid = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(relaid, canonOrient=True, clearConfs=True)
    block = Chem.MolToMolBlock(relaid)
    restored = _read_molblock(block)
    if restored is not None and _same_structure(restored, expected):
        return block
    raise ChemSceneSDFRoundtripError(
        "Cannot encode the compiled ChemScene graph as a lossless SDF record"
    )


def sdf_from_chemscene_v0_1(scene: dict[str, Any] | ParsedChemSceneV01) -> str:
    """Compile a scene to a single-record SDF string."""

    mol = mol_from_chemscene_v0_1(scene, apply_display_directions=False)
    return _lossless_molblock(mol) + "\n$$$$\n"


def validate_chemscene_v0_1_roundtrip(
    scene: dict[str, Any],
    expected_smiles: str,
    expected_inchikey: str,
) -> Chem.Mol:
    """Validate Scene -> ChemSVG -> RDKit -> SDF readback without byte comparison."""

    chemsvg = chemsvg_from_chemscene_v0_1(scene)
    mol = mol_from_parsed_chemsvg_v4(parse_chemsvg_v4(chemsvg))
    expected = Chem.MolFromSmiles(expected_smiles)
    if expected is None:
        raise ChemMolSequenceError("Expected canonical SMILES is invalid")
    direct_signature = _mol_signature(mol)
    if direct_signature[1] != expected_inchikey:
        raise ChemSceneDepictionConflictError(
            "ChemScene visual geometry contradicts the source stereo label: "
            f"{direct_signature[0]} / {direct_signature[1]}"
        )

    sdf = _lossless_molblock(mol)
    restored = _read_molblock(sdf)
    if restored is None:
        raise ChemMolSequenceError("RDKit cannot read the compiled SDF")
    actual_signature = _mol_signature(restored)
    if actual_signature[1] != expected_inchikey:
        raise ChemMolSequenceError(
            "ChemScene SDF roundtrip mismatch: "
            f"{actual_signature[0]} / {actual_signature[1]}"
        )
    return restored
