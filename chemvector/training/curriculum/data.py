"""Adapt same-render teaching data to existing ChemVector target tensors.

No new model, OCR training, legacy relabeling, or coordinate-to-identity lookup.
The quarantine pilot can only be read explicitly for diagnostics.
"""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re

from PIL import Image
import torch

from Data.sdf.chemscene_v0_2 import _clip_port_to_text_bbox
from chemvector.atom_bond_scene import ATOM_ROLES, AtomBondSceneConfig
from chemvector.bond_object import BOND_TYPE_TO_ID, DIRECTION_TO_ID
from .pixels import image_tensor

QMAX = 1023.0
SAFE_KEY = re.compile(r"[A-Za-z0-9_-]+\Z")
PAYLOADS = (".graph.json", ".layers.json", ".text-records.json", ".png", ".svg", ".sdf")


def _index(records, field):
    result = {}
    for record in records:
        key = record[field]
        if type(key) is not int or key < 0 or key in result:
            raise ValueError(f"Invalid or duplicate {field}")
        result[key] = record
    return result


def _point(point, *, limit=767):
    if len(point) != 2 or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                              or not math.isfinite(v) or not 0 <= v <= limit for v in point):
        raise ValueError("Invalid curriculum pixel coordinate")
    return tuple(round(v / 767 * QMAX) for v in point)


def validate_records(graph, layers, text_records):
    if (graph.get("schema_version") != "curriculum-chemical-graph-0.1"
            or layers.get("schema_version") != "chemvision-text-source-layers-0.2"
            or text_records.get("schema_version") != "same-svg-text-records-0.2"):
        raise ValueError("Unsupported curriculum schemas")
    if (layers.get("width"), layers.get("height")) != (768, 768):
        raise ValueError("Curriculum target adapter requires original 768px inputs")
    if (not text_records.get("same_source_raster")
            or text_records.get("evidence_kind") != "same_render_text_record"
            or layers.get("text_supervision") != "same_svg_visible_characters"):
        raise ValueError("Same-source visible-text evidence is required")
    if graph["molecule_id"] != layers["molecule_id"]:
        raise ValueError("Graph/layers molecule identity mismatch")
    atoms, bonds = _index(graph["atoms"], "atom_id"), _index(graph["bonds"], "bond_id")
    visual_atoms, visual_bonds = _index(layers["atoms"], "atom_id"), _index(layers["bonds"], "bond_id")
    if not atoms or atoms.keys() != visual_atoms.keys() or bonds.keys() != visual_bonds.keys():
        raise ValueError("Graph/layers object identities disagree")
    glyphs = {}
    for glyph in text_records["glyphs"]:
        if glyph["atom_id"] not in atoms or not glyph.get("text"):
            raise ValueError("Invalid glyph atom reference or empty transcription")
        glyphs.setdefault(glyph["atom_id"], []).append(glyph)
    for identity, atom in atoms.items():
        visual = visual_atoms[identity]
        if any(visual.get(k) != v for k, v in atom.items()):
            raise ValueError("Graph/layers atom chemistry mismatch")
        _point(visual["center"])
        records = sorted(glyphs.get(identity, []), key=lambda g: g["node_index"])
        text = "".join(g["text"] for g in records)
        if visual["text"] != text or visual["visible_text"] != bool(records):
            raise ValueError("Visible text disagrees with same-render glyph records")
        if records:
            boxes = [g["bbox"] for g in records]
            expected = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                        max(b[2] for b in boxes), max(b[3] for b in boxes)]
            if visual["bbox"] != expected:
                raise ValueError("Atom text bbox is not the union of glyph ink")
            x0, y0, x1, y1 = expected
            _point([x0, y0]); _point([x1, y1])
            if not x0 < x1 or not y0 < y1:
                raise ValueError("Invalid text bbox")
        elif visual["bbox"] is not None:
            raise ValueError("Invisible atom must not have a text bbox")
    seen_pairs = set()
    for identity, bond in bonds.items():
        visual = visual_bonds[identity]
        if any(visual.get(k) != v for k, v in bond.items()):
            raise ValueError("Graph/layers bond chemistry mismatch")
        endpoints = {bond["begin_atom"], bond["end_atom"]}
        pair = frozenset(endpoints)
        if len(endpoints) != 2 or not endpoints <= atoms.keys() or pair in seen_pairs:
            raise ValueError("Invalid or duplicate chemical edge")
        seen_pairs.add(pair)
        drawing = visual["drawing"]
        if {drawing["begin_atom"], drawing["end_atom"]} != endpoints:
            raise ValueError("Drawing edge disagrees with semantic endpoints")
        if (bond["chemical_type"] not in BOND_TYPE_TO_ID
                or drawing["bond_type"] not in BOND_TYPE_TO_ID
                or drawing["direction"] not in DIRECTION_TO_ID):
            raise ValueError("Unsupported existing ChemVector bond target class")
        for endpoint, field in ((bond["begin_atom"], "begin"), (bond["end_atom"], "end")):
            if list(visual[field]) != list(visual_atoms[endpoint]["center"]):
                raise ValueError("Bond anchor evidence disagrees with atom position")
    return atoms, bonds, visual_atoms, visual_bonds


def load_sample(root, key, *, purpose="training"):
    if purpose not in {"training", "diagnostic"}:
        raise ValueError("Unknown curriculum read purpose")
    if not isinstance(key, str) or not SAFE_KEY.fullmatch(key):
        raise ValueError("Unsafe curriculum sample key")
    directory = Path(root) / "samples"
    if directory.is_symlink():
        raise ValueError("Symlinked samples directory is not allowed")

    def read(suffix):
        path = directory / (key + suffix)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symlinked curriculum artifact: {suffix}")
        return path.read_bytes()

    meta = json.loads(read(".meta.json"))
    if meta.get("sample_key") != key:
        raise ValueError("Sample key does not match metadata")
    if purpose == "training" and (meta.get("training_authorized") is not True
            or meta.get("split") != "train" or meta.get("external_overlap_status") != "checked"):
        raise ValueError("Curriculum is not admitted for training; quarantine is diagnostic-only")
    payload = {suffix: read(suffix) for suffix in PAYLOADS}
    for suffix, data in payload.items():
        if sha256(data).hexdigest() != meta["sha256"].get(suffix):
            raise ValueError(f"Curriculum artifact hash mismatch: {suffix}")
    graph, layers, records = [json.loads(payload[s]) for s in PAYLOADS[:3]]
    if (graph["molecule_id"] != meta["molecule_id"] or layers["sample_key"] != key
            or layers["profile_id"] != meta["profile_id"]):
        raise ValueError("Curriculum bundle identity mismatch")
    if (layers.get("split") != meta.get("split")
            or layers.get("training_authorized") != meta.get("training_authorized")
            or graph.get("training_authorized") != meta.get("training_authorized")):
        raise ValueError("Curriculum admission flags disagree")
    for record in (layers, records):
        if (record["source_png_sha256"] != sha256(payload[".png"]).hexdigest()
                or record["source_svg_sha256"] != sha256(payload[".svg"]).hexdigest()):
            raise ValueError("Same-source pixel evidence hashes disagree")
    if layers["text_records_sha256"] != sha256(payload[".text-records.json"]).hexdigest():
        raise ValueError("Text-record binding mismatch")
    validate_records(graph, layers, records)
    with Image.open(io.BytesIO(payload[".png"])) as image:
        if image.size != (768, 768) or image.mode != "RGB":
            raise ValueError("Original RGB PNG required; no implicit resizing")
        image = image.copy()
    return {"key": key, "image": image, "graph": graph, "layers": layers,
            "text_records": records, "meta": meta, "purpose": purpose}


class CurriculumCollator:
    """Existing tensor names/classes; semantic IDs survive coordinate collisions.

    Ports follow the existing q1023 text-bbox-ray convention. They are derived
    supervision, not claimed to be exact SVG stroke endpoints. No semantic mask
    or OCR token loss is introduced. Chemistry remains in ``chemical_graphs``.
    """

    def __init__(self, config=AtomBondSceneConfig(), *, purpose="training"):
        if purpose not in {"training", "diagnostic"}:
            raise ValueError("Unknown curriculum collator purpose")
        self.config, self.purpose = config, purpose

    def __call__(self, samples):
        if not samples:
            raise ValueError("Empty curriculum batch")
        batch, na, nb = len(samples), self.config.atom_queries, self.config.bond_queries
        result = {"atom_anchors": torch.zeros(batch, na, 2),
                  "atom_roles": torch.zeros(batch, na, dtype=torch.long),
                  "atom_text_center": torch.zeros(batch, na, 2),
                  "atom_text_bbox": torch.zeros(batch, na, 4),
                  "atom_text_mask": torch.zeros(batch, na, dtype=torch.bool),
                  "atom_mask": torch.zeros(batch, na, dtype=torch.bool),
                  "bond_ports": torch.zeros(batch, nb, 2, 2),
                  "bond_display": torch.zeros(batch, nb, dtype=torch.long),
                  "bond_chemical": torch.zeros(batch, nb, dtype=torch.long),
                  "bond_direction": torch.zeros(batch, nb, dtype=torch.long),
                  "bond_atom_refs": torch.zeros(batch, nb, 2, dtype=torch.long),
                  "bond_mask": torch.zeros(batch, nb, dtype=torch.bool),
                  "keys": [], "atom_ids": [], "bond_ids": [], "visible_text": [],
                  "chemical_graphs": [], "spatial_cell_collision_pairs": []}
        pixels = []
        for row, sample in enumerate(samples):
            if sample.get("purpose") != self.purpose:
                raise ValueError("Diagnostic curriculum samples cannot enter a training batch")
            meta = sample["meta"]
            if self.purpose == "training" and (meta.get("training_authorized") is not True
                    or meta.get("split") != "train" or meta.get("external_overlap_status") != "checked"):
                raise ValueError("Training admission is required")
            atoms, bonds, visual_atoms, visual_bonds = validate_records(
                sample["graph"], sample["layers"], sample["text_records"])
            if len(atoms) > na or len(bonds) > nb:
                raise ValueError("Curriculum graph exceeds existing query capacity")
            atom_ids, bond_ids = sorted(atoms), sorted(bonds)
            slots = {identity: index for index, identity in enumerate(atom_ids)}
            anchors, boxes, strings = {}, {}, []
            for identity in atom_ids:
                slot, visual = slots[identity], visual_atoms[identity]
                anchor = _point(visual["center"])
                anchors[identity] = anchor
                box = visual["bbox"]
                boxes[identity] = None if box is None else (*_point(box[:2]), *_point(box[2:]))
                result["atom_anchors"][row, slot] = torch.tensor(anchor) / QMAX
                result["atom_mask"][row, slot] = True
                result["atom_roles"][row, slot] = ATOM_ROLES.index("TEXT" if visual["visible_text"] else "IMPLICIT")
                strings.append(visual["text"])
                if box is not None:
                    x0, y0, x1, y1 = boxes[identity]
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    result["atom_text_center"][row, slot] = torch.tensor((cx, cy)) / QMAX
                    result["atom_text_bbox"][row, slot] = torch.tensor((cx, cy, x1 - x0, y1 - y0)) / QMAX
                    result["atom_text_mask"][row, slot] = True
            for slot, identity in enumerate(bond_ids):
                bond, drawing = bonds[identity], visual_bonds[identity]["drawing"]
                begin, end = drawing["begin_atom"], drawing["end_atom"]
                ports = [_clip_port_to_text_bbox(anchors[a], anchors[b], boxes[a])[0]
                         for a, b in ((begin, end), (end, begin))]
                result["bond_ports"][row, slot] = torch.tensor(ports) / QMAX
                result["bond_atom_refs"][row, slot] = torch.tensor((slots[begin], slots[end]))
                result["bond_chemical"][row, slot] = BOND_TYPE_TO_ID[bond["chemical_type"]]
                result["bond_display"][row, slot] = BOND_TYPE_TO_ID[drawing["bond_type"]]
                result["bond_direction"][row, slot] = DIRECTION_TO_ID[drawing["direction"]]
                result["bond_mask"][row, slot] = True
            cells = Counter(tuple(min(191, int(v / QMAX * 192)) for v in xy) for xy in anchors.values())
            result["spatial_cell_collision_pairs"].append(sum(n * (n - 1) // 2 for n in cells.values()))
            result["keys"].append(sample["key"])
            result["atom_ids"].append(atom_ids); result["bond_ids"].append(bond_ids)
            result["visible_text"].append(strings)
            result["chemical_graphs"].append(deepcopy(sample["graph"]))
            pixels.append(image_tensor(sample["image"]))
        result.update(pixel_values=torch.stack(pixels),
                      target_schema="chemvector-curriculum-targets-0.1", purpose=self.purpose,
                      training_authorized=self.purpose == "training")
        return result
