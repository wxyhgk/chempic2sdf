"""Opt-in, train-only label revisions backed by explicit visible-text evidence.

This module never infers a transcription from chemical attributes. It does not
read/write shards or replace the legacy generator. Evidence must refer to the
actual source raster, not to a different renderer's diagnostic replay.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math


REVISION_SCHEMA = "chemvision-layers-visible-text-0.1"
LEGACY_SCHEMAS = {"chemvision-layers-0.1", "chemvision-layers-0.2"}
EVIDENCE_KINDS = {"reviewed_source_raster", "same_render_text_record"}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class VisibleTextCorrection:
    atom_id: int
    expected_text: str
    visible_text: str
    evidence_kind: str
    evidence_reference: str


@dataclass(frozen=True)
class VisibleTextRevision:
    sample_key: str
    source_layers_sha256: str
    source_png_sha256: str
    corrections: tuple[VisibleTextCorrection, ...]


def revise_training_visible_text(
    layers_payload: bytes,
    png_payload: bytes,
    revision: VisibleTextRevision,
) -> dict:
    """Return a new versioned layer document, preserving geometry and chemistry.

    Hashes prevent applying a reviewed correction to another image/version.
    They bind the supplied evidence claim; they cannot themselves prove that
    a human or renderer transcribed the raster correctly. Unreviewed labels
    remain explicitly unverified, even in the revised document.
    """
    if sha256(layers_payload) != revision.source_layers_sha256:
        raise ValueError("Source layer bytes differ from reviewed revision")
    if sha256(png_payload) != revision.source_png_sha256:
        raise ValueError("Source PNG bytes differ from reviewed revision")
    layers = json.loads(layers_payload)
    if not isinstance(layers, dict):
        raise ValueError("Source layers must be a JSON object")
    if not isinstance(revision.sample_key, str) or not revision.sample_key.strip():
        raise ValueError("Revision needs a nonempty sample identity")
    if layers.get("schema_version") not in LEGACY_SCHEMAS:
        raise ValueError("Expected an unrevised legacy layer schema")
    if layers.get("split") != "train":
        raise ValueError("Only train labels may be revised; evaluation labels are frozen")
    if layers.get("sample_key") != revision.sample_key:
        raise ValueError("Sample identity differs from reviewed revision")
    if "visible_text_revision" in layers:
        raise ValueError("Existing revision provenance cannot be overwritten")
    if not revision.corrections:
        raise ValueError("An explicit, nonempty correction list is required")

    if not isinstance(layers.get("atoms"), list):
        raise ValueError("Layer atoms must be a list")
    atoms = {}
    for atom in layers["atoms"]:
        if not isinstance(atom, dict):
            raise ValueError("Layer atoms must be objects")
        identity = atom.get("atom_id")
        if type(identity) is not int or identity in atoms:
            raise ValueError("Layer atom identities must be unique integers")
        atoms[identity] = atom
    corrected_ids = set()
    for correction in revision.corrections:
        identity = correction.atom_id
        if type(identity) is not int or identity not in atoms or identity in corrected_ids:
            raise ValueError("Correction identity is missing, invalid, or duplicated")
        corrected_ids.add(identity)
        if correction.evidence_kind not in EVIDENCE_KINDS:
            raise ValueError("Correction needs same-render or reviewed-source-raster evidence")
        if not isinstance(correction.evidence_reference, str) or not correction.evidence_reference.strip():
            raise ValueError("Correction needs an explicit evidence reference")
        if not isinstance(correction.visible_text, str) or not correction.visible_text.strip():
            raise ValueError("This revision only changes nonempty visible transcriptions")
        if not isinstance(correction.expected_text, str) or not correction.expected_text.strip():
            raise ValueError("Correction needs an explicit nonempty old transcription")
        atom = atoms[identity]
        if atom.get("visible_text") is not True:
            raise ValueError("Cannot invent visibility or turn implicit atoms into text")
        box = atom.get("bbox")
        if (not isinstance(box, list) or len(box) != 4
                or not all(type(v) in (float, int) and math.isfinite(v) for v in box)
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("Reviewed visible atom requires a valid existing bbox")
        if atom.get("text") != correction.expected_text:
            raise ValueError("Expected transcription differs; correction is stale")
        if correction.visible_text == correction.expected_text:
            raise ValueError("A correction must change the transcription")

    revised = deepcopy(layers)
    changes = {item.atom_id: item.visible_text for item in revision.corrections}
    for atom in revised["atoms"]:
        if atom["atom_id"] in changes:
            atom["text"] = changes[atom["atom_id"]]
    revised["schema_version"] = REVISION_SCHEMA
    revised["visible_text_revision"] = {
        "source_schema_version": layers["schema_version"],
        "sample_key": revision.sample_key,
        "source_layers_sha256": revision.source_layers_sha256,
        "source_png_sha256": revision.source_png_sha256,
        "corrections": [asdict(item) for item in sorted(revision.corrections, key=lambda item: item.atom_id)],
        "scope": "explicitly_corrected_atoms_only",
        "other_text_labels": "legacy_unverified",
        "chemistry_modified": False,
        "geometry_modified": False,
    }
    return revised
