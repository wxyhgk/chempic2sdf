"""Load the current self-developed components, with local provenance checks."""

from hashlib import sha256
from dataclasses import fields
import json
from pathlib import Path

import torch

from chemvector.direct_scene_inference import load_direct_model
from chemvector.evaluate_direct_atom_ocr import load_bbox, load_role
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.model import ChemVectorConfig
from chemvector.train_bond_presence_calibration import load_components
from .continuation import ContinuationComponents

PROFILE = "structure-v01010-direct-aromatic-ocr"


def file_digest(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(records):
    for record in records.values():
        if file_digest(record["path"]) != record["sha256"]:
            raise ValueError(f"Checkpoint changed: {record['path']}")


def load_current_components(registry=Path("frontend/inference_profiles.json")):
    """CPU only. Never restores an optimizer or loads OCR/evaluation records."""
    registry = Path(registry)
    contents = registry.read_bytes()
    profiles = json.loads(contents)
    profile = profiles.get("profiles", profiles)[PROFILE]
    arguments = profile["arguments"]
    root = Path("outputs/checkpoints").resolve()
    records = {}

    def record(name, path, expected=None):
        path = Path(path).resolve(strict=True)
        if not path.is_relative_to(root) or path.suffix != ".pt":
            raise ValueError("Continuation sources must be local checkpoint files")
        actual = file_digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Registered checkpoint hash mismatch: {name}")
        records[name] = {"path": str(path), "sha256": actual}
        return str(path)

    for name, flag, digest_field in (
        ("pair", "--checkpoint", "checkpoint_sha256"),
        ("role", "--role-checkpoint", "role_checkpoint_sha256"),
        ("bbox", "--bbox-checkpoint", "bbox_checkpoint_sha256"),
        ("ocr_frozen_not_loaded", "--ocr-checkpoint", "ocr_checkpoint_sha256"),
    ):
        record(name, arguments[arguments.index(flag) + 1], profile[digest_field])

    def payload(name):
        return torch.load(records[name]["path"], map_location="cpu", weights_only=False, mmap=True)

    direct_payload = payload("pair")
    args = direct_payload["training_args"]
    record("source", args["source_checkpoint"])
    record("graph", args["grounded_graph_checkpoint"])
    record("selector", payload("graph")["grounded_selector_checkpoint"])
    record("query", payload("selector")["query_checkpoint"])
    record("base", payload("query")["base_checkpoint"])
    if Path(payload("role")["training_args"]["source_checkpoint"]).resolve() != Path(records["source"]["path"]):
        raise ValueError("Role and pair checkpoints have different structure sources")
    if Path(payload("bbox")["training_args"]["role_checkpoint"]).resolve() != Path(records["role"]["path"]):
        raise ValueError("BBox and role checkpoints disagree")
    device = torch.device("cpu")
    # Reuse the production loader, including query-checkpoint vision updates.
    # No newly initialized head is substituted for a missing old checkpoint.
    vision, atom, bond, set_head, source = load_components(
        records["source"]["path"], records["graph"]["path"], device)
    pair, _ = load_direct_model(records["pair"]["path"], device)
    role, _ = load_role(records["role"]["path"], device)
    bbox, _ = load_bbox(records["bbox"]["path"], device)
    model = ContinuationComponents(vision=vision, atom=atom, bond=bond, set_head=set_head,
                                   pair=pair, role=role, bbox=bbox)
    valid = {f.name for f in fields(ChemVectorConfig)}
    model.vision_config = ChemVectorConfig(**{k: v for k, v in payload("base")["config"].items() if k in valid})
    verify_sources(records)
    return model, PredictedProposalConfig(**source["proposal_config"]), {
        "profile": PROFILE, "registry_sha256": sha256(contents).hexdigest(), "checkpoints": records,
        "atom_threshold": float(source["training_args"]["atom_decision_threshold"]),
        "bond_threshold": float(source["training_args"]["bond_decision_threshold"]),
        "pair_threshold": float(direct_payload["validation"]["threshold"]),
        "role_threshold": float(payload("role")["validation"]["threshold"]),
        "ocr_loaded": False,
    }
