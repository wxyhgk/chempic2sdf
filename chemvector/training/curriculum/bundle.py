"""Self-contained structure weights: never fall back to old upstream weights."""

from dataclasses import asdict
import os
from pathlib import Path
import tempfile

import torch

from chemvector.model import ChemVectorConfig, ChemVisionEncoder
from chemvector.spatial_atom_object import SpatialAtomObjectConfig, SpatialAtomObjectDetector
from chemvector.dense_bond_object import DenseBondObjectConfig, MultiScaleDenseBondObjectDetector
from chemvector.proposal_object_set import ProposalObjectSetConfig, ProposalObjectSetHead
from chemvector.spatial_incidence import SpatialIncidenceConfig
from chemvector.direct_atom_pair_bond import DirectAtomPairBond, DirectAtomPairBondConfig
from chemvector.direct_atom_role import DirectAtomRoleGate, DirectAtomRoleConfig
from chemvector.direct_atom_text_bbox import DirectAtomTextBBox, DirectAtomTextBBoxConfig
from chemvector.predicted_incidence import PredictedProposalConfig
from .checkpoints import file_digest
from .continuation import ContinuationComponents

SCHEMA = "chemvector-continuation-bundle-0.1"


def save_bundle(path, model, proposal_config, *, provenance, progress):
    """Exclusive atomic publication; partial writes are never a valid checkpoint."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Continuation bundles cannot overwrite existing paths")
    if not hasattr(model, "vision_config"):
        raise ValueError("Explicit original vision config is required for a portable bundle")
    recipe = model.update_recipe
    if progress.get("recipe", {}).get("update_recipe", recipe) != recipe:
        raise ValueError("Bundle progress and model update recipe disagree")
    architecture = {name: asdict(getattr(model, name).config)
                    for name in ("atom", "bond", "set_head", "pair", "role", "bbox")}
    architecture["vision"] = asdict(model.vision_config)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if any(v.is_floating_point() and not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError("Cannot publish non-finite model weights")
    payload = {"schema_version": SCHEMA, "architecture": architecture, "model": state,
               "update_recipe": recipe,
               "proposal_config": asdict(proposal_config), "provenance": provenance, "progress": progress,
               "resume_supported": False, "ocr": "external frozen checkpoint pinned in provenance"}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".bundle-partial-", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Fails if target appeared concurrently; no overwrite.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(path), "sha256": file_digest(path)}


def load_bundle(path, *, expected_sha256):
    """CPU restoration from this file alone; OCR is intentionally separate."""
    if not expected_sha256 or file_digest(path) != expected_sha256:
        raise ValueError("Continuation bundle hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported continuation bundle schema")
    a = payload["architecture"]
    vision_config = ChemVectorConfig(**a["vision"])
    set_config = dict(a["set_head"])
    set_config["incidence"] = SpatialIncidenceConfig(**set_config["incidence"])
    model = ContinuationComponents(
        vision=ChemVisionEncoder(vision_config),
        atom=SpatialAtomObjectDetector(SpatialAtomObjectConfig(**a["atom"])),
        bond=MultiScaleDenseBondObjectDetector(DenseBondObjectConfig(**a["bond"]),
            stride8_channels=vision_config.vision_dims[1], stride16_channels=vision_config.vision_dims[2]),
        set_head=ProposalObjectSetHead(ProposalObjectSetConfig(**set_config)),
        pair=DirectAtomPairBond(DirectAtomPairBondConfig(**a["pair"])),
        role=DirectAtomRoleGate(DirectAtomRoleConfig(**a["role"])),
        bbox=DirectAtomTextBBox(DirectAtomTextBBoxConfig(**a["bbox"])),
    )
    model.vision_config = vision_config
    model.load_state_dict(payload["model"], strict=True)
    recorded_recipe = payload.get("progress", {}).get("recipe", {}).get("update_recipe")
    recipe = payload.get("update_recipe", recorded_recipe or "legacy-all-v1")
    if recorded_recipe is not None and recipe != recorded_recipe:
        raise ValueError("Bundle progress and model update recipe disagree")
    model.configure_update_recipe(recipe)
    if any(v.is_floating_point() and not bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
        raise ValueError("Non-finite continuation bundle weights")
    model.eval()
    return model, PredictedProposalConfig(**payload["proposal_config"]), payload
