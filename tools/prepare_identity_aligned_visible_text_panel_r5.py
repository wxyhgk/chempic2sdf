"""Prepare a fresh O/N panel with the measured 48 px minimum bond spacing.

This wrapper reuses the reviewed r1 planner but overrides only the output,
seed, and renderer profiles.  It creates quarantine metadata and a smoke
render; no model, sealed test data, GPU, or training is touched.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import prepare_identity_aligned_visible_text_panel as base  # noqa: E402


base.OUT = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5"
base.SEED = "chemvector-identity-aligned-visible-text-20261017-r5:"
_old_candidates = base.candidates


def _connection_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("candidate SMILES is not parseable")
    Chem.RemoveStereochemistry(mol)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return "conn-v1-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


_blocked_path = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r4/rejected.jsonl"
_blocked_ids = {
    row["molecule_id"]
    for line in _blocked_path.read_text(encoding="utf-8").splitlines()
    if line
    for row in (json.loads(line),)
}


def candidates():
    rows = _old_candidates()
    result = []
    for row in rows:
        if row["molecule_id"] in _blocked_ids:
            continue
        row = dict(row)
        row["connection_identity"] = _connection_key(row["canonical_smiles"])
        result.append(row)
    return result


base.candidates = candidates
# RDKit's measured anchor length is 1.5x fixedBondLength.  A request of 32px
# therefore yields the required 48px minimum for the short-bond profiles.
base.PROFILES = (
    replace(base.PROFILES[0], profile_id="cov13_base48", bond_length_cap=32.0),
    replace(base.PROFILES[1], profile_id="cov13_cap52", bond_length_cap=52.0),
    replace(base.PROFILES[2], profile_id="cov13_stroke16_48", bond_length_cap=32.0),
    replace(base.PROFILES[3], profile_id="cov33_base", bond_length_cap=52.0, font_family="Nimbus Roman"),
    replace(base.PROFILES[4], profile_id="cov33_cap32", bond_length_cap=32.0, font_family="Nimbus Roman"),
    replace(base.PROFILES[5], profile_id="cov33_stroke32", bond_length_cap=52.0, font_family="Nimbus Roman"),
)


if __name__ == "__main__":
    base.run()
