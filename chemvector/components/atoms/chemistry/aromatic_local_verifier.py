"""Bounded local aromatic demotion verifier.

This is an opt-in compiler-side diagnostic.  It never runs on a scene that
already compiles, never edits atoms or connectivity, and never claims that a
valid repaired molecule matches the image.  Every attempted demotion is
recorded so identity scoring can reject an intervention when necessary.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math

from Data.sdf.chemscene_v0_2 import mol_from_chemscene_v0_2


MAX_ATTEMPTS = 64


def _digest(scene: dict) -> str:
    return sha256(json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _compile(scene: dict, compiler):
    try:
        molecule = compiler(deepcopy(scene))
        if molecule is None:
            raise RuntimeError("Compiler returned no molecule")
        return molecule, None
    except (ValueError, RuntimeError) as error:
        return None, str(error)


def verify_aromatic_scene(
    scene: dict,
    aromatic_bond_scores: dict[int, float],
    *,
    compiler=mol_from_chemscene_v0_2,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[dict, object | None, dict]:
    """Compile a scene, then try deterministic one-edge aromatic demotions.

    The score map is keyed by scene bond id and should contain the predicted
    aromatic probability.  Lower-confidence aromatic bonds are tried first.
    A successful local repair is explicitly marked as an intervention; it is
    not equivalent to the unmodified model prediction.
    """
    if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_ATTEMPTS:
        raise ValueError("Local verifier attempt budget is invalid")
    original = deepcopy(scene)
    before = _digest(original)
    bonds = {int(item["id"]): item for item in original.get("bond_objects", [])}
    if len(bonds) != len(original.get("bond_objects", [])):
        raise ValueError("Scene bond ids must be unique")
    candidates = []
    for bond_id, score in aromatic_bond_scores.items():
        if type(bond_id) is not int or bond_id not in bonds:
            raise ValueError("Aromatic score references an unknown bond")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("Aromatic bond scores must be finite")
        bond = bonds[bond_id]
        if bond.get("chemical_type") == "AROMATIC":
            candidates.append((float(score), bond_id))
    candidates.sort()
    candidates = candidates[:max_attempts]
    molecule, error = _compile(original, compiler)
    journal = {
        "schema_version": "aromatic-local-verifier-0.1",
        "input_scene_sha256": before,
        "compiler_calls": 1,
        "baseline_valid": error is None,
        "baseline_error": error,
        "candidate_count": len(candidates),
        "attempts": [],
        "intervention": False,
        "identity_claim": "unscored_after_local_repair",
    }
    if error is None:
        journal.update(status="baseline_valid", output_scene_sha256=before)
        return original, molecule, journal
    for score, bond_id in candidates:
        candidate = deepcopy(original)
        bond = next(item for item in candidate["bond_objects"] if int(item["id"]) == bond_id)
        display = bond.get("display_type")
        replacement = display if display in {"SINGLE", "DOUBLE", "TRIPLE"} else "SINGLE"
        bond["chemical_type"] = replacement
        bond["display_type"] = replacement
        repaired_molecule, repair_error = _compile(candidate, compiler)
        journal["compiler_calls"] += 1
        attempt = {
            "bond_id": bond_id,
            "aromatic_probability": score,
            "replacement": replacement,
            "accepted": repair_error is None,
            "error": repair_error,
        }
        journal["attempts"].append(attempt)
        if repair_error is None:
            journal.update(
                status="local_repair_accepted",
                intervention=True,
                repaired_bond_id=bond_id,
                output_scene_sha256=_digest(candidate),
            )
            # Only chemical_type/display_type on the selected bond may differ.
            for old, new in zip(original["atom_objects"], candidate["atom_objects"], strict=True):
                if old != new:
                    raise ValueError("Local verifier changed atom fields")
            return candidate, repaired_molecule, journal
    journal.update(status="local_repair_failed", output_scene_sha256=before)
    return original, None, journal


__all__ = ["MAX_ATTEMPTS", "verify_aromatic_scene"]
