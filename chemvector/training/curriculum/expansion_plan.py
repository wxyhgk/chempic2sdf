"""CPU train-candidate expansion; known development groups protected before rendering.

Historical public inventories are conservative prefilters, not new clearance.
No sealed manifests, private-review service, model, optimizer or image is opened.
"""

from collections import Counter
from dataclasses import asdict

from rdkit import Chem

from Data.sdf.chemvector_curriculum_v0_1.rendering import profiles
from .admission import _cached_keys, identities
from .ctab_inventory import deficiency_witness, molecular_inventory
from .full_molecule_data import select_molecules, size_bucket

PROFILE_IDS = ("reference", "font_small", "font_large", "bond_short", "line_thick", "serif")
SEED = 20260915
MOLECULES = 2000
MINIMUM_IMAGES = 10000


def selected_profiles():
    by_id = {p.profile_id: p for p in profiles()}
    return [by_id[name] for name in PROFILE_IDS]


def chemistry_preflight(entries):
    """Exclude an entire unsupported export, never repair its chemical identity.

    These are recorded chemistry-contract failures, NOT visual quality rejects.
    Unknown implementation errors still stop generation.
    """
    from Data.sdf.chemvector_curriculum_v0_1.rendering import make_molecule, sdf_bytes
    from Data.sdf.text_source_renderer.v2.renderer import prepare_molecule

    supported, failures = [], []
    known = {"SDF roundtrip changed chemical identity", "SDF roundtrip changed atom index identity",
             "SDF roundtrip changed bond index identity"}
    for entry in entries:
        try:
            molecule = make_molecule(entry["canonical_smiles"])
            sdf_bytes(molecule)
            prepare_molecule(molecule)
        except ValueError as error:
            if str(error) not in known:
                raise
            failures.append({"molecule_id": entry["molecule_id"], "canonical_smiles": entry["canonical_smiles"],
                "reason": str(error), "excluded_all_profiles": list(PROFILE_IDS),
                "chemical_labels_modified": False, "training_authorized": False})
        else:
            supported.append(entry)
    return supported, failures


def validation_protection(manifest):
    """Use the already released validation rows; never redivide or open sealed data."""
    records = [r for r in manifest["samples"] if r["split"] == "validation"]
    if not records:
        raise ValueError("Existing validation protection is required")
    result = {"molecule_ids": set(), "inchikeys": set(), "connectivities": set(), "groups": set()}
    for row in records:
        ids = identities(row["canonical_smiles"])
        if row["group_key"] != ids["group_key"]:
            raise ValueError("Validation molecular group mismatch")
        result["molecule_ids"].add(row["molecule_id"])
        result["inchikeys"].add(ids["inchikey"])
        result["connectivities"].add(ids["connectivity_prefix"])
        result["groups"].add(ids["group_key"])
    return result


def prefilter(molecules, manifest, index, public_records, unresolved):
    protected = validation_protection(manifest)
    keys, _ = _cached_keys(index, "benchmark_inchikeys")
    if not keys:
        raise ValueError("Missing public exclusion keys")
    public_keys = {r["inchikey"] for r in public_records if r["inchikey"]}
    native = {r["native_probe"]["key"] for r in unresolved if r["native_probe"]["key"]}
    prefixes = {k[:14] for k in keys | public_keys | native}
    kept, excluded = [], []
    for row in molecules:
        if row["split"] != "train":
            raise ValueError("Expansion requires train-only source identities")
        reason, witnesses = None, []
        if row["inchikey"][:14] in prefixes:
            reason = "cached_public_key_or_connectivity"
        elif size_bucket(int(row["heavy_atoms"])) is None:
            reason = "outside_fixed_8_to_60_heavy_atom_range"
        else:
            ids = identities(row["canonical_isomeric_smiles"])
            if ids["inchikey"] != row["inchikey"]:
                raise ValueError("Train source chemistry/identity mismatch")
            if (row["molecule_id"] in protected["molecule_ids"] or ids["inchikey"] in protected["inchikeys"]
                    or ids["connectivity_prefix"] in protected["connectivities"] or ids["group_key"] in protected["groups"]):
                reason = "existing_validation_identity_connectivity_or_core"
            else:
                inventory = molecular_inventory(Chem.MolFromSmiles(row["canonical_isomeric_smiles"]))
                witnesses = [r["sample_id"] for r in unresolved if deficiency_witness(inventory, r["inventory"]) is None]
                if witnesses:
                    reason = "possible_overlap_with_cached_unresolved_public_record"
        if reason:
            excluded.append({"molecule_id": row["molecule_id"], "reason": reason, "public_record_ids": witnesses})
        else:
            kept.append(row)
    return kept, excluded, protected


def make_plan(molecules, manifest, index, public_records, unresolved, *, count=MOLECULES):
    kept, excluded, protected = prefilter(molecules, manifest, index, public_records, unresolved)
    selected, selection = select_molecules(kept, index, count=count, seed=SEED)
    train_smiles = {r["canonical_smiles"] for r in manifest["samples"] if r["split"] == "train"}
    return {"schema_version": "chemvector-expansion-plan-0.1", "seed": SEED,
        "purpose": "quarantined_train_replay_candidates_not_unseen_validation",
        "selected": selected, "selection": selection, "prefilter_excluded": excluded,
        "prefilter_counts": dict(Counter(r["reason"] for r in excluded)),
        "protected_validation_counts": {k: len(v) for k, v in protected.items()},
        "selected_known_validation_intersections": 0,
        "selected_already_in_previous_train": sum(r["canonical_isomeric_smiles"] in train_smiles for r in selected),
        "profiles": [asdict(p) for p in selected_profiles()], "typography": "stereo-aware-v1",
        "planned_images": len(selected) * len(PROFILE_IDS), "minimum_qualified_image_target": MINIMUM_IMAGES,
        "training_authorized": False, "private_identity_review": "new_fixed_candidates_require_new_authorization_and_receipt",
        "public_review": "cached_record_prefilter_only_not_new_candidate_clearance",
        "gpu_used": False, "training_performed": False, "validation_changed": False,
        "limitations": ["Two font families and one RDKit layout engine; six profiles are not six independent renderers.",
            "Existing training identities are new renders, not necessarily unseen molecules to the historical model.",
            "Minimum image target applies to a future admitted train split, not planned/rejected/duplicate or validation images.",
            "Role-head regression is unresolved by adding more images; no old training recipe is relaunched.",
            "Historical private review and source-use approval do not automatically extend to these candidates."]}
