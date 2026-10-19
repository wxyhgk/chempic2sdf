"""Fixed, outcome-stratified case selection for a diagnostic, never an accuracy set."""

from collections import Counter
from hashlib import sha256

from rdkit import Chem

SEED = 20260915
VARIANTS = ("original", "rdkit", "coordgen", "stress")
PER_STYLE = (("broken_composition", 3), ("broken_other", 1),
             ("stable_correct", 1), ("stable_wrong", 1))
ARMS = ("initial", "last", "last_initial_role", "last_initial_bbox", "last_initial_role_bbox")


def atom_inventory(smiles):
    if not smiles:
        return None
    params = Chem.SmilesParserParams()
    params.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, params)
    if molecule is None:
        return None
    return Counter((a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope()) for a in molecule.GetAtoms())


def stratum(initial, last):
    if initial["exact"] and not last["exact"]:
        before, after = (atom_inventory(r["predicted_smiles"]) for r in (initial, last))
        if before is not None and after is not None and sum(before.values()) == sum(after.values()) and before != after:
            return "broken_composition"
        return "broken_other"
    if initial["exact"] and last["exact"]:
        return "stable_correct"
    if not initial["exact"] and not last["exact"]:
        return "stable_wrong"
    return "rescued_not_sampled"


def prediction_index(payload):
    records = payload["predictions"]
    index = {r["key"]: r for r in records}
    if not records or len(index) != len(records):
        raise ValueError("Empty/duplicate saved predictions")
    return index


def select_cases(manifest, initial_payload, last_payload):
    """Pick 24 distinct molecules using saved outcomes BEFORE running any new arm."""
    initial, last = map(prediction_index, (initial_payload, last_payload))
    validation = {r["source"] + ":" + r["sample_key"]: r for r in manifest["samples"] if r["split"] == "validation"}
    if initial.keys() != last.keys() or initial.keys() != validation.keys():
        raise ValueError("Saved predictions do not match the fixed validation set")
    pools = {}
    for key, row in validation.items():
        if any(record["domain"] != row["domain"] or record["expected_smiles"] != row["canonical_smiles"]
               for record in (initial[key], last[key])):
            raise ValueError("Saved validation identity/domain changed")
        if row["domain"] != "full_molecule":
            continue
        variant = row["variant"]
        if variant not in VARIANTS:
            raise ValueError("Unexpected full-molecule style")
        group = stratum(initial[key], last[key])
        pools.setdefault((group, variant), []).append({"key": key, "molecule_id": row["molecule_id"],
            "variant": variant, "stratum": group})
    selected, used = [], set()
    for group, count in PER_STYLE:
        for variant in VARIANTS:
            pool = sorted(pools.get((group, variant), []), key=lambda r: (
                sha256(f"{SEED}:{r['key']}".encode()).hexdigest(), r["key"]))
            available = [r for r in pool if r["molecule_id"] not in used][:count]
            if len(available) != count:
                raise ValueError(f"Insufficient distinct molecules for fixed stratum {group}/{variant}")
            selected.extend(available)
            used.update(r["molecule_id"] for r in available)
    return {"schema_version": "chemvector-text-interface-plan-0.1", "seed": SEED, "samples": selected,
        "arms": list(ARMS), "device": "cpu", "optimizer_steps": 0, "training_authorized": False,
        "selection": "saved initial/last outcomes; fixed per-style quotas; sha256(seed:key); distinct molecule IDs",
        "quotas_per_style": dict(PER_STYLE), "structure_for_swaps": "identical cached last predictions",
        "purpose": "development_diagnostic_only", "sealed_access": False,
        "limitations": ["Outcome-stratified cases cannot estimate validation or benchmark accuracy.",
            "This swaps inference modules; it does not test the effect of freezing them during training.",
            "Initial/last CPU replay is checked against saved GPU identities, validity and fallback."]}
