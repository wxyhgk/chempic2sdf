"""Pure design and paired summaries for the fixed P--O style diagnostic."""

from dataclasses import asdict
from itertools import product
import statistics

from rdkit import Chem

from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile

FACTORS = ("font_family", "font_size", "bond_length_cap", "bond_line_width")
LEVELS = (("DejaVu Sans", "Liberation Sans"), (12, 24), (28., 40.), (.8, 3.2))


def profiles():
    result = []
    for bits in product((0, 1), repeat=4):
        pid = "oa00" if bits == (0,)*4 else "oa22" if bits == (1,)*4 else "po" + "".join(map(str,bits))
        profile = Profile(profile_id=pid, **{name:levels[bit] for name,levels,bit in zip(FACTORS,LEVELS,bits)})
        result.append(dict(bits=list(bits), profile=asdict(profile)))
    return result


def select_entries(prior_samples, registry, remaining):
    by_id = {row["molecule_id"]:row for row in registry}
    allowed = {row["molecule_id"]:row for row in remaining}
    if len(by_id) != len(registry) or len(allowed) != len(remaining):
        raise ValueError("Duplicate source registry")
    selected = {}
    for row in prior_samples:
        mid = row["molecule_id"]
        molecule = Chem.MolFromSmiles(row["canonical_smiles"])
        if molecule is None:
            raise ValueError("Invalid prior diagnostic identity")
        singles = [b for b in molecule.GetBonds() if str(b.GetBondType()) == "SINGLE"
                   and {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()} == {"P", "O"}]
        if not singles:
            continue
        if (mid not in by_id or mid not in allowed or len(singles) != 2
                or row["canonical_smiles"] != by_id[mid]["canonical_smiles"]
                or row["canonical_smiles"] != allowed[mid]["canonical_smiles"]
                or row["sample_key"] not in allowed[mid]["sample_keys"]
                or by_id[mid]["split"] != "quarantine" or by_id[mid]["training_authorized"] is not False):
            raise ValueError("P--O cohort is not an unchanged screened prior identity")
        selected[mid] = by_id[mid]
    if len(selected) != 4:
        raise ValueError("Exactly the prior four P--O structures are required")
    return [selected[mid] for mid in sorted(selected)]


def summarize(cases):
    by_cell = {(c["molecule_id"], tuple(c["bits"])):c for c in cases}
    mids = sorted({c["molecule_id"] for c in cases})
    if (len(cases) != len(by_cell) or len(mids) != 4
            or set(by_cell) != {(mid,bits) for mid in mids for bits in product((0,1),repeat=4)}):
        raise ValueError("Incomplete/duplicate factorial; do not impute missing cells")
    records = []
    for c in cases:
        focus = c["po_bonds"]
        if (len(focus) != 3 or len({b["bond_id"] for b in focus}) != 3
                or sorted(b["expected_type"] for b in focus) != ["DOUBLE", "SINGLE", "SINGLE"]):
            raise ValueError("Expected two P--O singles and a P=O control per structure")
        records.extend(dict(molecule_id=c["molecule_id"],bits=c["bits"],**b) for b in focus)
    by_profile = []
    for cell in profiles():
        rows = [r for r in records if r["bits"] == cell["bits"]]
        single = [r for r in rows if r["expected_type"] == "SINGLE"]
        double = [r for r in rows if r["expected_type"] == "DOUBLE"]
        reachable = [r for r in single if r["presence_probability"] is not None]
        by_profile.append(dict(**cell,single_targets=len(single),single_reachable=len(reachable),
            single_correct=sum(r["stage"] == "reachable_correct" for r in single),
            single_missing=sum(r["stage"] == "reachable_missing" for r in single),
            single_wrong_type=sum(r["stage"] == "reachable_wrong_type" for r in single),
            mean_reachable_probability=statistics.mean(r["presence_probability"] for r in reachable) if reachable else None,
            double_targets=len(double),double_correct=sum(r["stage"] == "reachable_correct" for r in double)))
    effects = []
    for index, factor in enumerate(FACTORS):
        pairs = []
        for (mid, bits), low in sorted(by_cell.items()):
            if bits[index]: continue
            high_bits = list(bits); high_bits[index] = 1
            high = by_cell[(mid,tuple(high_bits))]
            high_bonds = {b["bond_id"]:b for b in high["po_bonds"]}
            for left in low["po_bonds"]:
                if left["expected_type"] != "SINGLE": continue
                right = high_bonds[left["bond_id"]]
                reachable = left["presence_probability"] is not None and right["presence_probability"] is not None
                pairs.append(dict(molecule_id=mid,bond_id=left["bond_id"],low_bits=list(bits),
                    both_reachable=reachable,
                    probability_delta=None if not reachable else right["presence_probability"]-left["presence_probability"],
                    correct_delta=int(right["stage"] == "reachable_correct")-int(left["stage"] == "reachable_correct"),
                    visible_fraction_delta=right["visible_fraction"]-left["visible_fraction"],
                    anchor_length_delta=right["anchor_length_px"]-left["anchor_length_px"]))
        valid = [r["probability_delta"] for r in pairs if r["both_reachable"]]
        effects.append(dict(factor=factor,low=LEVELS[index][0],high=LEVELS[index][1],
            paired_comparisons=len(pairs),both_reachable=len(valid),
            mean_probability_delta=statistics.mean(valid) if valid else None,
            min_probability_delta=min(valid) if valid else None,max_probability_delta=max(valid) if valid else None,
            lost_correct=sum(r["correct_delta"] < 0 for r in pairs),gained_correct=sum(r["correct_delta"] > 0 for r in pairs),
            mean_visible_fraction_delta=statistics.mean(r["visible_fraction_delta"] for r in pairs),
            mean_anchor_length_delta=statistics.mean(r["anchor_length_delta"] for r in pairs),pairs=pairs))
    return dict(images=len(cases),structures=len(mids),single_targets=128,double_targets=64,
                by_profile=by_profile,paired_factor_effects=effects,
                scope="within these four selected structures; repeated factors/bonds, not independent observations or accuracy")
