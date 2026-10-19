"""Isolated CPU supervision contracts; NOT wired into training or inference.

Unknown correspondence is not a confirmed non-edge. Chemical bond labels need
whole-molecule context. Literal OCR and compiled atom attributes are separate
observations. None of these local checks replaces strict SDF identity scoring.
"""

from rdkit import Chem
import torch

from chemvector.bond_object import BOND_TYPES
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.train_direct_atom_pair_bond import build_pair_targets, select_training_pairs


@torch.no_grad()
def pair_targets_with_unknown(atoms, atom_mapping, targets):
    """Keep the existing assignment/labels; expose uncertain pairs separately.

    Callers must supply the existing predicted selected candidates and matching.
    No rematching, radius change, replacement candidates or graph repair occurs.
    The old target dictionary is deliberately nested to prevent treating this
    result as a drop-in legacy loss/sampler input that would ignore known_mask.
    """
    selected, target_atoms = atoms['atom_mask'], targets['atom_mask']
    target_bonds = targets['bond_mask']
    masks = (selected, target_atoms, target_bonds)
    if any(m.ndim != 2 or m.dtype != torch.bool for m in masks):
        raise ValueError('Expected two-dimensional boolean masks')
    if (atom_mapping.shape != selected.shape or atom_mapping.dtype != torch.long
            or any(m.shape[0] != selected.shape[0] for m in masks)):
        raise ValueError('Assignment shape, type or batch mismatch')
    required = (atom_mapping, target_atoms, target_bonds,
                targets['bond_atom_refs'], targets['bond_chemical'])
    if any(t.device != selected.device for t in required):
        raise ValueError('Contract tensors must share a device')
    if (targets['bond_atom_refs'].shape != (*target_bonds.shape, 2)
            or targets['bond_atom_refs'].dtype != torch.long
            or targets['bond_chemical'].shape != target_bonds.shape
            or targets['bond_chemical'].dtype != torch.long):
        raise ValueError('Malformed bond target tensors')
    # The existing label builder assumes prefix-packed targets. Fail rather
    # than silently feeding it a different mask convention.
    for mask in (target_atoms, target_bonds):
        prefix = torch.arange(mask.shape[1], device=mask.device)[None] < mask.sum(1)[:, None]
        if not torch.equal(prefix, mask):
            raise ValueError('Existing pair labels require prefix-packed targets')
    if bool((atom_mapping < -1).any()) or bool((atom_mapping[~selected] != -1).any()):
        raise ValueError('Only selected atoms may have a target assignment')
    for i in range(selected.shape[0]):
        mapping = atom_mapping[i, selected[i]]
        matched = mapping[mapping >= 0]
        atom_count = int(target_atoms[i].sum())
        if bool((matched >= atom_count).any()) or len(matched.unique()) != len(matched):
            raise ValueError('Matched target assignments must be in-range and one-to-one')
        refs = targets['bond_atom_refs'][i, target_bonds[i]]
        kinds = targets['bond_chemical'][i, target_bonds[i]]
        if (bool(((refs < 0) | (refs >= atom_count)).any())
                or bool((refs[:, 0] == refs[:, 1]).any())
                or len(refs.sort(dim=1).values.unique(dim=0)) != len(refs)
                or bool(((kinds < 0) | (kinds >= len(BOND_TYPES))).any())):
            raise ValueError('Invalid or duplicate target bonds')
    pairs, rows = enumerate_unordered_atom_pairs(selected)
    legacy = build_pair_targets(pairs, rows, atom_mapping, targets, atoms)
    known = (atom_mapping[rows[:, None], pairs] >= 0).all(dim=1)
    if bool((legacy['presence'] & ~known).any()):
        raise ValueError('A positive target cannot have unknown endpoints')
    reachable = int(legacy['presence'].sum())
    return {
        'pair_indices': pairs, 'sample_indices': rows,
        'known_mask': known, 'unknown_mask': ~known, 'legacy_targets': legacy,
        'coverage': {
            'target_atoms': int(target_atoms.sum()),
            'selected_atoms': int(selected.sum()),
            'unmatched_selected_atoms': int((selected & (atom_mapping < 0)).sum()),
            'target_bonds': int(target_bonds.sum()),
            'reachable_positive_pairs': reachable,
            'unreachable_bonds': int(target_bonds.sum()) - reachable,
            'known_negative_pairs': int((known & ~legacy['presence']).sum()),
            'unknown_pairs': int((~known).sum()),
        },
    }


def select_known_pair_rows(contract, atoms, *, negative_ratio=6, minimum_negatives=48):
    """Use the legacy sampler only on known pairs, returning original row IDs.

    This opt-in prototype excludes uncertain negatives, not real non-edges.
    It does not teach the pair head to suppress unmatched false-positive atoms;
    their count and unreachable targets remain explicit in coverage.
    """
    if any(type(v) is not int or v < 0 for v in (negative_ratio, minimum_negatives)):
        raise ValueError('Negative sampling budgets must be nonnegative integers')
    eligible = torch.where(contract['known_mask'])[0]
    if not len(eligible):
        return eligible
    labels = {k: contract['legacy_targets'][k][eligible]
              for k in ('presence', 'chemical', 'structured_negative')}
    chosen = select_training_pairs(
        contract['pair_indices'][eligible], contract['sample_indices'][eligible],
        atoms, labels, negative_ratio=negative_ratio, minimum_negatives=minimum_negatives,
    )
    return eligible[chosen]


def chemical_bond_labels(molecule):
    """Canonical chemical types by input atom indices, never drawing strokes.

    Sanitize a copy of a complete target molecule. Do not accept a SINGLE bond
    as equivalent to AROMATIC without its surrounding graph, or silently repair
    charge/isotope/stereo identities. This function does not rewrite any data.
    """
    if molecule is None or not molecule.GetNumAtoms():
        raise ValueError('A nonempty complete molecule is required')
    result = Chem.Mol(molecule)

    def protected(mol):
        return (
            [(a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope(),
              a.GetNumRadicalElectrons(), str(a.GetChiralTag())) for a in mol.GetAtoms()],
            [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetStereo()),
              tuple(b.GetStereoAtoms())) for b in mol.GetBonds()],
        )

    before = protected(result)
    try:
        Chem.SanitizeMol(result)
    except (ValueError, RuntimeError) as error:
        raise ValueError('Invalid target molecule; no automatic chemistry repair') from error
    if protected(result) != before:
        raise ValueError('Sanitization changed protected target identity fields')
    labels = {}
    for bond in result.GetBonds():
        kind = str(bond.GetBondType())
        if kind not in BOND_TYPES:
            raise ValueError('Unsupported chemical bond type')
        key = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        labels[key] = kind
    return labels


def compiled_atom_fields(atom):
    """Read a SANITIZED molecule atom; explicit/implicit H share one count.

    This is not a full atom identity or an atom-matching algorithm: stereo,
    connectivity and whole-molecule identity must still be scored separately.
    """
    return {
        'element': atom.GetSymbol(), 'formal_charge': atom.GetFormalCharge(),
        'isotope': atom.GetIsotope(), 'radical_electrons': atom.GetNumRadicalElectrons(),
        'total_h_count': atom.GetTotalNumHs(includeNeighbors=True),
        'aromatic': atom.GetIsAromatic(),
    }


def compare_text_result(expected_text, predicted_text, *, expected_atom, compiled_atom):
    """Compare literal text and independently compiled fields at known endpoints.

    Do not parse a reversed GT glyph string as a formula, or assume an omitted
    H suffix means zero H: the actual compiler may infer H from graph context.
    Missing endpoint/correspondence is unscorable, not a semantic success.
    """
    if not isinstance(expected_text, str) or not expected_text:
        raise ValueError('A visible target transcription is required')
    if predicted_text is not None and not isinstance(predicted_text, str):
        raise ValueError('Predicted transcription must be a string or missing')
    result = {'literal_match': expected_text == predicted_text,
              'compiled_fields_match': None, 'differences': None}
    if expected_atom is not None and compiled_atom is not None:
        expected, actual = compiled_atom_fields(expected_atom), compiled_atom_fields(compiled_atom)
        result['differences'] = {k: {'expected': v, 'predicted': actual[k]}
                                 for k, v in expected.items() if v != actual[k]}
        result['compiled_fields_match'] = not result['differences']
    return result
