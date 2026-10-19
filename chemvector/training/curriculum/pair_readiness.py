"""Training-side pair coverage, not a model score or a data-release authority."""

from collections import Counter
from itertools import combinations
import math

from rdkit import Chem


MOTIFS = ('N-N:SINGLE', 'N-N:DOUBLE', 'N-S:SINGLE', 'C-N:SINGLE', 'O-O:SINGLE')
REQUIRED = ('N-N:SINGLE', 'N-S:SINGLE', 'C-N:SINGLE', 'O-O:SINGLE',
            'intervening_nonedge', 'O-S-O_nonedge')


def chemistry(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('Invalid training chemistry')
    motifs = Counter()
    for bond in molecule.GetBonds():
        elements = sorted((bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()))
        motifs['-'.join(elements) + ':' + str(bond.GetBondType())] += 1
    stereo = any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in molecule.GetAtoms())
    stereo |= any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in molecule.GetBonds())
    return {'heavy_atoms': molecule.GetNumHeavyAtoms(), 'specified_stereo': stereo,
            'bond_counts': dict(motifs)}


def training_inventory(manifest):
    """Count manifest images and distinct chemical identities separately.

    All sources are included; no pixels, predictions, or external identity
    index are read. Small means 2--7 heavy atoms, NOT verified clean rendering.
    """
    rows = [r for r in manifest['samples'] if r['split'] == 'train']
    result, cache = {}, {}
    for source in sorted({r['source'] for r in rows}):
        source_rows = [r for r in rows if r['source'] == source]
        stats = {}
        for row in source_rows:
            smiles = row['canonical_smiles']
            if smiles not in cache:
                cache[smiles] = chemistry(smiles)
        for tier in ('all', 'small_2_to_7'):
            current = [r for r in source_rows if tier == 'all'
                       or 2 <= cache[r['canonical_smiles']]['heavy_atoms'] <= 7]
            stats[tier] = {'images': len(current),
                'molecules': len({r['canonical_smiles'] for r in current}),
                'motifs': {}}
            for motif in MOTIFS:
                matching = [r for r in current if cache[r['canonical_smiles']]['bond_counts'].get(motif, 0)]
                stats[tier]['motifs'][motif] = {
                    'images': len(matching), 'molecules': len({r['canonical_smiles'] for r in matching}),
                    'bond_instances_across_images': sum(cache[r['canonical_smiles']]['bond_counts'][motif]
                                                       for r in matching)}
        result[source] = stats
    return result


def reference_training_rows(manifest):
    """Every admitted curriculum train/reference row, never validation fallback."""
    rows = sorted((dict(r) for r in manifest['samples'] if r['split'] == 'train'
                   and r['source'] == 'curriculum' and r['domain'] == 'curriculum'
                   and r['sample_key'].endswith('__reference')), key=lambda r: r['sample_key'])
    if not rows or any(len({r[k] for r in rows}) != len(rows)
                       for k in ('sample_key', 'canonical_smiles', 'molecule_id')):
        raise ValueError('Missing or duplicate training reference identities')
    # Release verification checks scaffold/connectivity splits more strongly;
    # this local check also prevents accidentally assembling a validation panel.
    for field in ('canonical_smiles', 'molecule_id', 'group_key'):
        held = {r[field] for r in manifest['samples'] if r['split'] == 'validation'}
        if held & {r[field] for r in rows}:
            raise ValueError('Training reference rows overlap validation')
    return rows


def geometry_negatives(graph, layers):
    """Label-space hard negatives; never an inference veto or pseudo-label.

    A confirmed non-edge with a shared neighbor whose anchor lies inside the
    segment (10--90%, <= 2 px away). Count each endpoint pair once, retaining
    all witnessing middle nodes. Other non-edges are still valid negatives.
    """
    atoms = {a['atom_id']: a for a in graph['atoms']}
    positions = {a['atom_id']: a['center'] for a in layers['atoms']}
    if (len(atoms) != len(graph['atoms']) or len(positions) != len(layers['atoms'])
            or positions.keys() != atoms.keys()):
        raise ValueError('Geometry atom IDs disagree')
    if any(len(p) != 2 or any(not math.isfinite(x) for x in p) for p in positions.values()):
        raise ValueError('Invalid anchor geometry')
    adjacency = {i: set() for i in atoms}
    for bond in graph['bonds']:
        i, j = bond['begin_atom'], bond['end_atom']
        if i == j or i not in atoms or j not in atoms or j in adjacency[i]:
            raise ValueError('Invalid target edge')
        adjacency[i].add(j)
        adjacency[j].add(i)
    structured, crossing = [], []
    for i, j in combinations(sorted(atoms), 2):
        if j in adjacency[i]:
            continue  # A genuine O--O bond is never a negative.
        common = sorted(adjacency[i] & adjacency[j])
        if not common:
            continue
        structured.append([i, j])
        x, y = positions[i]
        dx, dy = positions[j][0] - x, positions[j][1] - y
        length2 = dx * dx + dy * dy
        if length2 == 0:
            continue
        witnesses = []
        for k in common:
            kx, ky = positions[k][0] - x, positions[k][1] - y
            fraction = (kx * dx + ky * dy) / length2
            distance = abs(kx * dy - ky * dx) / math.sqrt(length2)
            if .1 <= fraction <= .9 and distance <= 2.:
                witnesses.append({'middle_atom': k, 'fraction': fraction, 'distance_px': distance,
                    'elements': [atoms[i]['element'], atoms[k]['element'], atoms[j]['element']]})
        if witnesses:
            crossing.append({'pair': [i, j], 'witnesses': witnesses})
    return {'structured_negative_pairs': structured, 'intervening_nonedge_pairs': crossing}


def describe_reference(row, graph, layers):
    facts = chemistry(row['canonical_smiles'])
    geometry = geometry_negatives(graph, layers)
    counts = {key: facts['bond_counts'].get(key, 0) for key in MOTIFS}
    crossing = geometry['intervening_nonedge_pairs']
    counts['intervening_nonedge'] = len(crossing)
    counts['O-S-O_nonedge'] = sum(any(w['elements'] == ['O', 'S', 'O'] for w in p['witnesses'])
                                     for p in crossing)
    eligible = 2 <= facts['heavy_atoms'] <= 7 and not facts['specified_stereo']
    return {'row': dict(row), **facts, **geometry, 'coverage_counts': counts,
            'eligible_clean_pair_stage': eligible,
            'exclusion_reasons': ([] if 2 <= facts['heavy_atoms'] <= 7 else ['heavy_atom_count'])
                                 + (['specified_stereo'] if facts['specified_stereo'] else [])}


def summarize_references(records):
    eligible = [r for r in records if r['eligible_clean_pair_stage']]
    coverage = {key: {'molecules': sum(r['coverage_counts'][key] > 0 for r in eligible),
                      'instances': sum(r['coverage_counts'][key] for r in eligible)} for key in REQUIRED}
    missing = [key for key in REQUIRED if not coverage[key]['molecules']]
    return {'reference_training_molecules': len(records), 'eligible_molecules': len(eligible),
            'excluded_molecules': len(records) - len(eligible), 'coverage': coverage,
            'missing_required_motifs': missing, 'minimal_target_coverage_passed': not missing,
            'training_ready': False,
            'note': 'Target coverage is necessary, not sufficient; predicted reachability and a new run budget are still required'}
