"""Fixed, hand-authored pair teaching candidates; quarantine only.

Local release checks cannot replace public/private external identity admission.
No evaluation index, image, or checkpoint is opened by this module.
"""

from collections import Counter
from itertools import combinations

from rdkit import Chem

from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from .admission import identities
from .pair_readiness import chemistry, geometry_negatives


# (canonical SMILES, descriptive name, intended pair concept). Frozen before
# rendering; failed samples are recorded, never replaced or retuned.
SPECS = (
    ('CNN', 'Methylhydrazine', 'N-N:SINGLE'),
    ('CCNN', 'Ethylhydrazine', 'N-N:SINGLE'),
    ('CNNC', '1,2-Dimethylhydrazine', 'N-N:SINGLE'),
    ('CN(C)N', '1,1-Dimethylhydrazine', 'N-N:SINGLE'),
    ('CC(=O)NN', 'Acetohydrazide', 'N-N:SINGLE'),
    ('NNC(N)=O', 'Semicarbazide', 'N-N:SINGLE'),
    ('NNC=O', 'Formohydrazide', 'N-N:SINGLE'),
    ('COC(=O)NN', 'Methyl hydrazinecarboxylate', 'N-N:SINGLE'),
    ('CS(N)(=O)=O', 'Methanesulfonamide', 'N-S:SINGLE'),
    ('CCS(N)(=O)=O', 'Ethanesulfonamide', 'N-S:SINGLE'),
    ('CCCS(N)(=O)=O', 'Propanesulfonamide', 'N-S:SINGLE'),
    ('CCS(=O)(=O)NC', 'N-Methylethanesulfonamide', 'N-S:SINGLE'),
    ('NS(N)(=O)=O', 'Sulfamide', 'N-S:SINGLE'),
    ('CNS(N)(=O)=O', 'N-Methylsulfamide', 'N-S:SINGLE'),
    ('CNS(=O)(=O)NC', "N,N'-Dimethylsulfamide", 'N-S:SINGLE'),
    ('CN(C)S(N)(=O)=O', 'N,N-Dimethylsulfamide', 'N-S:SINGLE'),
    ('NCCS', 'Cysteamine neutral representation', 'N-S:NONEDGE'),
    ('NC(N)=S', 'Thiourea', 'N-S:NONEDGE'),
)

CONTROL_SMILES = ('OO', 'CS(C)(=O)=O', 'CCS(C)(=O)=O', 'N=N', 'C#C', 'c1ccncc1', 'CCN')
REQUIRED_COVERAGE = ('N-N:SINGLE', 'N-S:SINGLE', 'C-N:SINGLE', 'O-O:SINGLE',
    'N-N:NONEDGE', 'N-S:NONEDGE', 'O-S-O:INTERVENING_NONEDGE',
    'TYPE:SINGLE', 'TYPE:DOUBLE', 'TYPE:TRIPLE', 'TYPE:AROMATIC')


def candidate_entries(specs=SPECS):
    records, seen, seen_core = [], set(), set()
    for index, (smiles, name, concept) in enumerate(specs):
        mol = Chem.MolFromSmiles(smiles)
        if (mol is None or not 2 <= mol.GetNumAtoms() <= 7 or len(Chem.GetMolFrags(mol)) != 1
                or any(a.GetAtomicNum() <= 1 or a.GetIsotope() or a.GetFormalCharge()
                       or a.GetNumRadicalElectrons() or a.GetAtomMapNum() for a in mol.GetAtoms())):
            raise ValueError('Expected neutral, connected, isotope-free 2--7 node candidates')
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        facts = chemistry(canonical)
        core = identities(canonical)['group_key']
        if canonical != smiles or facts['specified_stereo'] or canonical in seen or core in seen_core:
            raise ValueError('Candidates must be unique canonical non-stereo identities/cores')
        if concept in ('N-N:SINGLE', 'N-S:SINGLE'):
            if not facts['bond_counts'].get(concept):
                raise ValueError('Candidate lacks its declared positive single bond')
        elif concept == 'N-S:NONEDGE':
            symbols = {a.GetSymbol() for a in mol.GetAtoms()}
            if not {'N', 'S'} <= symbols or any(k.startswith('N-S:') for k in facts['bond_counts']):
                raise ValueError('Negative control must contain N and S without a direct N--S edge')
        else:
            raise ValueError('Unsupported candidate concept')
        seen.add(canonical)
        seen_core.add(core)
        records.append({'record_id': f'pair_gap_{index + 1:02d}', 'smiles': canonical,
            'name': name, 'stage': 'functional_groups', 'concepts': [concept, 'pair_gap_quarantine']})
    if not records:
        raise ValueError('Empty candidate plan')
    entries = merged_records(records)
    if len(entries) != len(records):
        raise ValueError('Candidate identities unexpectedly merged')
    return entries


def local_identity_review(entries, manifest):
    """Only compare to already released identities, not an external index."""
    lookup = {field: set() for field in ('inchikey', 'connectivity_prefix', 'group_key')}
    released = {row['canonical_smiles'] for row in manifest['samples']}
    for smiles in sorted(released):
        identity = identities(smiles)
        for field in lookup:
            lookup[field].add(identity[field])
    records = []
    for entry in entries:
        identity = identities(entry['canonical_smiles'])
        records.append({'molecule_id': entry['molecule_id'], 'canonical_smiles': entry['canonical_smiles'],
                        **identity, 'released_overlap_fields': [f for f in lookup if identity[f] in lookup[f]]})
    return {'scope': 'Current released training and exposed validation manifest only',
            'released_unique_smiles': len(released), 'candidates': records,
            'local_clear': all(not r['released_overlap_fields'] for r in records),
            'external_overlap_status': 'unchecked', 'training_authorized': False}


def control_rows(manifest):
    result = []
    for smiles in CONTROL_SMILES:
        rows = [r for r in manifest['samples'] if r['source'] == 'curriculum' and r['split'] == 'train'
                and r['domain'] == 'curriculum' and r['sample_key'].endswith('__reference')
                and r['canonical_smiles'] == smiles]
        if len(rows) != 1:
            raise ValueError('Missing or duplicate existing training control; no validation fallback')
        result.append(dict(rows[0]))
    return result


def target_coverage(sample):
    """Coverage from graph and actual rendered anchors, not predicted atoms."""
    graph = sample['graph']
    counts = Counter(chemistry(graph['canonical_smiles'])['bond_counts'])
    atoms = {a['atom_id']: a for a in graph['atoms']}
    edges = {tuple(sorted((b['begin_atom'], b['end_atom']))) for b in graph['bonds']}
    for bond in graph['bonds']:
        counts['TYPE:' + bond['chemical_type']] += 1
    for i, j in combinations(sorted(atoms), 2):
        if (i, j) not in edges:
            counts['-'.join(sorted((atoms[i]['element'], atoms[j]['element']))) + ':NONEDGE'] += 1
    geometry = geometry_negatives(graph, sample['layers'])
    counts['O-S-O:INTERVENING_NONEDGE'] = sum(
        any(w['elements'] == ['O', 'S', 'O'] for w in pair['witnesses'])
        for pair in geometry['intervening_nonedge_pairs'])
    return {'sample_key': sample['key'], 'canonical_smiles': graph['canonical_smiles'],
            'counts': dict(counts), 'geometry': geometry}


def coverage_summary(records):
    # Records explicitly retain origin and target compatibility. A failed
    # render/adapter is a reported denominator failure, never a passing control.
    compatible = [r for r in records if r['compatible']]
    coverage = {key: {'molecules': len({r['canonical_smiles'] for r in compatible if r['counts'].get(key, 0)}),
                      'instances': sum(r['counts'].get(key, 0) for r in compatible)} for key in REQUIRED_COVERAGE}
    missing = [k for k in REQUIRED_COVERAGE if not coverage[k]['molecules']]
    return {'coverage': coverage, 'missing_controls': missing,
            'minimal_target_coverage_passed': not missing, 'predicted_reachability_checked': False,
            'training_authorized': False, 'training_ready': False}
