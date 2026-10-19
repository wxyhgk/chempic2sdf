"""Deterministic coverage-first selection from code templates and fixed TRAIN32.

Only the explicitly bound manifest's training_samples are inspected. No held-out
sample files or identity services. Re-rendered training identities are not new
unseen molecules; template chemistry is not experimental stability evidence.
"""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from Data.sdf.chemvector_curriculum_v0_2.catalog import source_records
from Data.sdf.chemvector_curriculum_v0_2.coverage import chemical_features


MANIFEST = Path('outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json')
MANIFEST_SHA = 'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111'
SEEDS = ('CC[NH2+]C', 'CC[NH3+]', 'C[N+](C)(C)C', 'CC(=O)[O-]', '[13CH3]CO',
         'O=P(O)(O)OCC', 'CS(=O)(=O)N', 'CC#N', 'ClCCBr', 'O=c1cc[nH]c(=O)[nH]1',
         'F[C@H](Cl)Br', 'C/C=C/C')


def size_bucket(n):
    return '02-05' if n <= 5 else '06-12' if n <= 12 else '13-24' if n <= 24 else '25-40' if n <= 40 else '41+'


def features(entry):
    chemistry = chemical_features(entry['canonical_smiles']); result = {'size:' + size_bucket(entry['heavy_atoms'])}
    for name in ('elements', 'bond_types', 'element_pair_bond_types', 'functional_groups', 'formal_charge_atoms', 'bond_stereo'):
        result.update(name + ':' + value for value in chemistry[name])
    for name in ('specified_tetrahedral_centers', 'isotope_atoms'):
        if chemistry[name]: result.add(name)
    result.update('stage:' + stage for stage in entry['stages'])
    mol = Chem.MolFromSmiles(entry['canonical_smiles'])
    for atom in mol.GetAtoms():
        neighbors = sorted(n.GetSymbol() + ':' + str(mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondType())
                           for n in atom.GetNeighbors())
        result.add('local:' + atom.GetSymbol() + ':' + str(atom.GetFormalCharge()) + ':' + ','.join(neighbors))
    return sorted(result)


def pool(*, include_train32=True):
    records = source_records()
    records.extend(dict(record_id=f'diversity_seed_{i:02d}', name='diversity seed ' + str(i),
                        smiles=s, stage='atoms_bonds', concepts=['label_context_seed']) for i, s in enumerate(SEEDS))
    provenance = dict(template_sources='existing_v0.2_code_catalog_plus_12_label_context_seeds',
                      training_manifest_read=False, evaluation_sample_files_read=False, identity_queries=0)
    if include_train32:
        data = MANIFEST.read_bytes()
        if sha256(data).hexdigest() != MANIFEST_SHA: raise ValueError('Fixed TRAIN manifest changed')
        rows = json.loads(data)['training_samples']
        if len(rows) != 58 or any(r['split'] != 'train' for r in rows): raise ValueError('TRAIN-only manifest required')
        selected = [r for r in rows if r['fit_domain'] == 'old_full_molecule']
        if len(selected) != 32: raise ValueError('Fixed32 source rows required')
        for i, row in enumerate(selected):
            records.append(dict(record_id=f'diversity_train32_{i:02d}', name=row['sample_key'],
                smiles=row['canonical_smiles'], stage='full_molecule', concepts=['existing_train_identity_rerender']))
        provenance.update(training_manifest_read=True, training_manifest_sha256=MANIFEST_SHA,
                          source_training_rows=32, source_sample_files_read=False)
    entries = merged_records(records)
    for e in entries:
        e['diversity_features'] = features(e)
        e['source_kind'] = 'existing_train_identity' if 'full_molecule' in e['stages'] else 'template_chemistry'
    return entries, provenance


def select(entries, count):
    if type(count) is not int or not 1 <= count <= len(entries): raise ValueError('Invalid fixed case count')
    by_id = {e['molecule_id']: e for e in entries}
    if len(by_id) != len(entries): raise ValueError('Duplicate molecule identity')
    # Seeds and TRAIN32 precede greedy coverage; membership never uses model outcomes.
    mandatory = sorted(e['molecule_id'] for e in entries if 'full_molecule' in e['stages']
                       or any(r.startswith('diversity_seed_') for r in e['source_record_ids']))
    chosen = mandatory[:count]; remaining = set(by_id) - set(chosen)
    counts = Counter(f for key in chosen for f in by_id[key]['diversity_features'])
    while len(chosen) < count:
        # Deterministic ties by molecule hash. Diminishing feature support avoids
        # repeatedly spending cases on the most common C-C/single environments.
        key = min(remaining, key=lambda key: (-sum(1 / (1 + counts[f]) for f in by_id[key]['diversity_features']), key))
        chosen.append(key); remaining.remove(key); counts.update(by_id[key]['diversity_features'])
    return [by_id[key] for key in chosen]


def inventory(entries):
    scaffolds, cores = set(), set()
    for e in entries:
        mol = Chem.MolFromSmiles(e['canonical_smiles']); Chem.RemoveStereochemistry(mol)
        cores.add(Chem.MolToSmiles(mol)); scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if scaffold: scaffolds.add(scaffold)
    return dict(distinct_isomeric_structures=len(entries), nonisomeric_connectivity_ids=len(cores),
        nonempty_murcko_scaffolds=len(scaffolds), acyclic_not_counted_as_one_scaffold=True,
        size_buckets=dict(Counter(size_bucket(e['heavy_atoms']) for e in entries)),
        source_kinds=dict(Counter(e['source_kind'] for e in entries)),
        feature_support=dict(sorted(Counter(f for e in entries for f in e['diversity_features']).items())))
