from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json

import pytest
from rdkit import Chem

from Data.sdf.case_diversity_v1.catalog import features, select, inventory
from Data.sdf.case_diversity_v1.profiles import profiles, pair_coverage, FAMILIES, FACTORS, font_record
from Data.sdf.case_diversity_v1.rendering import render
from Data.sdf.case_diversity_v1.coverage import partition, visible_features
from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile, render as legacy_render
from chemvector.training.curriculum.data import validate_records


def entry(smiles='CC[NH3+]'):
    value = merged_records([dict(record_id='synthetic', name='synthetic', stage='atoms_bonds',
                                concepts=['synthetic'], smiles=smiles)])[0]
    value['source_kind'] = 'template_chemistry'; value['diversity_features'] = features(value)
    return value


def test_orthogonal_profiles_cover_every_pair_once():
    ps = profiles()
    assert len(ps) == len({p.profile_id for p in ps}) == 16
    assert set(pair_coverage(ps).values()) == {16} and len(pair_coverage(ps)) == 10
    for factor in FACTORS: assert set(Counter(getattr(p, factor) for p in ps).values()) == {4}
    assert all(p.width == p.height == 768 for p in ps)


def test_selection_deterministic_and_chemical_counts_not_style_counts():
    es = [entry(s) for s in ('CC', 'CC#N', 'CC[NH3+]', 'CC(=O)O', 'F[C@H](Cl)Br', 'c1ccccc1')]
    chosen = select(es, 4)
    assert chosen == select(list(reversed(es)), 4)
    assert len({e['canonical_smiles'] for e in chosen}) == 4
    assert inventory(chosen)['distinct_isomeric_structures'] == 4
    with pytest.raises(ValueError): select(es, 7)
    with pytest.raises(ValueError): select(es + [es[0]], 3)


@pytest.mark.parametrize('family', FAMILIES)
def test_font_is_explicit_and_same_source_labels_remain_valid(family):
    p = Profile(font_family=family, font_size=18, bond_length_cap=52, bond_line_width=1.6)
    files, meta = render(entry(), p)
    graph, layers, records = (json.loads(files[s]) for s in ('.graph.json', '.layers.json', '.text-records.json'))
    validate_records(graph, layers, records)
    assert meta['training_authorized'] is False and meta['split'] == layers['split'] == 'quarantine'
    assert layers['renderer']['font']['family'] == family
    assert all(family in g['svg_attributes']['style'] for g in records['glyphs'])
    assert all(sha256(files[s]).hexdigest() == h for s, h in meta['sha256'].items())
    coverage = visible_features(layers, records)
    assert coverage['features']['visible_charge_mark'] >= 1 and coverage['features']['visible_digits'] >= 1


def test_unsupported_font_and_admitted_entry_rejected():
    with pytest.raises(ValueError): font_record('fallback-font')
    e = entry(); e['training_authorized'] = True
    with pytest.raises(ValueError): render(e, Profile())


@pytest.mark.parametrize('smiles', ['CC(=O)N', 'F[C@H](Cl)Br', 'C/C=C/C'])
def test_frozen_renderer_pixel_and_stereo_parity_for_existing_font(smiles):
    e = entry(smiles); p = Profile()
    files, _ = render(e, p); old, _ = legacy_render(e, p, typography='stereo-aware-v1')
    for suffix in ('.png', '.sdf', '.text.png', '.bond.png', '.auxiliary.png', '.semantic.png'):
        assert files[suffix] == old[suffix]
    mol = Chem.MolFromMolBlock(files['.sdf'].decode().split('$$$$')[0], removeHs=False)
    assert Chem.MolToSmiles(mol) == e['canonical_smiles']


def test_partition_quarantines_whole_conflict_identity_and_preserves_all_rows():
    rows = [dict(sample_key=k, molecule_id=m, rgb_sha256=h) for k, m, h in
            [('a0','a','same'), ('b0','b','same'), ('a1','a','unique_a'),
             ('c0','c','c'), ('c1','c','c'), ('d0','d','d')]]
    p = partition(rows)
    assert p['conflict_identity_quarantine'] == ['a0','a1','b0']
    assert p['same_identity_duplicates'] == ['c1'] and p['retained'] == ['c0','d0']
    assert not p['training_authorized'] and p['no_files_deleted']
