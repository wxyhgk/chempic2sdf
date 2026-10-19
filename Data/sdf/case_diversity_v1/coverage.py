"""Actual visible case categories and conservative internal pixel partition."""

from collections import Counter, defaultdict


def visible_features(layers, records):
    by_atom = defaultdict(list)
    for g in records['glyphs']: by_atom[g['atom_id']].append(g)
    labels, layouts, features = Counter(), Counter(), Counter()
    for atom in layers['atoms']:
        if not atom['visible_text']: continue
        text = atom['text']; labels[text] += 1
        gs = by_atom[atom['atom_id']]
        vertical = any(a['bbox'][1] >= b['bbox'][3] or b['bbox'][1] >= a['bbox'][3]
                       for i, a in enumerate(gs) for b in gs[i + 1:])
        layouts['vertical_multiglyph' if vertical else 'other_multiglyph' if len(gs) > 1 else 'single_glyph'] += 1
        if 'H' in text and len(text) > 1: features['visible_H_suffix_or_prefix'] += 1
        if any(c.isdigit() for c in text): features['visible_digits'] += 1
        if '+' in text or '-' in text: features['visible_charge_mark'] += 1
        if len(atom['element']) > 1: features['multi_letter_element'] += 1
        incident = [b for b in layers['bonds'] if atom['atom_id'] in (b['begin_atom'], b['end_atom'])]
        if any(b['chemical_type'] in ('DOUBLE', 'TRIPLE') for b in incident): features['text_adjacent_multiple_bond'] += 1
    return dict(labels=dict(labels), layouts=dict(layouts), features=dict(features))


def partition(samples):
    """Never choose a winner between distinct identities with identical RGB."""
    groups = defaultdict(list)
    for sample in samples: groups[sample['rgb_sha256']].append(sample)
    conflict_ids = {s['molecule_id'] for group in groups.values()
                    if len({s['molecule_id'] for s in group}) > 1 for s in group}
    retained, conflict, duplicates, seen = [], [], [], set()
    for s in sorted(samples, key=lambda s: s['sample_key']):
        key = s['sample_key']
        if s['molecule_id'] in conflict_ids: conflict.append(key)
        elif s['rgb_sha256'] in seen: duplicates.append(key)
        else: retained.append(key); seen.add(s['rgb_sha256'])
    return dict(retained=retained, conflict_identity_quarantine=conflict, same_identity_duplicates=duplicates,
        conflict_molecule_ids=sorted(conflict_ids), proposed_internal_quality_only=True,
        training_authorized=False, no_files_deleted=True)
