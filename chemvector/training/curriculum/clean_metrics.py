"""Post-prediction traces for fixed, aligned, clean curriculum validation images."""

import torch

from chemvector.bond_object import BOND_TYPES
from chemvector.ocr import parse_atom_display_text
from chemvector.train_direct_atom_pair_bond import match_atom_targets
from .headroom_metrics import box_iou
from .upstream_diagnostic import labeled_coverage


def select_clean_rows(manifest):
    """All 16 existing reference-style validation molecules; no score filtering."""
    rows = [dict(r) for r in manifest['samples'] if r['source'] == 'curriculum'
            and r['split'] == 'validation' and r['domain'] == 'curriculum'
            and r['sample_key'].endswith('__reference')]
    if (len(rows) != 16 or len({r['molecule_id'] for r in rows}) != 16
            or len({r['canonical_smiles'] for r in rows}) != 16):
        raise ValueError('Expected the complete 16-molecule clean validation slice')
    return sorted(rows, key=lambda r: r['sample_key'])


def predicted_box(record):
    """Reproduce the float32 scaling used by the actual OCR crop path."""
    x, y = (torch.tensor(record['center'], dtype=torch.float32) * 768).tolist()
    w, h = (torch.tensor(record['half_extent'], dtype=torch.float32) * 768).tolist()
    return [x - w, y - h, x + w, y + h]


def parsed_difference(text, expected):
    """Basic parsed fields only; H needs compiled connectivity/context."""
    parsed = parse_atom_display_text(text) if text is not None else None
    if parsed is None:
        return {'parsed': None, 'basic_field_differences': None}
    fields = ('element', 'formal_charge', 'isotope', 'radical_electrons')
    return {'parsed': parsed, 'basic_field_differences': {
        k: {'expected': expected[k], 'predicted': parsed[k]}
        for k in fields if expected[k] != parsed[k]}}


@torch.no_grad()
def measure_clean(sample, batch, raw, structure, trace, config, pair_threshold, reference_ocr):
    """Truth only scores completed predictions; reference OCR is a separate probe."""
    meta = sample['meta']
    if (sample.get('purpose') != 'diagnostic' or meta['split'] != 'validation'
            or meta['profile_id'] != 'reference' or meta['external_overlap_status'] != 'checked'):
        raise ValueError('Only aligned reference-style validation diagnostics are allowed')
    atoms = structure['atoms']
    coverage = labeled_coverage(raw, atoms, batch, config)
    mapping, _ = match_atom_targets(atoms, batch, config)
    selected_ids = atoms['atom_mask'][0].nonzero().flatten().tolist()
    roles = {r['candidate']: r for r in trace['roles']}
    decoded = {r['candidate']: r for r in trace['ocr']}
    if set(roles) != set(selected_ids):
        raise ValueError('Role trace differs from selected candidates')
    slots = batch['atom_ids'][0]
    chemical = {a['atom_id']: a for a in sample['graph']['atoms']}
    visual = {a['atom_id']: a for a in sample['layers']['atoms']}
    visible_ids = {i for i, a in visual.items() if a['visible_text']}
    if set(reference_ocr) != visible_ids:
        raise ValueError('Reference OCR must cover every and only visible target label')
    target_to_candidate = {int(mapping[0, c]): c for c in selected_ids if int(mapping[0, c]) >= 0}
    raw_ids = raw['atom_mask'][0].nonzero().flatten()
    distance = torch.cdist(batch['atom_anchors'][0, :len(slots)].float(),
                          raw['atom_geometry'][0, raw_ids, :2].float()) * config.image_size
    compiled_ids = trace.get('compiled_candidate_indices', [])
    scene = trace.get('scene_before_fallback')
    scene_atoms = {} if scene is None else dict(zip(compiled_ids, scene['atom_objects']))
    atom_rows = []
    for slot, identity in enumerate(slots):
        candidate = target_to_candidate.get(slot)
        nearest = None
        if len(raw_ids):
            local = int(distance[slot].argmin())
            nearest = {'candidate': int(raw_ids[local]), 'distance_px': float(distance[slot, local])}
        a, v = chemical[identity], visual[identity]
        ocr = decoded.get(candidate)
        entry = {'target_slot': slot, 'atom_id': identity,
                 'expected': {k: a[k] for k in ('element', 'formal_charge', 'isotope', 'radical_electrons')},
                 'anchor_px': (batch['atom_anchors'][0, slot] * config.image_size).tolist(),
                 'candidate': candidate, 'nearest_raw': nearest,
                 'coverage': coverage['target_details'][slot], 'visible_text': v['visible_text'],
                 'expected_text': v['text'], 'role': roles.get(candidate),
                 'predicted_text': None if ocr is None else ocr['text'],
                 'scene_fields': None if candidate not in scene_atoms else scene_atoms[candidate]['chemical']}
        if v['visible_text']:
            entry['reference_crop'] = {'bbox_px': v['bbox'], **reference_ocr[identity],
                **parsed_difference(reference_ocr[identity]['text'], a)}
            entry['predicted_crop'] = None if ocr is None else {
                'bbox_px': predicted_box(ocr), 'text': ocr['text'], 'score': ocr['score'],
                'bbox_iou': box_iou(predicted_box(ocr), v['bbox']),
                **parsed_difference(ocr['text'], a)}
        atom_rows.append(entry)
    pairs = {tuple(sorted(pair)): i for i, pair in enumerate(structure['pairs'].tolist())}
    if len(pairs) != len(structure['pairs']):
        raise ValueError('Duplicate predicted pairs')
    predicted = structure['output']['presence_logits'].sigmoid()
    chemical_probabilities = structure['output']['chemical_logits'].softmax(-1)
    target_pairs, bond_rows = set(), []
    for index in range(int(batch['bond_mask'].sum())):
        left, right = batch['bond_atom_refs'][0, index].tolist()
        target_pairs.add(tuple(sorted((left, right))))
        first, second = target_to_candidate.get(left), target_to_candidate.get(right)
        row = None if first is None or second is None else pairs[tuple(sorted((first, second)))]
        kind = int(batch['bond_chemical'][0, index])
        bond_rows.append({'target_slots': [left, right], 'candidates': [first, second],
            'expected_type': BOND_TYPES[kind], 'reachable': row is not None,
            'presence_probability': None if row is None else float(predicted[row]),
            'present': False if row is None else bool(predicted[row] >= pair_threshold),
            'predicted_type': None if row is None else BOND_TYPES[int(structure['chemistry'][row])],
            'chemical_probabilities': None if row is None else chemical_probabilities[row].tolist()})
    extra = []
    for pair, row in pairs.items():
        mapped = mapping[0, list(pair)].tolist()
        if bool(predicted[row] >= pair_threshold) and tuple(sorted(mapped)) not in target_pairs:
            extra.append({'candidates': list(pair), 'target_slots': mapped,
                'status': 'unknown_correspondence' if min(mapped) < 0 else 'confirmed_extra_edge',
                'presence_probability': float(predicted[row])})
    return {'coverage': coverage, 'atoms': atom_rows, 'bonds': bond_rows, 'extra_edges': extra,
            'candidate_bijection': coverage['selected_matched'] == coverage['selected_atoms'] == coverage['target_atoms'],
            'limits': ['Six-pixel correspondence gaps are not automatically absent visual atoms',
                       'Literal differences and basic parsed fields do not equal full chemical identity',
                       'Reference crops do not replace predictions and do not produce an SDF score']}
