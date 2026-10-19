"""Lossless candidate-ID connectivity adapter; no geometry deduplication or rescue."""

from copy import deepcopy
import math

from chemvector.ocr import parse_atom_display_text
from chemvector.experiments.id_graph.graph import SCHEMA, integer, records_by_id

BOND_TYPES = ('SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC')


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected finite numeric prediction')
    return float(value)


def from_prediction(record, threshold):
    """Input already contains only selected atoms, but all scored unordered pairs.

    Carbon defaults and the frozen OCR parser reproduce the existing chemical
    interpretation policy. Defaults remain explicit in observations. No truth
    or SDF is accepted by this interface. Empty/invalid chemistry is kept for scoring.
    """
    finite(threshold)
    atoms = records_by_id(record['atoms'], 'atom')
    pairs = records_by_id(record['pairs'], 'pair')
    ocr = records_by_id(record['ocr'], 'ocr')
    if not set(ocr).issubset(atoms):
        raise ValueError('OCR references an unselected candidate')
    graph = {'schema_version': SCHEMA, 'molecule_id': record['sample_key'], 'atoms': [], 'bonds': [],
             'observations': {'coordinate_frame': 'image-px', 'atom_positions': [],
                              'text_predictions': deepcopy(record['ocr']), 'chemical_provenance': []}}
    stats = {'selected_atoms': len(atoms), 'scored_pairs': len(pairs), 'accepted_bonds': 0,
             'parsed_text': 0, 'unparsed_text': 0, 'default_carbon_atoms': 0}
    for key, item in sorted(atoms.items()):
        if not isinstance(item['xy'], list) or len(item['xy']) != 2:
            raise ValueError('Expected two pixel coordinates')
        xy = [finite(x) for x in item['xy']]
        if any(x < 0 or x > 768 for x in xy):
            raise ValueError('Prediction coordinates outside fixed frame')
        finite(item['score'])
        chemical = {'element': 'C', 'formal_charge': 0, 'isotope': 0, 'aromatic': False,
                    'explicit_h_count': 0, 'no_implicit': False, 'radical_electrons': 0}
        text = ocr.get(key, {}).get('text', '')
        if not isinstance(text, str):
            raise ValueError('OCR text must be a string')
        parsed = parse_atom_display_text(text) if text else None
        if parsed is not None:
            chemical.update(parsed)
            chemical['no_implicit'] = parsed['explicit_h_count'] > 0
            stats['parsed_text'] += 1
            source = 'frozen_ocr_parsed'
        else:
            stats['default_carbon_atoms'] += 1
            stats['unparsed_text'] += bool(text)
            source = 'unparsed_text_carbon_default' if text else 'no_text_carbon_default'
        graph['atoms'].append({'id': key, **chemical,
                               'stereo': {'tag': 'CHI_UNSPECIFIED', 'neighbor_ids': []}})
        graph['observations']['atom_positions'].append({'atom_id': key, 'xy': xy})
        graph['observations']['chemical_provenance'].append({'atom_id': key, 'source': source})
    seen = set()
    for key, item in sorted(pairs.items()):
        if not isinstance(item['atom_ids'], list) or len(item['atom_ids']) != 2:
            raise ValueError('Expected an atom pair')
        first, second = [integer(i, 'pair.atom_id', 0) for i in item['atom_ids']]
        if first == second or first not in atoms or second not in atoms:
            raise ValueError('Self or dangling predicted pair')
        pair = frozenset((first, second))
        if pair in seen:
            raise ValueError('Duplicate predicted pair')
        seen.add(pair)
        presence = finite(item['presence_logit'])
        logits = [finite(v) for v in item['chemical_logits']]
        if len(logits) != 4:
            raise ValueError('Expected four chemical bond classes')
        if presence < threshold:
            continue
        kind = BOND_TYPES[max(range(4), key=logits.__getitem__)]
        graph['bonds'].append({'id': key, 'atom_ids': [first, second], 'type': kind,
                               'aromatic': kind == 'AROMATIC', 'conjugated': False,
                               'stereo': 'STEREONONE', 'stereo_atom_ids': []})
    neighbors = {key: [] for key in atoms}
    for bond in graph['bonds']:
        first, second = bond['atom_ids']
        neighbors[first].append(second)
        neighbors[second].append(first)
    for atom in graph['atoms']:
        atom['stereo']['neighbor_ids'] = neighbors[atom['id']]
    stats['accepted_bonds'] = len(graph['bonds'])
    if len(seen) != len(atoms) * (len(atoms) - 1) // 2:
        raise ValueError('Prediction must retain every scored unordered selected-atom pair')
    return graph, stats


def assert_connectivity_preserved(record, graph, threshold):
    """Independent direct check against raw logits; before any RDKit cleanup."""
    if sorted(a['id'] for a in record['atoms']) != sorted(a['id'] for a in graph['atoms']):
        raise ValueError('Adapter changed selected atom identities')
    expected = {p['id']: (p['atom_ids'], BOND_TYPES[max(range(4), key=p['chemical_logits'].__getitem__)])
                for p in record['pairs'] if p['presence_logit'] >= threshold}
    actual = {b['id']: (b['atom_ids'], b['type']) for b in graph['bonds']}
    if expected != actual:
        raise ValueError('Adapter changed thresholded bonds or bond types')
    if graph['observations']['atom_positions'] != [
            {'atom_id': a['id'], 'xy': a['xy']} for a in sorted(record['atoms'], key=lambda a: a['id'])]:
        raise ValueError('Adapter changed observed coordinates')
