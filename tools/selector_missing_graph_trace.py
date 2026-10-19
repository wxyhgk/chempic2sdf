"""Saved-graph diagnosis, not a replacement matcher or a training-label source."""

import math
from types import SimpleNamespace

from rdkit import Chem
import torch

from chemvector.bond_object import BOND_TYPE_TO_ID
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.train_direct_atom_pair_bond import build_pair_targets, match_atom_targets


def selected_candidates(geometry, record):
    if geometry['image_size'] != 768 or geometry['match_radius_px'] != 6:
        raise ValueError('Original 768/6px contract required')
    candidates = geometry['candidates']
    slots = [c['slot'] for c in candidates]
    logits = record['metrics']['logits']
    if (len(set(slots)) != len(slots) or set(logits) != {str(s) for s in slots}
            or any(not math.isfinite(v) for v in logits.values())):
        raise ValueError('Unique candidates and complete finite scores required')
    selected = [c for c in candidates if logits[str(c['slot'])] >= -.25]
    if len(selected) != record['metrics']['selected_atoms']:
        raise ValueError('Saved selection count changed')
    return selected


def spatial_correspondence(geometry, selected):
    """Coordinate-only, reciprocal, unique association; never uses element/bonds.

    An association must also lie strictly inside half the nearest GT separation.
    This intentionally differs from the official 6px acceptance: it identifies
    what an already saved node probably represents, without calling it a pass.
    Duplicate claimants, ties and remote points remain unresolved.
    """
    anchors = geometry['target_anchors']
    if len(anchors) < 2 or len({tuple(p) for p in anchors}) != len(anchors):
        raise ValueError('At least two distinct target anchors required')
    if any(not math.isfinite(v) or not 0 <= v <= 1
           for p in anchors + [c['geometry'][:2] for c in selected] for v in p):
        raise ValueError('Finite normalized coordinates required')
    distances = {c['slot']: [math.dist(c['geometry'][:2], a) * 768 for a in anchors]
                 for c in selected}
    nearest = {s: min(range(len(anchors)), key=ds.__getitem__) for s, ds in distances.items()}
    rows = []
    for c in selected:
        slot = c['slot']
        ds, target = distances[slot], nearest[slot]
        separation = min(math.dist(anchors[target], a) * 768
                         for i, a in enumerate(anchors) if i != target)
        reasons = []
        if sorted(ds)[0] == sorted(ds)[1]:
            reasons.append('target_distance_tie')
        if sum(t == target for t in nearest.values()) != 1:
            reasons.append('multiple_selected_claimants')
        if ds[target] >= separation / 2:
            reasons.append('outside_local_separation_ball')
        if any(other != slot and values[target] <= ds[target] for other, values in distances.items()):
            reasons.append('not_unique_reciprocal_nearest')
        rows.append(dict(slot=slot, target_id=target, distance_px=ds[target],
            second_target_distance_px=sorted(ds)[1], local_separation_radius_px=separation / 2,
            accepted_for_diagnostic_only=not reasons, unresolved_reasons=reasons))
    return rows


def compiled_graph(prediction, selected):
    """Recover exact saved candidate lineage through the compiler's q1023 anchors.

    SDF coordinate units are q1023 / 100, with inverted y. Do not geometrically
    fit, use SMILES atom order, or guess provenance if this exact join fails.
    """
    if prediction['molblock'] is None:
        if prediction['valid'] or prediction['exact']:
            raise ValueError('Missing MolBlock cannot certify a valid/exact graph')
        return None
    mol = Chem.MolFromMolBlock(prediction['molblock'], sanitize=True, removeHs=False)
    if mol is None or not prediction['valid'] or prediction['fallback']:
        raise ValueError('Require the saved valid non-fallback molecular graph')
    by_anchor = {}
    for c in selected:
        anchor = tuple(max(0, min(1023, round(v * 1023))) for v in c['geometry'][:2])
        if anchor in by_anchor:
            raise ValueError('Quantized candidate collision: lineage is ambiguous')
        by_anchor[anchor] = c['slot']
    if mol.GetNumAtoms() != len(selected):
        raise ValueError('Compiled/selected atom count mismatch')
    nodes, slot_by_index = [], {}
    conformer = mol.GetConformer()
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        p = conformer.GetAtomPosition(index)
        scaled = (p.x * 100, -p.y * 100)
        anchor = tuple(round(v) for v in scaled)
        if any(abs(a - b) > 1e-5 for a, b in zip(scaled, anchor)) or anchor not in by_anchor:
            raise ValueError('Saved SDF coordinate lacks exact q1023 lineage')
        slot = by_anchor[anchor]
        slot_by_index[index] = slot
        nodes.append(dict(mol_atom_index=index, slot=slot, q1023_anchor=list(anchor),
            element=atom.GetSymbol(), formal_charge=atom.GetFormalCharge(), isotope=atom.GetIsotope(),
            aromatic=atom.GetIsAromatic(), total_h=atom.GetTotalNumHs(),
            radical_electrons=atom.GetNumRadicalElectrons()))
    if len(set(slot_by_index.values())) != len(selected):
        raise ValueError('SDF lineage is not one-to-one')
    edges = [dict(mol_bond_index=b.GetIdx(), begin_slot=slot_by_index[b.GetBeginAtomIdx()],
                  end_slot=slot_by_index[b.GetEndAtomIdx()], chemical_type=str(b.GetBondType()))
             for b in mol.GetBonds()]
    return dict(nodes=nodes, edges=edges)


@torch.no_grad()
def replay_pair_labels(geometry, record, layers):
    """Call the existing target builder unchanged, with saved selected coordinates.

    No pair sampling, model call, gradients or substitute positive labels. A
    false presence entry is eligible for the current negative sampler, not proof
    that a historical optimizer step actually sampled it.
    """
    selected = selected_candidates(geometry, record)
    capacity = max(c['slot'] for c in geometry['candidates']) + 1
    xy = torch.zeros(1, capacity, 8)
    mask = torch.zeros(1, capacity, dtype=torch.bool)
    for c in selected:
        xy[0, c['slot']] = torch.tensor(c['geometry'])
        mask[0, c['slot']] = True
    atoms = dict(atom_geometry=xy, atom_mask=mask, atom_indices=torch.arange(capacity)[None])
    bonds = layers['bonds']
    batch = dict(atom_anchors=torch.tensor([geometry['target_anchors']], dtype=torch.float32),
        atom_mask=torch.ones(1, len(layers['atoms']), dtype=torch.bool),
        bond_mask=torch.ones(1, len(bonds), dtype=torch.bool),
        bond_atom_refs=torch.tensor([[[b['begin_atom'], b['end_atom']] for b in bonds]],
                                   dtype=torch.long).reshape(1, len(bonds), 2),
        bond_chemical=torch.tensor([[BOND_TYPE_TO_ID[b['chemical_type']] for b in bonds]], dtype=torch.long))
    mapping, matched = match_atom_targets(atoms, batch,
        SimpleNamespace(image_size=768, atom_match_radius_px=6))
    if matched[0].nonzero().flatten().tolist() != record['metrics']['matched_target_ids']:
        raise ValueError('Original selected matcher replay differs from saved baseline')
    pairs, rows = enumerate_unordered_atom_pairs(mask)
    labels = build_pair_targets(pairs, rows, mapping, batch, atoms)
    pair_labels = {tuple(sorted(pair)): bool(label) for pair, label
                   in zip(pairs.tolist(), labels['presence'].tolist())}
    return mapping[0].tolist(), pair_labels


def inspect_case(geometry, record, layers, target_ids):
    selected = selected_candidates(geometry, record)
    correspondence = spatial_correspondence(geometry, selected)
    by_slot = {r['slot']: r['target_id'] for r in correspondence if r['accepted_for_diagnostic_only']}
    by_target = {target: slot for slot, target in by_slot.items()}
    graph = compiled_graph(record['prediction'], selected)
    mapping, labels = replay_pair_labels(geometry, record, layers)
    nodes = {n['slot']: n for n in graph['nodes']} if graph else {}
    results = []
    for target in target_ids:
        atom = layers['atoms'][target]
        if atom['atom_id'] != target:
            raise ValueError('Source atom IDs must be ordered')
        slot = by_target.get(target)
        location = next((c for c in correspondence if c['slot'] == slot), None)
        expected = []
        for b in layers['bonds']:
            if target not in (b['begin_atom'], b['end_atom']):
                continue
            other = b['end_atom'] if b['begin_atom'] == target else b['begin_atom']
            other_slot = by_target.get(other)
            pair = tuple(sorted((slot, other_slot))) if slot is not None and other_slot is not None else None
            expected.append(dict(bond_id=b['bond_id'], neighbor_target=other,
                chemical_type=b['chemical_type'], candidate_pair=list(pair) if pair else None,
                current_training_presence=labels[pair] if pair else None,
                current_negative_pool_eligible=not labels[pair] if pair else None))
        result = dict(target_id=target, text=atom['text'], expected_element=atom['element'],
            selected_correspondence=location, official_selected_target=mapping[slot] if slot is not None else None,
            expected_incident_bonds=expected, compiled_node=nodes.get(slot),
            local_status='unresolved_correspondence' if slot is None else 'no_saved_final_graph',
            missing_bond_ids=None, extra_edges=None, wrong_type_bonds=None)
        if slot in nodes:
            predicted = []
            for edge in graph['edges']:
                if slot in (edge['begin_slot'], edge['end_slot']):
                    other = edge['end_slot'] if slot == edge['begin_slot'] else edge['begin_slot']
                    predicted.append(dict(neighbor_slot=other, neighbor_target=by_slot.get(other),
                                          chemical_type=edge['chemical_type']))
            expected_by_neighbor = {b['neighbor_target']: b for b in expected}
            present = {e['neighbor_target']: e for e in predicted if e['neighbor_target'] is not None}
            missing = [b['bond_id'] for b in expected if b['neighbor_target'] not in present]
            extra = [e for e in predicted if e['neighbor_target'] not in expected_by_neighbor]
            wrong_type = [dict(bond_id=b['bond_id'], expected=b['chemical_type'],
                              predicted=present[b['neighbor_target']]['chemical_type']) for b in expected
                          if b['neighbor_target'] in present
                          and b['chemical_type'] != present[b['neighbor_target']]['chemical_type']]
            element_correct = nodes[slot]['element'] == atom['element']
            status = ('unresolved_neighbor_correspondence' if any(e['neighbor_target'] is None for e in predicted)
                else 'wrong_element' if not element_correct
                else 'incident_bond_error' if missing or extra or wrong_type
                else 'element_and_incident_bonds_correct')
            result.update(local_status=status, predicted_incident_bonds=predicted,
                element_correct=element_correct, missing_bond_ids=missing, extra_edges=extra,
                wrong_type_bonds=wrong_type)
        results.append(result)
    return dict(sample_key=record['sample_key'], whole_molecule={k: record['prediction'][k]
        for k in ('valid', 'exact', 'fallback', 'error')}, correspondence=correspondence,
        compiled_graph=graph, focused_nodes=results, training_authorized=False,
        correspondence_is_not_training_labels=True, acceptance_or_predictions_changed=False,
        comparison_scope='Element, incident endpoints and sanitized bond types; not complete atom/stereo equivalence')
