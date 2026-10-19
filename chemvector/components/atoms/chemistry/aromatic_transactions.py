"""Opt-in CPU aromatic promotion transactions, after predicted atom chemistry.

This is NOT wired into inference. No labels, OCR calls, models, search, or
automatic hydrogen/charge edits. Invalid transaction baselines use the legacy
path; they are not grounds for a repair search. Sanitization is not identity.
"""

from copy import deepcopy
from hashlib import sha256
import json

import networkx as nx

from Data.sdf.chemscene_v0_2 import mol_from_chemscene_v0_2, parse_chemscene_v0_2

MAX_PROPOSALS = 512
MAX_TRANSACTIONS = 128


def scene_digest(scene):
    return sha256(json.dumps(scene, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def transaction_groups(scene, proposals):
    """Validate complete simple rings; overlapping atoms imply ONE transaction."""
    parse_chemscene_v0_2(scene)
    proposals = list(proposals)
    if len(proposals) > MAX_PROPOSALS: raise ValueError('Aromatic proposal cap exceeded')
    bonds = {b['id']: b for b in scene['bond_objects']}
    rings, seen = [], set()
    for proposal in proposals:
        ids = proposal['bond_ids']
        if (not isinstance(ids, list) or len(ids) not in (5, 6) or len(set(ids)) != len(ids)
                or any(type(i) is not int or i not in bonds for i in ids)):
            raise ValueError('A proposal must identify five or six distinct existing bonds')
        ids = tuple(sorted(ids)); graph = nx.Graph()
        for i in ids: graph.add_edge(*bonds[i]['atom_refs'])
        if (len(graph) != len(ids) or not nx.is_connected(graph)
                or any(d != 2 for _, d in graph.degree()) or ids in seen):
            raise ValueError('Distinct complete simple ring proposals required')
        seen.add(ids); rings.append(dict(bond_ids=list(ids), atom_ids=sorted(graph.nodes)))
    rings.sort(key=lambda r: tuple(r['bond_ids']))
    overlap = nx.Graph(); overlap.add_nodes_from(range(len(rings)))
    for i, left in enumerate(rings):
        for j in range(i):
            if set(left['atom_ids']) & set(rings[j]['atom_ids']): overlap.add_edge(i, j)
    groups = []
    for component in nx.connected_components(overlap):
        members = [rings[i] for i in sorted(component)]
        groups.append(dict(rings=members, bond_ids=sorted({b for r in members for b in r['bond_ids']}),
                           atom_ids=sorted({a for r in members for a in r['atom_ids']})))
    groups.sort(key=lambda g: tuple(g['bond_ids']))
    if len(groups) > MAX_TRANSACTIONS: raise ValueError('Aromatic transaction cap exceeded')
    return groups


def guarded_promotions(raw_scene, proposals, *, compiler=mol_from_chemscene_v0_2):
    """Return a scene plus an explicit acceptance/rollback journal.

    ``preexisting_invalid_use_legacy`` requires the caller to retain its legacy
    prediction/fallback; the returned raw scene is NOT an alternate prediction.
    Tests may inject a compiler. It always receives an isolated deep copy.
    """
    before_hash = scene_digest(raw_scene)
    groups = transaction_groups(raw_scene, proposals)
    current = deepcopy(raw_scene)
    journal = dict(status='checked', input_scene_sha256=before_hash, transactions=[],
                   compiler_calls=0, local_rollback=False, use_legacy=False,
                   chemistry_validity_is_not_identity=True)

    def check(scene):
        journal['compiler_calls'] += 1
        try:
            result = compiler(deepcopy(scene))
            if result is None: raise RuntimeError('Compiler returned no molecule')
            return None
        except (ValueError, RuntimeError) as error:
            return str(error)

    error = check(current)
    if error is not None:
        journal.update(status='preexisting_invalid_use_legacy', use_legacy=True, baseline_error=error,
                       skipped_transactions=len(groups), output_scene_sha256=before_hash)
        if scene_digest(raw_scene) != before_hash: raise ValueError('Caller scene mutated')
        return current, journal
    for ordinal, group in enumerate(groups):
        previous_hash = scene_digest(current)
        candidate = deepcopy(current); changes = []
        for i in group['bond_ids']:
            bond = candidate['bond_objects'][i]
            old = {k: bond[k] for k in ('chemical_type', 'display_type')}
            if old != dict(chemical_type='AROMATIC', display_type='AROMATIC'):
                changes.append(dict(bond_id=i, before=old))
                bond.update(chemical_type='AROMATIC', display_type='AROMATIC')
        error = check(candidate) if changes else None
        record = dict(transaction=ordinal, **group, changes=changes, before_scene_sha256=previous_hash,
                      action='rolled_back' if error is not None else ('accepted' if changes else 'noop'))
        if error is not None:
            record['compile_error'] = error
            journal['local_rollback'] = True
        else:
            current = candidate
        record['after_scene_sha256'] = scene_digest(current)
        journal['transactions'].append(record)
    # Elements, H, charge, text, geometry, incidence and connectivity are immutable.
    if current['atom_objects'] != raw_scene['atom_objects']: raise ValueError('Atom chemistry changed')
    allowed = {i for g in groups for i in g['bond_ids']}
    for old, new in zip(raw_scene['bond_objects'], current['bond_objects'], strict=True):
        delta = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
        if delta - {'chemical_type', 'display_type'} or (delta and old['id'] not in allowed):
            raise ValueError('Nontransactional scene edit')
    if scene_digest(raw_scene) != before_hash: raise ValueError('Caller scene mutated')
    journal['output_scene_sha256'] = scene_digest(current)
    return current, journal
