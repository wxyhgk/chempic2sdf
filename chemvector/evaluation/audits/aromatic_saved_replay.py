"""Prediction-only reconstruction of saved aromatic proposals; no model I/O.

Accept a predicted scene, candidate IDs, and saved pair logits. Targets, sample
names, and metrics must stay outside this API. Reconstruct legacy output first
and require exact scene equality before an alternate decoder can be tested.
"""

from copy import deepcopy

import networkx as nx
import torch

from chemvector.bond_object import BOND_TYPES
from chemvector.direct_atom_pair_chemistry import promote_complete_aromatic_rings
from chemvector.shadow_chemscene_v02_oracle import apply_aromatic_consistency
from chemvector.components.atoms.chemistry.aromatic_transactions import MAX_PROPOSALS, scene_digest

PRESENCE_THRESHOLD, MEAN_THRESHOLD, MINIMUM_THRESHOLD = .85, .40, .05


def reconstruct(scene, candidate_indices, pairs, presence_logits, chemical_logits):
    """Recover raw types and proposal provenance, preserving post-OCR attributes."""
    original_hash = scene_digest(scene)
    if (pairs.dtype != torch.int64 or pairs.shape != (len(presence_logits), 2)
            or presence_logits.ndim != 1 or chemical_logits.shape != (len(pairs), 4)
            or any(t.device.type != 'cpu' or t.requires_grad for t in (pairs, presence_logits, chemical_logits))
            or any(t.dtype != torch.float32 or not bool(torch.isfinite(t).all())
                   for t in (presence_logits, chemical_logits))
            or len(pairs) > 8128 or bool((pairs < 0).any())
            or bool((pairs[:, 0] >= pairs[:, 1]).any())):
        raise ValueError('Detached finite CPU FP32 predictions and canonical bounded pairs required')
    if (len(candidate_indices) != len(scene['atom_objects'])
            or len(set(candidate_indices)) != len(candidate_indices)
            or any(type(c) is not int or c < 0 for c in candidate_indices)):
        raise ValueError('Distinct saved candidate-to-scene mapping required')
    # This saved trial's empty-scene builder/OCR never sets aromatic atom flags.
    # A different format must supply a pre-promotion trace instead of guessing.
    if any(a['chemical']['aromatic'] for a in scene['atom_objects']):
        raise ValueError('Cannot recover pre-promotion atom flags from this trace')
    pair_rows = {tuple(pair): i for i, pair in enumerate(pairs.tolist())}
    if len(pair_rows) != len(pairs): raise ValueError('Duplicate pair predictions')
    presence = presence_logits.sigmoid(); raw_types = chemical_logits.argmax(-1)
    legacy_types, legacy_diagnostics = promote_complete_aromatic_rings(pairs, presence, chemical_logits,
        presence_threshold=PRESENCE_THRESHOLD, mean_threshold=MEAN_THRESHOLD, minimum_threshold=MINIMUM_THRESHOLD)
    raw, legacy, scene_rows = deepcopy(scene), deepcopy(scene), {}
    for bond in scene['bond_objects']:
        pair = tuple(sorted(candidate_indices[i] for i in bond['atom_refs']))
        if pair not in pair_rows or float(presence[pair_rows[pair]]) < PRESENCE_THRESHOLD:
            raise ValueError('Saved scene bond not present in fixed pair predictions')
        row, i = pair_rows[pair], bond['id']
        if row in scene_rows: raise ValueError('Duplicate compiled pair')
        scene_rows[row] = i
        for target, types in ((raw, raw_types), (legacy, legacy_types)):
            label = BOND_TYPES[int(types[row])]
            target['bond_objects'][i].update(chemical_type=label, display_type=label)
    raw, _ = apply_aromatic_consistency(raw, 'demote-noncycle-clear-atoms')
    legacy, _ = apply_aromatic_consistency(legacy, 'demote-noncycle-clear-atoms')
    if legacy != scene: raise ValueError('CPU reconstruction differs from saved legacy scene')
    graph, proposed, cycles = nx.Graph(), [], 0
    for row in torch.where(presence >= PRESENCE_THRESHOLD)[0].tolist(): graph.add_edge(*pairs[row].tolist())
    aromatic = chemical_logits.softmax(-1)[:, BOND_TYPES.index('AROMATIC')]
    unsupported = []
    for cycle in nx.chordless_cycles(graph, length_bound=6):
        if len(cycle) not in (5, 6): continue
        cycles += 1
        if cycles > MAX_PROPOSALS: raise ValueError('Candidate cycle enumeration cap')
        rows = [pair_rows[tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)])))] for i in range(len(cycle))]
        values = aromatic[rows]
        if float(values.mean()) < MEAN_THRESHOLD or float(values.min()) < MINIMUM_THRESHOLD: continue
        record = dict(pair_rows=sorted(rows), candidate_atoms=sorted(cycle),
                      mean_aromatic_probability=float(values.mean()), minimum_aromatic_probability=float(values.min()))
        if any(r not in scene_rows for r in rows):
            unsupported.append(record)
        else:
            proposed.append(dict(**record, bond_ids=sorted(scene_rows[r] for r in rows)))
    # No outcome-based threshold or CPU/GPU numerical tolerance adjustment.
    if len(proposed) + len(unsupported) != legacy_diagnostics['accepted_cycles']:
        raise ValueError('Proposal enumeration differs from historical rule')
    union = {r for p in [*proposed, *unsupported] for r in p['pair_rows']}
    rebuilt = raw_types.clone()
    if union: rebuilt[sorted(union)] = BOND_TYPES.index('AROMATIC')
    if not torch.equal(rebuilt, legacy_types): raise ValueError('Promotion provenance differs from historical types')
    if scene_digest(scene) != original_hash: raise ValueError('Saved input mutated')
    return raw, sorted(proposed, key=lambda p: tuple(p['bond_ids'])), dict(
        legacy_scene_exact=True, raw_scene_sha256=scene_digest(raw),
        legacy_diagnostics=legacy_diagnostics, unsupported_proposals=unsupported,
        use_legacy=bool(unsupported), saved_scene_sha256=original_hash)
