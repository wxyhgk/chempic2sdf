"""Prediction-only gates and separate scoring for a fixed offline safety audit.

This adds no production policy. Unknown correspondence is never a negative.
"""

from collections import Counter

import torch

from chemvector.bond_object import BOND_TYPES
from chemvector.experiments.selector_recheck import CONFIG
from chemvector.experiments.single_candidate_replay import first_review_candidate, scene_state
from chemvector.train_direct_atom_pair_bond import match_atom_targets
from chemvector.training.curriculum.supervision_contract import pair_targets_with_unknown
from chemvector.training.curriculum.validation_truth import canonical_molblock


def structural_prefilter(trace):
    """Exact necessary conditions; no GT, pixel geometry or unseen raw guesses."""
    atoms, edges = scene_state(trace)
    orphans = sorted(atoms.keys() - {s for pair in edges for s in pair})
    status = ('insufficient_predicted_length_reference' if len(edges) < CONFIG['minimum_observed_edges']
              else 'no_orphan' if not orphans else 'needs_raw_replay')
    return dict(status=status, selected_slots=sorted(atoms), orphan_slots=orphans, edges=len(edges))


def intervention_decision(queue):
    """Keep last round's single-request/single-first-candidate scope, or abstain."""
    if queue['status'] != 'requests_ready':
        return dict(slot=None, reason=queue['status'])
    if len(queue['requests']) != 1:
        return dict(slot=None, reason='multiple_requests_outside_single_candidate_scope')
    if queue['requests'][0]['first_review_slot'] is None:
        return dict(slot=None, reason='tied_ranking_abstain')
    return dict(slot=first_review_candidate(queue), reason='single_first_candidate_counterfactual')


def score_prediction(record, atoms, targets, config, expected):
    """Read-only scoring after prediction persistence; never called by a gate."""
    mapping, matched = match_atom_targets(atoms, targets, config)
    contract = pair_targets_with_unknown(atoms, mapping, targets)
    pairs = torch.tensor(record['pairs'], dtype=torch.long).reshape(-1, 2)
    if not torch.equal(pairs, contract['pair_indices']):
        raise ValueError('Saved pair lineage differs from scoring candidates')
    labels, known = contract['legacy_targets'], contract['known_mask']
    probability = torch.tensor(record['presence_logits']).sigmoid()
    logits = torch.tensor(record['chemical_logits']).reshape(-1, len(BOND_TYPES))
    if not bool(torch.isfinite(probability).all() and torch.isfinite(logits).all()):
        raise ValueError('Finite saved predictions required')
    present = probability >= .85  # audited original provenance; not a tunable threshold
    raw_type = logits.argmax(-1)
    lookup = {tuple(sorted(ref)): i for i, ref in enumerate(
        targets['bond_atom_refs'][0, targets['bond_mask'][0]].tolist())}
    slots = atoms['atom_mask'][0].nonzero().flatten().tolist()
    assignment = {s: int(mapping[0, s]) for s in slots}
    correct_raw = []
    for i, pair in enumerate(pairs.tolist()):
        if bool(present[i] and labels['presence'][i] and raw_type[i] == labels['chemical'][i]):
            correct_raw.append(lookup[tuple(sorted(assignment[s] for s in pair))])
    scene_edges = None
    if 'scene_before_fallback' in record['trace']:
        _, edges = scene_state(record['trace'])
        scene_edges = []
        for pair, bond in edges.items():
            refs = [assignment[s] for s in pair]
            bid = lookup.get(tuple(sorted(refs))) if min(refs) >= 0 else None
            if min(refs) < 0:
                status = 'unknown_correspondence'
            elif bid is None:
                status = 'confirmed_nonedge'
            elif bond['chemical_type'] != BOND_TYPES[int(targets['bond_chemical'][0, bid])]:
                status = 'wrong_type'
            else:
                status = 'correct_typed'
            scene_edges.append(dict(pair=list(pair), mapped_targets=refs,
                target_bond_id=bid, type=bond['chemical_type'], status=status))
    identity = canonical_molblock(record['prediction']['molblock'])
    return dict(coverage=contract['coverage'], assignment=assignment,
        matched_targets=matched[0].nonzero().flatten().tolist(),
        correct_raw_typed_bonds=sorted(correct_raw),
        confirmed_false_positives=int((present & known & ~labels['presence']).sum()),
        unknown_present=int((present & ~known).sum()),
        scene_edges=scene_edges, scene_counts=dict(Counter(r['status'] for r in scene_edges or [])),
        sdf_valid=identity is not None, sdf_exact=identity is not None and identity == expected,
        fallback=record['prediction']['fallback'], predicted_smiles=identity)


def score_first_candidates(queue, raw, targets, baseline_score, radius=6):
    """Position evidence only, not a claim that far-away proposals are strokes."""
    expected = targets['atom_anchors'][0, targets['atom_mask'][0]].double()
    matched = set(baseline_score['matched_targets'])
    results = []
    for request in queue['requests']:
        slot = request['first_review_slot']
        if slot is None:
            continue
        # Direct distance avoids matrix-size-dependent cdist arithmetic.
        distances = (expected - raw['atom_geometry'][0, slot, :2].double()).norm(dim=1) * 768
        distance, nearest = distances.min(0)
        within = float(distance) <= radius
        results.append(dict(slot=slot, orphan=request['orphan_slot'], partner=request['partner_slot'],
            nearest_target=int(nearest), distance_px=float(distance), within_original_radius=within,
            nearest_target_already_matched=int(nearest) in matched,
            category=('outside_target_radius' if not within else
                      'near_already_matched_target' if int(nearest) in matched else 'near_unmatched_target')))
    return results
