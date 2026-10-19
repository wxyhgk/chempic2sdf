"""Read-only pairing and extra geometry serialization; never affects predictions."""

import json

import torch

from chemvector.train_direct_atom_pair_bond import match_atom_targets
from .selector_retention_stage import index_rows, compare, THRESHOLD


def reference_updates(root):
    events = [json.loads(line) for line in (root / 'steps.jsonl').read_text().splitlines()]
    reserved = [(r['step'], r['key']) for r in events if r['event'] == 'step_reserved']
    updated = [(r['step'], r['key']) for r in events if r['event'] == 'update']
    if reserved != updated or [s for s, _ in updated] != list(range(1, 401)):
        raise ValueError('Reference update sequence incomplete')
    return dict(updated)


def baseline_parity(record, previous):
    if any(record[k] != previous[k] for k in ('key', 'sample_key', 'group', 'metrics')):
        raise ValueError('Fresh initial baseline differs from reference')
    a, b = record['prediction'], previous['prediction']
    if (a is None) != (b is None) or (a is not None and any(a[k] != b[k]
            for k in ('predicted_smiles', 'expected_smiles', 'valid', 'exact', 'fallback'))):
        raise ValueError('Fresh initial SDF baseline differs from reference')


@torch.no_grad()
def geometry(cache, config):
    raw, labels, batch = cache['raw'], cache['labels'], cache['batch']
    slots = raw['atom_mask'][0].nonzero().flatten()
    targets = batch['atom_anchors'][0, batch['atom_mask'][0]]
    distances = torch.cdist(raw['atom_geometry'][0, slots, :2].float(), targets.float()) * config.image_size
    nearest_distance, nearest_target = distances.min(-1)
    return dict(key=cache['key'], image_size=config.image_size, match_radius_px=config.atom_match_radius_px,
        coordinate_space='normalized image coordinates; target slots follow collated atom_ids',
        target_anchors=targets.tolist(), candidates=[dict(slot=int(s), grid_id=int(raw['atom_indices'][0, s]),
            geometry=raw['atom_geometry'][0, s].tolist(), raw_score=float(raw['atom_scores'][0, s]),
            raw_target=int(labels['atom_target_indices'][0, s]),
            nearest_target=int(nearest_target[i]), nearest_distance_px=float(nearest_distance[i]))
            for i, s in enumerate(slots)])


@torch.no_grad()
def matching(cache, record, config):
    raw, batch = cache['raw'], cache['batch']
    selected = torch.zeros_like(raw['atom_mask'])
    for slot, logit in record['metrics']['logits'].items():
        selected[0, int(slot)] = logit >= THRESHOLD
    mapping, matched = match_atom_targets({**raw, 'atom_mask': raw['atom_mask'] & selected}, batch, config)
    if matched[0].nonzero().flatten().tolist() != record['metrics']['matched_target_ids']:
        raise ValueError('Evidence matching differs from assessment')
    return dict(selected_target_by_slot=mapping[0].tolist())


def summary(rows):
    index_rows(rows)
    return dict(images=len(rows), complete_endpoints=sum(r['metrics']['complete_endpoints'] for r in rows),
        complete_without_extra=sum(r['metrics']['complete_endpoints'] and r['metrics']['unmatched_selected'] == 0 for r in rows),
        matched_atoms=sum(len(r['metrics']['matched_target_ids']) for r in rows),
        reachable_bonds=sum(len(r['metrics']['reachable_bond_ids']) for r in rows),
        raw_false_selections=sum(len(r['metrics']['selected_negative_slots']) for r in rows),
        unmatched_selected=sum(r['metrics']['unmatched_selected'] for r in rows),
        strict_sdf_exact=None if all(r['prediction'] is None for r in rows)
            else sum(r['prediction']['exact'] for r in rows if r['prediction'] is not None))


def paired_reference(root, current, *, step, development=False):
    if step not in (100, 200, 400) or (development and step != 400):
        raise ValueError('No equal-step reference for this panel; do not compare unequal new64 endpoints')
    prefix = 'final-development' if development else f'check-{step:06d}'
    previous = [json.loads((root / (prefix + '--' + r['sample_key'] + '.json')).read_text()) for r in current]
    differences = compare(previous, current)
    return dict(step=step, reference=summary(previous), control=summary(current),
        reference_to_control=differences,
        interpretation='Equal-step fixed-run ablation; single seed and observed development evidence, not independent SDF accuracy')
