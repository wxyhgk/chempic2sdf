"""Fixed58 saved-candidate perturbation preparation; never loads/trains a model."""

import argparse
from collections import Counter
from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path
import struct
import sys

import torch

from chemvector.bond_object import BOND_TYPES, BOND_TYPE_TO_ID
from chemvector.training.curriculum import pair_anchor_augmentation as aug
from chemvector.training.curriculum import runtime, supervision_contract as supervision
from tools import audit_pair_supervision_wiring as wiring
from tools import audit_selector_stable_subset as source

ROOT = source.ROOT
OUTPUT = ROOT / 'outputs/audits/chemvector-pair-anchor-augmentation58-20260924'
POSTCHECK = OUTPUT.with_name(OUTPUT.name + '-postcheck')
MANIFEST_SHA = 'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111'
ADMISSION_SHA = '77c4b42da2f675e61924ee7904032deb05d30b8de7767f92bfd9888aed1e85f9'
WIRING_SHA = '3699d01515776b5e37996ae8f9e2089a18af0bd847aca32f4eb725cf41d0fe83'
REASONS = ('outside_image', 'outside_owner_radius', 'not_unique_nearest', 'unmatched_owner')


def preflight(inputs):
    _, manifest, rows, bindings = source.preflight(inputs)
    if inputs.hashes[str(source.MANIFEST)] != MANIFEST_SHA:
        raise ValueError('Fixed58 manifest changed')
    admission = inputs.json(source.MANIFEST.with_name('admission.json'), ADMISSION_SHA)
    previous = inputs.json(wiring.OUTPUT / 'report.json', WIRING_SHA)
    files = [Path(__file__), Path(aug.__file__), Path(supervision.__file__), Path(wiring.__file__),
             Path(source.__file__), Path(runtime.__file__), Path(wiring.trace.__file__),
             Path(source.groups.__file__), Path(wiring.supervision.__file__),
             Path(inspect.getfile(wiring.match_atom_targets)), Path(inspect.getfile(aug.direct_atom_pair_loss)),
             ROOT / 'tests/chemvector/test_pair_anchor_augmentation.py',
             ROOT / 'tests/chemvector/test_pair_anchor_preparation.py',
             ROOT / 'docs/model/training/20260924-pair-anchor-augmentation-protocol.md']
    for path in files:
        inputs.read(path, previous['input_sha256'].get(str(path), bindings.get(str(path))))
    contexts = []
    for ordinal, row in enumerate(rows):
        key = row['sample_key']
        base = Path(manifest['sources'][row['source']]) / 'samples'
        meta = inputs.json(base / (key + '.meta.json'), row['meta_sha256'])
        layers, graph = [inputs.json(base / (key + suffix), meta['sha256'][suffix])
                         for suffix in ('.layers.json', '.graph.json')]
        paths = [source.CONTROL / (prefix + '--' + key + '.json') for prefix in ('geometry', 'baseline')]
        geometry, record = [inputs.json(path, bindings[str(path)]) for path in paths]
        saved = inputs.json(wiring.OUTPUT / (key + '.json'), previous['case_sha256'][key + '.json'])
        if (layers['sample_key'] != key or record['sample_key'] != key or saved['sample_key'] != key
                or graph['molecule_id'] != row['molecule_id'] or layers['molecule_id'] != row['molecule_id']
                or geometry['key'] != row['source'] + ':' + key or record['key'] != geometry['key']
                or geometry['image_size'] != 768 or geometry['match_radius_px'] != 6
                or saved['seed'] != 20260921 + ordinal):
            raise ValueError('Source identity or original geometry contract changed')
        fields = ('bond_id', 'begin_atom', 'end_atom', 'chemical_type')
        if ([[b[k] for k in fields] for b in layers['bonds']] != [[b[k] for k in fields] for b in graph['bonds']]
                or [(a['atom_id'], a['element']) for a in layers['atoms']]
                != [(a['atom_id'], a['element']) for a in graph['atoms']]
                or [a['atom_id'] for a in graph['atoms']] != list(range(len(graph['atoms'])))):
            raise ValueError('Graph and layer targets differ')
        contexts.append((row, layers, graph, geometry, record, saved))
    return admission, contexts


def counts(rows):
    result = Counter(rows=len(rows), positive=sum(r['presence'] for r in rows),
                     negative=sum(not r['presence'] for r in rows))
    result.update({kind: sum(r['presence'] and r['kind'] == kind for r in rows) for kind in BOND_TYPES})
    result.update({'PO_' + kind: sum(r.get('po', False) and r['kind'] == kind for r in rows)
                   for kind in ('SINGLE', 'DOUBLE')})
    return dict(result)


def inventory(graph):
    return [dict(bond_id=b['bond_id'], kind=b['chemical_type'], presence=True,
                 po=sorted(graph['atoms'][i]['element'] for i in (b['begin_atom'], b['end_atom'])) == ['O', 'P'])
            for b in graph['bonds']]


def prepare_case(context):
    row, layers, graph, geometry, record, saved = context
    pairs, _, mapping, atoms, labels = wiring.tensors(geometry, record, layers)
    bonds = layers['bonds']
    targets = dict(atom_mask=torch.ones(1, len(layers['atoms']), dtype=torch.bool),
        bond_mask=torch.ones(1, len(bonds), dtype=torch.bool),
        bond_atom_refs=torch.tensor([[[b['begin_atom'], b['end_atom']] for b in bonds]], dtype=torch.long).reshape(1, len(bonds), 2),
        bond_chemical=torch.tensor([[BOND_TYPE_TO_ID[b['chemical_type']] for b in bonds]], dtype=torch.long))
    contract = supervision.pair_targets_with_unknown(atoms, mapping, targets)
    if (not torch.equal(pairs, contract['pair_indices']) or any(not torch.equal(labels[k], contract['legacy_targets'][k])
            for k in ('presence', 'chemical', 'structured_negative'))):
        raise ValueError('Original label wiring differs')
    before_rng = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(saved['seed'])
        chosen = supervision.select_known_pair_rows(contract, atoms, negative_ratio=6, minimum_negatives=48)
    if not torch.equal(before_rng, torch.get_rng_state()) or chosen.tolist() != saved['policies']['matched-only-v1']['sampled_row_ids']:
        raise ValueError('Historical sampled rows or random state changed')
    anchors, selected, owners = atoms['atom_geometry'][0, :, :2], atoms['atom_mask'][0], mapping[0]
    target_xy = torch.tensor(geometry['target_anchors'], dtype=torch.float32)
    views = aug.plan_views(anchors, selected, owners, target_xy, pairs, contract['known_mask'], chosen)
    by_ends = {tuple(sorted((b['begin_atom'], b['end_atom']))): b for b in bonds}
    items, target_items = [], inventory(graph)
    target_by_id = {b['bond_id']: b for b in target_items}
    for rid in chosen.tolist():
        candidate_pair = pairs[rid].tolist()
        endpoint_ids = owners[pairs[rid]].tolist()
        bond = by_ends.get(tuple(sorted(endpoint_ids)))
        present, kind_id = bool(labels['presence'][rid]), int(labels['chemical'][rid])
        if present != (bond is not None) or present and kind_id != BOND_TYPE_TO_ID[bond['chemical_type']]:
            raise ValueError('Source graph label mismatch')
        items.append(dict(row_id=rid, candidates=candidate_pair, owner_targets=endpoint_ids,
            presence=present, chemical=kind_id, kind=BOND_TYPES[kind_id] if present else None,
            bond_id=bond['bond_id'] if bond else None, po=target_by_id[bond['bond_id']]['po'] if bond else False))
    by_row = {r['row_id']: r for r in items}
    view_records = []
    for view in views:
        kept = view['rows'].tolist()
        flags = aug.other_nodes_in_ribbon(view['anchors'], pairs[view['rows']], owners, target_xy)
        exclusions = {reason: chosen[view['exclusion_masks'][reason][chosen]].tolist() for reason in REASONS}
        view_records.append(dict(offset_pixels=view['offset_pixels'], row_ids=kept,
            counts=counts([by_row[r] for r in kept]), guard_reason_row_ids=exclusions,
            ribbon_other_node_row_ids=[rid for rid, flag in zip(kept, flags.tolist()) if flag]))
    extra_count = Counter(rid for view in view_records[1:] for rid in view['row_ids'])
    return dict(sample_key=row['sample_key'], molecule_id=row['molecule_id'], group=row['fit_domain'],
        seed=saved['seed'], coverage=contract['coverage'], target_counts=counts(target_items),
        source_target_bonds=target_items, sampled_rows=items, views=view_records,
        additional_view_histogram={str(n): counts([r for r in items if extra_count[r['row_id']] == n]) for n in range(9)},
        base_rows_exact=True, unknown_sampled_rows=0, empty_base_branch=not items)


def summarize(cases):
    target, coverage = Counter(), Counter()
    per_view = [dict(offset_pixels=list(offset), counts=Counter(), excluded_counts=Counter(),
                    reason_counts={reason: Counter() for reason in REASONS}, ribbon_counts=Counter()) for offset in aug.OFFSETS]
    histogram = {str(n): Counter() for n in range(9)}
    for case in cases:
        target.update(case['target_counts']); coverage.update(case['coverage'])
        rows = {r['row_id']: r for r in case['sampled_rows']}
        for entry, view in zip(per_view, case['views']):
            entry['counts'].update(view['counts'])
            entry['excluded_counts'].update(counts([r for rid, r in rows.items() if rid not in set(view['row_ids'])]))
            entry['ribbon_counts'].update(counts([rows[rid] for rid in view['ribbon_other_node_row_ids']]))
            for reason in REASONS:
                entry['reason_counts'][reason].update(counts([rows[rid] for rid in view['guard_reason_row_ids'][reason]]))
        for n in histogram: histogram[n].update(case['additional_view_histogram'][n])
    return dict(images=len(cases), structures=len({c['molecule_id'] for c in cases}), target_counts=target,
        coverage=coverage, views=per_view, additional_view_histogram=histogram,
        empty_base_branches=sum(c['empty_base_branch'] for c in cases),
        po_cases=[dict(sample_key=c['sample_key'], molecule_id=c['molecule_id'], group=c['group'],
            target_PO_SINGLE=c['target_counts']['PO_SINGLE'], target_PO_DOUBLE=c['target_counts']['PO_DOUBLE'],
            bonds=[dict(bond_id=r['bond_id'], kind=r['kind'],
                additional_views=sum(r['row_id'] in v['row_ids'] for v in c['views'][1:]))
                for r in c['sampled_rows'] if r['po']]) for c in cases if c['target_counts']['PO_SINGLE'] or c['target_counts']['PO_DOUBLE']])


def check_scalar_geometry(case, context):
    """Independent scalar/FP32 replay, without plan_views or the ribbon helper."""
    _, _, _, geometry, record, _ = context
    selected = wiring.trace.selected_candidates(geometry, record)
    original = {c['slot']: c['geometry'][:2] for c in selected}
    targets = geometry['target_anchors']
    f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
    for view_index, view in enumerate(case['views']):
        dx, dy = view['offset_pixels']
        points = {slot: [f32(f32(x) + f32(dx/768)), f32(f32(y) + f32(dy/768))] for slot, (x, y) in original.items()}
        kept, ribbons, reasons = [], [], {reason: [] for reason in REASONS}
        for row in case['sampled_rows']:
            a, b = [points[s] for s in row['candidates']]
            rejected = set()
            for point, owner in zip((a, b), row['owner_targets']):
                distances = [sum(((p-f32(t))*768)**2 for p, t in zip(point, target)) for target in targets]
                if any(p < 0 or p > 1 for p in point): rejected.add('outside_image')
                if distances[owner] > 36: rejected.add('outside_owner_radius')
                if any(distances[owner] >= d for i, d in enumerate(distances) if i != owner): rejected.add('not_unique_nearest')
                if owner < 0: rejected.add('unmatched_owner')
            for reason in rejected: reasons[reason].append(row['row_id'])
            if view_index and rejected: continue
            kept.append(row['row_id'])
            delta = [(y-x)*768 for x, y in zip(a, b)]
            squared = max(sum(d*d for d in delta), 1e-12)
            for i, target in enumerate(targets):
                if i in row['owner_targets']: continue
                rel = [(f32(t)-p)*768 for t, p in zip(target, a)]
                fraction = sum(r*d for r, d in zip(rel, delta))/squared
                transverse = abs(rel[0]*delta[1]-rel[1]*delta[0])/math.sqrt(squared)
                if .04 <= fraction <= .96 and transverse <= 12:
                    ribbons.append(row['row_id']); break
        if kept != view['row_ids'] or ribbons != view['ribbon_other_node_row_ids'] or reasons != view['guard_reason_row_ids']:
            raise ValueError('Independent scalar geometry replay differs')


def worker(postcheck=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only saved-candidate preparation required')
    torch.set_num_threads(2)
    inputs = source.Inputs()
    if postcheck:
        report = inputs.json(OUTPUT / 'report.json')
        for path, digest in report['input_sha256'].items(): inputs.read(path, digest)
        inputs.read(OUTPUT / 'plan.json', report['plan_sha256'])
    admission, contexts = preflight(inputs)
    if not postcheck:
        source.write(OUTPUT / 'plan.json', dict(schema_version='pair-anchor-preparation-plan-0.1',
            sample_keys=[c[0]['sample_key'] for c in contexts], wall_seconds=120,
            offsets_pixels=aug.OFFSETS, image_size=768, owner_radius_px=6,
            center_policy='preserve_every_original_sampled_row', additional_policy='same_owner_strict_nearest_inside_radius_and_image',
            label_policy='original_correspondence_and_chemical_type; unknown_never_negative',
            ribbon_review='informational_only; never_delete_hard_negatives',
            loss_proposal='original_typed_loss_plus_one_safe_shifted_typed_loss; no_logit_averaging',
            future_shift_schedule='cycle_eight_offsets_by_attempt; no_best_view_search_or_resampling_if_empty',
            views_are_independent_images=False, model_forwards=0, checkpoint_loads=0, optimizer_steps=0,
            sampler_calls=58, training_authorized=False, input_sha256=dict(inputs.hashes)))
    cases, case_hashes = [], {}
    for context in contexts:
        case = prepare_case(context)
        path = OUTPUT / (case['sample_key'] + '.json')
        if postcheck:
            stored = inputs.json(path, report['case_sha256'][path.name])
            if stored != case: raise ValueError('Source replay differs from saved case')
            check_scalar_geometry(stored, context)
        else:
            source.write(path, case)
            case_hashes[path.name] = sha256(path.read_bytes()).hexdigest()
        cases.append(case)
    summary = summarize(cases)
    base = summary['views'][0]['counts']
    if (base['positive'], base['negative'], summary['coverage']['unknown_pairs'], summary['empty_base_branches'],
            summary['target_counts']['positive'], summary['coverage']['unreachable_bonds']) != (1103, 6244, 626, 4, 1150, 47):
        raise ValueError('Historical fixed58 denominators changed')
    for path, digest in list(inputs.hashes.items()): inputs.read(path, digest)
    if torch.cuda.is_initialized(): raise ValueError('Unexpected CUDA initialization')
    if postcheck:
        if summary != report['summary']: raise ValueError('Summary replay differs')
        source.write(POSTCHECK / 'report.json', dict(schema_version='pair-anchor-preparation-postcheck-0.1',
            source_replay_cases=58, scalar_geometry_cases=58, input_sha256=inputs.hashes,
            report_sha256=inputs.hashes[str(OUTPUT / 'report.json')], model_forwards=0,
            checkpoint_loads=0, optimizer_steps=0, gpu_used=False, training_authorized=False, passed=True))
    else:
        source.write(OUTPUT / 'report.json', dict(schema_version='pair-anchor-preparation-0.1', summary=summary,
            plan_sha256=sha256((OUTPUT / 'plan.json').read_bytes()).hexdigest(),
            input_sha256=inputs.hashes, case_sha256=case_hashes, sampler_exact_replay_cases=58,
            original_rows_and_labels_preserved=True, unknown_sampled_rows=0,
            existing_admission_use=admission['use'], training_authorized=False, safe_to_launch=False,
            model_forwards=0, checkpoint_loads=0, optimizer_steps=0, gpu_used=False, training_performed=False,
            validation_sample_artifacts_read=False, sealed_test_read=False, new_identity_queries=0,
            data_release_changed=False, default_changed=False,
            blockers=['New purpose-specific data-use and run authorization; prior one-shot grants consumed',
                'Fresh execution-device-bound candidates and features before any optimizer update',
                'Budgeted opt-in launcher and complete typed-bond/nonedge/SDF protection checks',
                'Independent development acceptance not covered by this preparation'],
            limitations=['58 existing images only; repeated views are not new chemical or style diversity',
                'Targets gate augmentation identity, not predicted inputs; no rematching or new positives',
                'Geometry ribbon flag is not visible-stroke validation; visual ambiguity remains unresolved',
                'Original correspondence may be ambiguous; base loss is retained, not certified correct',
                'Joint translation only; independent endpoint angle/length errors are not covered',
                'No training-effectiveness, new accuracy, CUDA-availability or deployment claim']))
    print(json.dumps(dict(images=58, base=base, postcheck=postcheck, training_performed=False), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--postcheck', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args.postcheck)
        return 0
    command = [sys.executable, '-m', 'tools.prepare_pair_anchor_augmentation', '--worker']
    if args.postcheck: command.append('--postcheck')
    receipt = runtime.run_budgeted(command, output=POSTCHECK if args.postcheck else OUTPUT,
                                  budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
