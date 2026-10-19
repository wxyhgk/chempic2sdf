"""CPU-only, finite training exposure plan. No model or execution authority."""

from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import heapq
import json
import math


SEED = '20260924-pair-scale-v1'
BOND_TYPES = ('SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC')


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def sample_id(row):
    return row['source'] + ':' + row['sample_key']


def size_bucket(heavy_atoms):
    if type(heavy_atoms) is not int or heavy_atoms < 0:
        raise ValueError('Nonnegative heavy atom count required')
    return '00-07' if heavy_atoms < 8 else '08-20' if heavy_atoms <= 20 else '21-40' if heavy_atoms <= 40 else '41+'


def bond_bucket(counts):
    if set(counts) - set(BOND_TYPES) or any(type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError('Unsupported/invalid chemical bond counts')
    return next((kind for kind in ('TRIPLE', 'DOUBLE', 'AROMATIC', 'SINGLE') if counts.get(kind)), 'NO_BOND')


def deduplicate(rows):
    """Choose one existing released reference per RGB image; never edit source rows."""
    if not rows or len({sample_id(r) for r in rows}) != len(rows):
        raise ValueError('Nonempty distinct source records required')
    kept, aliases, pixels = [], [], {}
    for row in sorted(rows, key=sample_id):
        if row['split'] != 'train' or row['domain'] not in ('curriculum', 'full_molecule'):
            raise ValueError('Only published training references may enter the schedule')
        h = row['pixel_sha256']
        if not isinstance(h, str) or len(h) != 64 or set(h) - set('0123456789abcdef'):
            raise ValueError('Decoded RGB digest required')
        if h not in pixels:
            pixels[h] = row
            kept.append(row)
        else:
            owner = pixels[h]
            if any(row[k] != owner[k] for k in ('canonical_smiles', 'domain', 'group_key')):
                raise ValueError('Same pixels conflict in chemistry/domain/group')
            # Same molecule, different layouts/atom indexing are not merged
            # as targets: use the selected owner's entire original bundle.
            aliases.append(dict(key=sample_id(row), retained_key=sample_id(owner), pixel_sha256=h))
    return kept, aliases


def stratified_order(rows, *, salt):
    """Size/bond strata are spread over a pass; molecule styles are interleaved."""
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        stratum = size_bucket(row['heavy_atoms']) + ':' + bond_bucket(row['bond_counts'])
        buckets[stratum][row['canonical_smiles']].append(row)
    queues = {}
    for stratum, molecules in sorted(buckets.items()):
        identities = sorted(molecules, key=lambda x: digest([SEED, salt, stratum, x]))
        variants = {identity: sorted(molecules[identity], key=lambda r: digest([SEED, salt, sample_id(r)]))
                    for identity in identities}
        queues[stratum] = [variants[identity][visit]
                            for visit in range(max(map(len, variants.values())))
                            for identity in identities if visit < len(variants[identity])]
    # Weighted fair merge: retain actual population weights, not an accidental
    # long tail of one stratum. No random sampling with replacement here.
    heap = [(Fraction(1, 2 * len(q)), name, 0) for name, q in queues.items()]
    heapq.heapify(heap)
    result = []
    while heap:
        _, name, i = heapq.heappop(heap)
        result.append(queues[name][i])
        if i + 1 < len(queues[name]):
            heapq.heappush(heap, (Fraction(2 * i + 3, 2 * len(queues[name])), name, i + 1))
    return result


def exposure_plan(rows):
    """One unique-image pass plus explicit foundation replay at about 20%."""
    if not rows or len({sample_id(r) for r in rows}) != len(rows):
        raise ValueError('Distinct train-only rows required')
    if any(r['split'] != 'train' or r['domain'] not in ('curriculum', 'full_molecule') for r in rows):
        raise ValueError('Validation/quarantine must never enter training exposure')
    foundation = [r for r in rows if r['domain'] == 'curriculum']
    full = stratified_order([r for r in rows if r['domain'] == 'full_molecule'], salt='full')
    if not foundation or not full:
        raise ValueError('Both foundation and full-molecule pools required')
    slots = math.ceil(len(full) / 4)
    if len(foundation) > slots:
        raise ValueError('20% replay recipe cannot cover all foundation images; review recipe')
    foundation_order = []
    for cycle in range(math.ceil(slots / len(foundation))):
        foundation_order.extend(stratified_order(foundation, salt=f'foundation-{cycle}'))
    positions = {sample_id(r): i for i, r in enumerate(rows)}
    visits, schedule = Counter(), []
    for block in range(slots):
        # Every block starts with real complete molecules, not a foundation prefix.
        for row in [*full[4 * block:4 * block + 4], foundation_order[block]]:
            key = sample_id(row)
            visits[key] += 1
            schedule.append(dict(index=positions[key], visit=visits[key],
                                 role='coverage' if visits[key] == 1 else 'foundation_replay'))
    if set(visits) != set(positions) or any(visits[sample_id(r)] != 1 for r in full):
        raise ValueError('Full unique-image exposure invariant failed')
    return schedule


def coverage(rows, schedule):
    indices = {s['index'] for s in schedule}
    selected = [rows[i] for i in sorted(indices)]
    visits = Counter(s['index'] for s in schedule)
    return dict(attempts=len(schedule), distinct_images=len(indices),
                distinct_structures=len({r['canonical_smiles'] for r in selected}),
                images_by_domain=dict(Counter(r['domain'] for r in selected)),
                attempts_by_domain=dict(Counter(rows[s['index']]['domain'] for s in schedule)),
                images_by_size=dict(Counter(size_bucket(r['heavy_atoms']) for r in selected)),
                images_by_bond_bucket=dict(Counter(bond_bucket(r['bond_counts']) for r in selected)),
                images_by_style=dict(Counter(r['profile_id'] for r in selected)),
                bond_instances={k: sum(r['bond_counts'].get(k, 0) for r in selected) for k in BOND_TYPES},
                zero_bond_images=sum(not any(r['bond_counts'].values()) for r in selected),
                replay_attempts=sum(s['role'] == 'foundation_replay' for s in schedule),
                maximum_image_visits=max(visits.values(), default=0))


def draft_protocol(attempts):
    if type(attempts) is not int or attempts < 10000:
        raise ValueError('Scale proposal requires a full >=10k image exposure schedule')
    return dict(schema_version='pair-scale-draft-0.1', training_authorized=False,
        deployment_authorized=False, status='prepared_not_runnable',
        objective='one_complete_unique_image_pass_with_foundation_replay',
        trainable=['pair'], frozen=['vision', 'atom', 'bond', 'set_head', 'role', 'bbox', 'ocr'],
        augmentation=False, architecture_changed=False,
        optimizer=dict(name='AdamW', learning_rate=2e-5, betas=[.9, .999], epsilon=1e-8,
                       weight_decay=0., gradient_clip=1., batch_size=1),
        supervision='existing_matched_only_typed_pair; unknown_not_negative; all_original_positives',
        empty='record_coverage_and_skip_optimizer; do_not_hide_unreachable_bonds',
        mode='all_eval_pair_autograd_fp32_no_tf32_no_autocast',
        thresholds=dict(atom=-.25, bond=-.5, pair=.85, role=.85, match_radius_px=6),
        proposed_budget=dict(gpu_devices=['0'], wall_seconds=3600, maximum_invocations=1,
            automatic_retries=0, resume_supported=False, maximum_attempts=attempts,
            maximum_updates=attempts, epochs=1, cpu_cache_bytes=2 * 1024**3,
            profiling_images=64, full_dev_evaluations=5, validation_images=944),
        checkpoint_attempts=sorted({math.ceil(attempts * i / 4) for i in (1, 2, 3, 4)}),
        cache='bounded_LRU_or_streaming; frozen_features_only; no_pair_hidden_cache; no_full_corpus_cache',
        time_gate='within_same_invocation_before_optimizer; profile_and_predict_full_pass_plus_5_dev_evals; abort_if_over_budget',
        validation='existing_fixed944_unchanged; full800_primary_and_curriculum144_guard; no_sealed_samples',
        early_stop='hard_errors_immediate; full_dev_exact_or_valid_drop_at_least_5pp_at_quarter_checks',
        individual_regressions='record_rescued_and_broken; not_individual_zero_regression_stop',
        final_selection='complete_coverage_required; full_exact_strict_gain; full_valid_not_lower; full_fallback_not_higher; curriculum_exact_drop_at_most_2pp',
        no_improvement='initial_retained_at_end; no_automatic_second_epoch',
        interpretation='development_comparison_not_independent_test; pair_only_does_not_train_element_OCR_stereo_or_missing_endpoints')
