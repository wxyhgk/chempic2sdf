"""Fixed two-arm single-pass experiment contract. No execution authority."""

from copy import deepcopy

from . import pair_trial_contract as original
from .pair_cached_trial import json_digest

USE = 'local_pair_anchor_ab58_once'
ARMS = ('control', 'augmentation')
LIMITS = dict(wall_seconds=600, gpu_devices=['0'], maximum_invocations=1, automatic_retries=0,
              attempts_per_arm=58, updates_per_arm=54, total_updates=108,
              upstream_forwards=58, full58_evaluations=6, sampled_parity_forwards=108,
              training_pair_forwards=162, cache_tensor_bytes=2 * 1024**3,
              checkpoint_attempts=[29, 58])


def protocol():
    return dict(schema_version='pair-anchor-ab-protocol-0.1', use=USE,
        initial_sha256=original.INITIAL_SHA, manifest_sha256=original.MANIFEST_SHA,
        original_admission_sha256=original.ADMISSION_SHA, limits=deepcopy(LIMITS),
        recipe={k: deepcopy(original.RECIPE[k]) for k in ('seed', 'image_size', 'maximum_atoms',
            'atom_match_radius_px', 'atom_threshold', 'bond_threshold', 'pair_threshold', 'role_threshold',
            'negative_ratio', 'minimum_negatives', 'learning_rate', 'betas', 'epsilon', 'weight_decay', 'gradient_clip')},
        arms=dict(control='original_typed_pair_loss', augmentation='original_typed_pair_loss + 1.0 * shifted_typed_pair_loss'),
        scope='one_manifest_pass_per_arm; paired_attempts; shared_fresh_frozen_execution_device_cache',
        initialization='two_independent_identical_initial_pair_heads_and_fresh_AdamW; all_eval; upstream_and_OCR_frozen',
        sampling='same_once_sampled_CPU_rows_per_image; seed=20260921+ordinal; no_unknown_negatives',
        shift='per_image_visit_v1; one_visit_uses_first_fixed_offset(-4,-4); no_resampling',
        gates='original_owner<=6px; unique_nearest; inside_image; no_rematching',
        evaluation='all58_all_pairs_normal_unshifted_inference; unchanged_aromatic_compiler_and_OCR',
        safety='per_bond_raw_and_promoted_correct_sets; no_new_confirmed_false_pair_ids; original_SDF_exact/valid; no_new_fallback',
        unknown='report_specific_new_unknown_present_rows; not_false_edges_or_training_negatives',
        decision='paired_checkpoints29/58; any_arm_safety_failure_stops_both; final_augmentation_must_dominate_control_and_improve',
        cache='fresh_predicted_candidates; original_device_mapping; bind_targets_and_sampled_rows; no_pair_hidden_cache',
        parity='before_updates: every_nonempty_image_weight0_adapter_equals_direct_pair_forward; no_real_backward',
        training_authorized=False, deployment_authorized=False, resume_supported=False,
        forbidden=['historical_row_id_reuse', 'threshold_tuning', 'extra_nonedge_replay', 'selector_or_OCR_updates',
                   'new_data', 'development_or_sealed_samples', 'identity_queries', 'automatic_retries_or_followup'],
        interpretation='fixed58_train_fit_only; one_offset_not_eight_direction_robustness_or_independent_accuracy')


def validate_protocol(value):
    if json_digest(value) != json_digest(protocol()):
        raise ValueError('Exact default-deny paired protocol required')


def _indexed(records):
    # Reuse strict denominator, signature, boolean and raw-bond validation.
    original.checkpoint_decision(records, records, step=50)
    result = {}
    for row in records:
        ids = row['promoted_correct_typed_bond_ids']
        for values, maximum in ((ids, row['target_bonds']),
                                (row['confirmed_false_pair_row_ids'], row['pair_count']),
                                (row['unknown_present_row_ids'], row['pair_count'])):
            if (type(maximum) is not int or maximum < 0 or not isinstance(values, list)
                    or len(values) != len(set(values))
                    or any(type(i) is not int or not 0 <= i < maximum for i in values)):
                raise ValueError('Unique in-range full-pair and promoted-bond IDs required')
        if (len(ids) > row['reachable_positive_pairs']
                or len(row['confirmed_false_pair_row_ids']) != row['confirmed_false_positives']
                or len(row['unknown_present_row_ids']) != row['unknown_present']
                or set(row['confirmed_false_pair_row_ids']) & set(row['unknown_present_row_ids'])):
            raise ValueError('Pair metric counts and identities disagree')
        result[row['key']] = row
    return result


def protections(baseline, current):
    before, after = _indexed(baseline), _indexed(current)
    checked = original.checkpoint_decision(baseline, current, step=50)
    failures = list(checked['protection_failures'])
    new_unknown = []
    for key in before:
        a, b = before[key], after[key]
        if a['pair_count'] != b['pair_count']:
            raise ValueError('Full pair denominator changed')
        lost = sorted(set(a['promoted_correct_typed_bond_ids']) - set(b['promoted_correct_typed_bond_ids']))
        new_fp = sorted(set(b['confirmed_false_pair_row_ids']) - set(a['confirmed_false_pair_row_ids']))
        if lost or new_fp:
            failures.append(dict(key=key, lost_promoted_correct_bond_ids=lost, new_confirmed_false_pair_row_ids=new_fp))
        new_unknown.append(dict(key=key, new_unknown_present_row_ids=sorted(
            set(b['unknown_present_row_ids']) - set(a['unknown_present_row_ids']))))
    return dict(passed=not failures, failures=failures, unknown_changes=new_unknown)


def paired_decision(baseline, control, augmentation, *, attempt):
    if type(attempt) is not int or attempt not in LIMITS['checkpoint_attempts']:
        raise ValueError('Only predeclared paired attempt checkpoints required')
    checks = {name: protections(baseline, rows) for name, rows in zip(ARMS, (control, augmentation))}
    if not all(c['passed'] for c in checks.values()):
        return dict(action='stop_regression', checks=checks, success=False, training_accuracy_only=True)
    if attempt != 58:
        return dict(action='continue_within_fixed_budget', checks=checks, success=False, training_accuracy_only=True)
    comparison = protections(control, augmentation)
    fields = ('correct_typed_bond_ids', 'promoted_correct_typed_bond_ids', 'sdf_exact')
    def totals(rows):
        return {k: sum(len(r[k]) if isinstance(r[k], list) else int(r[k]) for r in rows) for k in fields}
    a, b = totals(control), totals(augmentation)
    success = comparison['passed'] and all(b[k] >= a[k] for k in fields) and any(b[k] > a[k] for k in fields)
    return dict(action='complete_fit_probe_not_deployment' if success else 'stop_no_safe_control_improvement',
                checks=checks, control_comparison=comparison, control_totals=a, augmentation_totals=b,
                success=success, training_accuracy_only=True)
