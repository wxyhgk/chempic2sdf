"""Draft pair-only trial contract and CPU checks, NOT a training launcher.

No optimizer, checkpoint loading, data-use approval or execution authorization.
Historical stages/losses remain unchanged. Evaluation rules use complete fixed
cohorts, not the changing set of sampled negatives in a training loss.
"""

from collections import Counter
from copy import deepcopy
import json

import torch

from .pair_stage import FrozenUpstreamPairStage
from .pair_supervision import MATCHED_ONLY, sample_supervised_pairs
from .supervision_contract import pair_targets_with_unknown, select_known_pair_rows

INITIAL_SHA = '09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1'
MANIFEST_SHA = 'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111'
ADMISSION_SHA = '77c4b42da2f675e61924ee7904032deb05d30b8de7767f92bfd9888aed1e85f9'
USE = 'local_pair_only_matched_fit58'

RECIPE = dict(
    update_recipe='pair-only-matched-eval-v1', pair_supervision=MATCHED_ONLY,
    trainable=['pair'], frozen=['vision', 'atom', 'bond', 'set_head', 'role', 'bbox', 'ocr'],
    mode='all_eval_with_pair_autograd', dtype='float32', autocast=False, tf32=False,
    image_size=768, maximum_atoms=128, atom_match_radius_px=6,
    atom_threshold=-.25, bond_threshold=-.5, pair_threshold=.85, role_threshold=.85,
    negative_ratio=6, minimum_negatives=48, optimizer='AdamW', learning_rate=2e-5,
    betas=[.9, .999], epsilon=1e-8, weight_decay=0., gradient_clip=1., batch_size=1,
    seed=20260921, gpu_devices=['0'], wall_seconds=600, maximum_steps=200,
    maximum_batch_attempts=232, order='original_manifest_order_repeated_at_most_four_times',
    checkpoints=[0, 50, 100, 200], no_improvement_stop_step=100,
    maximum_invocations=1, automatic_retries=0, resume_supported=False,
    empty_branch='skip_without_optimizer_step; retain_image_and_target_denominators',
    zero_positive_branch='known_negatives_allowed; skip_only_if_no_known_pairs',
)

CACHE_CONTRACT = dict(
    build='fresh_on_execution_device_before_updates', maximum_upstream_forwards=58,
    storage='detached_float32_cpu_tensors; return_to_same_execution_device',
    contents=['frozen_visual_map', 'raw_candidates', 'selected_candidates', 'selector_context',
              'selected_mapping', 'all_pair_indices', 'all_pair_labels', 'known_mask'],
    bind=['initial_sha256', 'manifest_sha256', 'sample_key', 'image_sha256', 'target_sha256',
          'recipe_sha256', 'implementation_sha256', 'device_identity', 'torch_cuda_versions',
          'candidate_tensor_sha256', 'label_tensor_sha256', 'feature_tensor_sha256'],
    cache_pair_hidden=False, cache_pair_ribbon_encoder_output=False,
    reuse_historical_slot_labels=False, inject_gt_candidates=False,
    prediction_interface='pixels_and_frozen_predictions_only; labels_used_only_by_loss_and_scoring',
    device_change='abort; no_automatic_cpu_fallback_or_cache_migration',
)


def _encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def make_proposal(rows):
    if (len(rows) != 58 or Counter(r.get('fit_domain') for r in rows)
            != {'witness': 26, 'old_full_molecule': 32}
            or any(r.get('split') != 'train' for r in rows)
            or len({r['source'] + ':' + r['sample_key'] for r in rows}) != 58):
        raise ValueError('Exact fixed58 training/replay cohort required')
    return dict(schema_version='chemvector-pair-only-trial-proposal-0.1',
        status='draft_not_execution_authorization', training_authorized=False,
        data_use_approved=False, deployment_authorized=False,
        use=USE, initial_sha256=INITIAL_SHA, manifest_sha256=MANIFEST_SHA,
        original_admission_sha256=ADMISSION_SHA,
        samples=[{k: r[k] for k in ('source', 'sample_key', 'meta_sha256', 'fit_domain', 'split')}
                 for r in rows], recipe=deepcopy(RECIPE), cache=deepcopy(CACHE_CONTRACT),
        budget_scope='entire_worker_including_cache_baseline_updates_checks_and_artifact_writes',
        acceptance=dict(
            evaluation='all_58_train_replay_images_all_pairs_at_fixed_thresholds',
            primary='correct_typed_reachable_bond_ids; not_loss_or_candidate_coverage',
            protections=['every_initial_correct_typed_bond', 'every_initial_exact_SDF',
                         'every_initial_valid_SDF', 'no_new_fallback',
                         'no_increase_of_confirmed_false_positives_per_image',
                         'exact_frozen_candidate_label_and_text_signatures'],
            progress='strict_gain_in_typed_bonds_or_SDF_exact_without_any_protection_failure',
            unknown_pairs='count_predictions_separately; never_score_unknown_as_nonedge',
            full_graph='strict_SDF_scoring_keeps_unreachable_targets_and_compile_failures',
            interpretation='train_fit_feasibility_only; no_independent_accuracy_or_mask_causality_claim'),
        output=dict(format='pair_delta_bound_to_initial_protocol_cache_and_new_authorization',
                    record_supervision_policy=True, overwrite_initial=False,
                    checkpoint_selection='last_completed_check_for_diagnosis_only',
                    default_remains='initial'),
        excluded=['new_identity_queries', 'new_data', 'validation_sample_artifacts', 'sealed_tests',
                  'threshold_search', 'matching_changes', 'new_positive_labels', 'automatic_followup'])


def validate_proposal(proposal, rows):
    # Serialization is type-sensitive (False != 0, True != 1 here).
    if _encoded(proposal) != _encoded(make_proposal(rows)):
        raise ValueError('Proposal differs from the exact bounded draft; not an approval channel')


def compare_supervision(atoms, mapping, targets, *, seed):
    """Check existing opt-in stage against new runtime policy on identical rows.

    Use only original, one-to-one selected matching, never diagnostic ownership.
    No prediction, parameter change or device transfer. RNG is preserved.
    """
    if any(t.device.type != 'cpu' for t in (atoms['atom_mask'], mapping)):
        raise ValueError('Saved-candidate equivalence check is CPU only')
    contract = pair_targets_with_unknown(atoms, mapping, targets)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        existing = select_known_pair_rows(contract, atoms, negative_ratio=6, minimum_negatives=48)
        torch.manual_seed(seed)
        current, known = sample_supervised_pairs(contract['pair_indices'], contract['sample_indices'],
            mapping, atoms, contract['legacy_targets'], policy=MATCHED_ONLY,
            negative_ratio=6, minimum_negatives=48)
    if not torch.equal(known, contract['known_mask']) or not torch.equal(existing, current):
        raise ValueError('Existing pair stage and matched-only policy are not equivalent')
    positives = contract['legacy_targets']['presence']
    if (not bool(known[current].all())
            or set(positives.nonzero().flatten().tolist()) - set(current.tolist())):
        raise ValueError('Ignored pair sampled or original positive omitted')
    return dict(**contract['coverage'], sampled_rows=len(current),
        sampled_positive_rows=int(positives[current].sum()),
        sampled_negative_rows=int((~positives[current]).sum()),
        sampled_unknown_pairs=int((~known[current]).sum()), empty_branch=not len(current),
        mask_equal=True, sampled_row_ids_equal=True)


def configure_pair_stage(components, config):
    """Reuse old stage, explicitly in eval (autograd still enabled for pair).

    A caller must use fresh components; this neither creates nor authorizes an
    optimizer. Do not cache this trainable pair head's hidden/ribbon encodings.
    """
    stage = FrozenUpstreamPairStage(components, config, atom_threshold=RECIPE['atom_threshold'])
    stage.eval()
    stage.pair_parameters()
    return stage


def checkpoint_decision(baseline, current, *, step):
    """Pure preregistered per-image protections; inputs from full-cohort scoring.

    Bond IDs refer to original GT bonds in the fixed cache. They are present
    AND correctly typed under the original 6 px correspondence; raw chemistry
    correctness is distinct from compiled SDF identity. Never receive sampled
    loss rows or reinterpret diagnostic spatial ownership as target labels.
    """
    if type(step) is not int or step not in (50, 100, 200):
        raise ValueError('Only fixed positive checkpoint steps supported')

    def index(records):
        if not records or len({r['key'] for r in records}) != len(records):
            raise ValueError('Nonempty distinct full-cohort metric records required')
        result = {}
        for row in records:
            for field in ('target_bonds', 'reachable_positive_pairs', 'confirmed_false_positives', 'unknown_present'):
                if type(row[field]) is not int or row[field] < 0:
                    raise ValueError('Nonnegative integer metric required')
            ids = row['correct_typed_bond_ids']
            if (not isinstance(ids, list) or len(set(ids)) != len(ids)
                    or any(type(i) is not int or not 0 <= i < row['target_bonds'] for i in ids)
                    or not len(ids) <= row['reachable_positive_pairs'] <= row['target_bonds']):
                raise ValueError('Invalid reachable typed-bond IDs or denominator')
            for field in ('sdf_exact', 'sdf_valid', 'fallback'):
                if type(row[field]) is not bool:
                    raise ValueError('Boolean SDF metric required')
            if row['sdf_exact'] and not row['sdf_valid']:
                raise ValueError('An invalid SDF cannot be exact')
            for field in ('candidate_label_sha256', 'text_sha256'):
                value = row[field]
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('Bound frozen-candidate/label and text signatures required')
            result[row['key']] = row
        return result

    old, new = index(baseline), index(current)
    if old.keys() != new.keys():
        raise ValueError('Cannot drop empty/failed images or change the scoring cohort')
    failures, typed_gains, sdf_gains = [], 0, 0
    for key, before in old.items():
        after = new[key]
        if any(before[k] != after[k] for k in ('target_bonds', 'reachable_positive_pairs',
                                               'candidate_label_sha256', 'text_sha256')):
            raise ValueError('Frozen inputs or target denominators changed')
        lost = sorted(set(before['correct_typed_bond_ids']) - set(after['correct_typed_bond_ids']))
        checks = dict(lost_correct_typed_bonds=bool(lost),
            broken_exact_sdf=before['sdf_exact'] and not after['sdf_exact'],
            broken_valid_sdf=before['sdf_valid'] and not after['sdf_valid'],
            new_fallback=not before['fallback'] and after['fallback'],
            more_confirmed_false_positives=after['confirmed_false_positives'] > before['confirmed_false_positives'])
        if any(checks.values()):
            failures.append(dict(key=key, lost_correct_typed_bond_ids=lost,
                                 reasons=[k for k, v in checks.items() if v]))
        typed_gains += len(set(after['correct_typed_bond_ids']) - set(before['correct_typed_bond_ids']))
        sdf_gains += int(after['sdf_exact'] and not before['sdf_exact'])
    progress = bool(typed_gains or sdf_gains) and not failures
    if failures:
        action = 'stop_regression'
    elif step >= RECIPE['no_improvement_stop_step'] and not progress:
        action = 'stop_no_improvement'
    elif step == RECIPE['maximum_steps']:
        action = 'complete_fit_probe_not_deployment'
    else:
        action = 'continue_within_original_budget'
    return dict(action=action, protection_failures=failures, typed_bond_gains=typed_gains,
                sdf_exact_gains=sdf_gains, progress=progress,
                unknown_present_before=sum(r['unknown_present'] for r in old.values()),
                unknown_present_after=sum(r['unknown_present'] for r in new.values()),
                independent_accuracy_measured=False, deployment_authorized=False)
