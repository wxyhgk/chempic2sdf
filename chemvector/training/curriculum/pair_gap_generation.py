"""One fixed CPU quarantine generation; no model, identity service or training."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from . import pair_gap_data
from .interface_diagnostic import write_json
from .runtime import ResourceBudget, run_budgeted

RELEASE = Path('outputs/datasets/chemvector-expanded-release-20260915')
DATA = Path('outputs/datasets/chemvector-pair-gap-quarantine-20260918')
OUTPUT = Path('outputs/audits/chemvector-pair-gap-generation-20260918')


def bind_digest(inputs, path, digest):
    """Extending a binding must never replace a prior hash commitment."""
    name = str(path)
    if name in inputs and inputs[name] != digest:
        raise ValueError('Previously bound source changed')
    inputs.setdefault(name, digest)


def run(output):
    import torch
    from Data.sdf.chemvector_curriculum_v0_1 import build, catalog, rendering, typography, gallery
    from . import data, preflight
    from .checkpoints import file_digest
    from .release import verify_release

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only supervised generation required')
    torch.set_num_threads(2)
    output = Path(output)
    if not output.is_dir() or DATA.exists() or (output / 'plan.json').exists():
        raise ValueError('Fresh supervised output and fresh dataset required')
    manifest, _, bindings = verify_release(RELEASE / 'manifest.json', RELEASE / 'admission.json')
    entries = pair_gap_data.candidate_entries()
    controls = pair_gap_data.control_rows(manifest)
    local = pair_gap_data.local_identity_review(entries, manifest)
    if len(entries) != 18 or not local['local_clear']:
        raise ValueError('Fixed 18 candidates must be locally distinct from the whole release')
    files = [RELEASE / 'manifest.json', RELEASE / 'admission.json', Path(__file__),
             Path(pair_gap_data.__file__), *(Path(m.__file__) for m in (build, catalog, rendering, typography, gallery, data, preflight))]
    inputs = {str(p): file_digest(p) for p in files}
    for row in controls:
        p = Path(manifest['sources'][row['source']]) / 'samples' / (row['sample_key'] + '.meta.json')
        if p.is_symlink() or file_digest(p) != row['meta_sha256']:
            raise ValueError('Existing control metadata changed')
        inputs[str(p)] = row['meta_sha256']
    profile = rendering.Profile()
    font = rendering.font_record(profile.font_family)
    inputs[font['path']] = font['sha256']
    plan = {'schema_version': 'chemvector-pair-gap-generation-plan-0.1', 'release_bindings': bindings,
        'entries': entries, 'local_identity_review': local, 'existing_training_controls': controls,
        'profile': asdict(profile), 'typography': 'ink-aware-v1', 'font': font,
        'input_sha256': inputs, 'output_dataset': str(DATA), 'cpu_wall_seconds': 300,
        'render_workers': 2, 'per_image_timeout_seconds': 30, 'planned_new_images': 18,
        'planned_existing_control_reads': 7, 'model_forwards': 0, 'optimizer_steps': 0,
        'training_authorized': False, 'external_identity_queries': 0,
        'scope': 'Hand-authored quarantine candidates; only already released identities used for local exclusion',
        'failure_policy': 'Keep all rejects and adapter failures; no retries, replacements, style or threshold sweeps'}
    write_json(output / 'plan.json', plan)
    write_json(output / 'candidate-identities.json', {
        'schema_version': 'chemvector-pair-gap-identity-handoff-0.1', 'candidates': local['candidates'],
        'status': 'not_submitted', 'external_overlap_status': 'unchecked', 'training_authorized': False,
        'note': 'No private/public identity service was queried; previous batch authorization does not apply'})
    summary = build.build(DATA, entries=entries, selected_profiles=[profile], typography='ink-aware-v1',
                          workers=2, task_timeout=30, gallery=True)
    compatible = preflight.run(DATA, output / 'target-preflight') if summary['accepted_images'] else None
    checked = {} if compatible is None else {r['sample_key']: r for r in json.loads(
        (output / 'target-preflight' / 'samples.json').read_text())}
    rows = [json.loads(line) for line in (DATA / 'samples.jsonl').read_text().splitlines() if line.strip()]
    records, refusals = [], 0
    for row in rows:
        sample = data.load_sample(DATA, row['sample_key'], purpose='diagnostic')
        if (sample['meta']['split'] != 'quarantine' or sample['meta']['training_authorized'] is not False
                or sample['meta']['external_overlap_status'] != 'unchecked'
                or any(s['training_authorized'] is not False for s in (sample['graph'], sample['layers']))):
            raise ValueError('Generated data must stay quarantined')
        try:
            data.load_sample(DATA, row['sample_key'], purpose='training')
        except ValueError as exc:
            if 'not admitted for training' not in str(exc):
                raise
            refusals += 1
        else:
            raise ValueError('Quarantine escaped the training loader')
        records.append({**pair_gap_data.target_coverage(sample), 'origin': 'new_quarantine',
                        'compatible': checked[row['sample_key']]['compatible']})
    for row in controls:
        root = Path(manifest['sources'][row['source']])
        meta_path = root / 'samples' / (row['sample_key'] + '.meta.json')
        if file_digest(meta_path) != row['meta_sha256']:
            raise ValueError('Existing control metadata changed since the plan')
        sample = data.load_sample(root, row['sample_key'], purpose='diagnostic')
        if (sample['meta']['split'] != 'train' or sample['meta']['training_authorized'] is not True
                or sample['meta']['external_overlap_status'] != 'checked'
                or sample['meta']['molecule_id'] != row['molecule_id']
                or sample['graph']['canonical_smiles'] != row['canonical_smiles']
                or sample['meta']['profile_id'] != 'reference'):
            raise ValueError('Existing control must remain the same admitted training sample')
        for suffix in (*data.PAYLOADS, '.meta.json'):
            p = root / 'samples' / (row['sample_key'] + suffix)
            bind_digest(inputs, p, file_digest(p))
        batch = data.CurriculumCollator(purpose='diagnostic')([sample])
        records.append({**pair_gap_data.target_coverage(sample), 'origin': 'existing_train',
                        'compatible': batch['spatial_cell_collision_pairs'][0] == 0})
    coverage = pair_gap_data.coverage_summary(records)
    new = [r for r in records if r['origin'] == 'new_quarantine']
    by_concept = Counter(c for e in entries for c in e['concepts'] if c != 'pair_gap_quarantine')
    if torch.cuda.is_initialized():
        raise ValueError('CPU generation initialized CUDA')
    for path, digest in inputs.items():
        if file_digest(path) != digest:
            raise ValueError('Bound source changed during generation')
    write_json(output / 'target-coverage.json', records)
    # This mixed-origin reference list is intentionally NOT a training manifest.
    write_json(output / 'inspection-panel.json', {'schema_version': 'chemvector-pair-inspection-panel-0.1',
        'new_quarantine_root': str(DATA), 'new_samples': rows, 'existing_training_controls': controls,
        'training_authorized': False, 'purpose': 'Coverage review only; no training admission'})
    report = {'schema_version': 'chemvector-pair-gap-generation-result-0.1',
        'plan_sha256': file_digest(output / 'plan.json'), 'verified_input_sha256': inputs,
        'generated': True, 'planned_new_molecules': len(entries), 'planned_by_concept': dict(by_concept),
        'accepted_new_images': summary['accepted_images'], 'rejected_new_images': summary['rejected_images'],
        'compatible_new_images': sum(r['compatible'] for r in new), 'existing_control_images': len(controls),
        'all_planned_targets_compatible': len(new) == len(entries) and all(r['compatible'] for r in records),
        **coverage, 'local_release_identity_clear': local['local_clear'], 'external_overlap_status': 'unchecked',
        'training_loader_refusals': refusals, 'gpu_used': False, 'model_forwards': 0, 'optimizer_steps': 0,
        'training_performed': False, 'deployed': False, 'external_identity_queries': 0,
        'dataset': str(DATA), 'gallery': str(DATA / 'index.html'),
        'artifact_sha256': {str(p): file_digest(p) for p in (DATA / 'molecules.json', DATA / 'summary.json',
                           DATA / 'samples.jsonl', DATA / 'rejected.jsonl', DATA / 'sample-sha256.json',
                           output / 'candidate-identities.json', output / 'target-coverage.json',
                           output / 'inspection-panel.json')},
        'limits': ['No new training/evaluation split; all new identities and pixels remain quarantine',
                   'Local release exclusion does not check external/sealed overlap',
                   'N--S positives are sulfonamides/sulfamides, not all sulfur oxidation states',
                   'One reference style per identity; no claim of real-world style coverage',
                   'Coverage and adapter success are not model reachability or recognition accuracy',
                   'Independent identity admission, predicted reachability and a fresh budget remain required']}
    write_json(output / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('accepted_new_images', 'rejected_new_images',
        'compatible_new_images', 'all_planned_targets_compatible', 'minimal_target_coverage_passed', 'training_ready')}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        run(OUTPUT)
        return 0
    receipt = run_budgeted([sys.executable, '-m', 'chemvector.training.curriculum.pair_gap_generation', '--worker'],
                           output=OUTPUT, budget=ResourceBudget(300))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
