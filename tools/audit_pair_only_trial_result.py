"""CPU saved-result postcheck; no inference, optimizer, GPU or trial retry."""

from collections import Counter
from hashlib import sha256
import json
import os

import torch

from chemvector.training.curriculum import pair_only_trial as launch, pair_cached_trial as cached
from chemvector.training.curriculum import pair_trial_contract as rules
from chemvector.training.curriculum.interface_diagnostic import tensor_tree_digest, write_json
from chemvector.training.curriculum.validation_truth import canonical_molblock

REPORT_SHA = 'd12d93e5e012f7566f4cc3123abf411b67f49c44ecb898404034105b3c4603eb'


def state_hash(state):
    digest = sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only saved-result postcheck required')
    torch.set_num_threads(2)
    manifest, _, auth = launch.authorize()
    out = launch.OUTPUT
    report = launch.read_bound(out / 'report.json', REPORT_SHA)
    runtime = json.loads((out / 'runtime.json').read_text())
    assert runtime['status'] == 'completed' and runtime['elapsed_seconds'] <= 600
    assert report['complete'] and report['optimizer_steps'] == 50 and report['stop_reason'] == 'stop_regression'
    for path, digest in report['input_sha256'].items():
        assert launch.file_digest(path) == digest, path
    rows = manifest['training_samples']
    events = [json.loads(line) for line in (out / 'events.jsonl').read_text().splitlines()]
    reserved = [r['step'] for r in events if r['event'] == 'step_reserved']
    batches = [r for r in events if r['event'] == 'batch_complete']
    updates = [r for r in batches if r['updated']]
    assert reserved == [r['step'] for r in updates] == list(range(1, 51))
    assert [r['attempt'] for r in batches] == list(range(54))
    assert sum(not r['updated'] for r in batches) == 4
    for i, event in enumerate(batches):
        row = rows[i % 58]
        assert event['key'] == row['source'] + ':' + row['sample_key']
        if event['updated']:
            assert event['sampled_unknown_pairs'] == 0
            assert len(event['sampled_row_ids']) == len(set(event['sampled_row_ids'])) == event['sampled_rows']
    assert len([r for r in events if r['event'] == 'cache_complete']) == 58
    assert len([r for r in events if r['event'] == 'evaluation_complete']) == 2
    baseline, final, artifact_hashes = [], [], {}
    for row in rows:
        key = row['source'] + ':' + row['sample_key']
        for step, records in ((0, baseline), (50, final)):
            path = out / (f'check-{step:06d}--' + row['sample_key'] + '.json')
            record = json.loads(path.read_text())
            metric = record['metrics']
            assert metric['key'] == key and metric['expected_smiles'] == row['canonical_smiles']
            canonical = canonical_molblock(record['prediction']['molblock'])
            assert (canonical is not None) == metric['sdf_valid']
            assert (canonical is not None and canonical == row['canonical_smiles']) == metric['sdf_exact']
            assert canonical == metric['predicted_smiles']
            assert metric['text_sha256'] == cached.json_digest({k: record['trace'][k] for k in ('roles', 'ocr')})
            records.append(metric)
            artifact_hashes[path.name] = launch.file_digest(path)
    assert cached.summarize(baseline) == report['baseline'] and cached.summarize(final) == report['final']
    decision = rules.checkpoint_decision(baseline, final, step=50)
    assert dict(step=50, **decision) == report['decisions'][0]
    assert launch.file_digest(out / 'cache-manifest.json') == report['cache_manifest_sha256']
    cache_rows = json.loads((out / 'cache-manifest.json').read_text())
    assert [r['key'] for r in cache_rows] == [r['key'] for r in baseline]
    for row, receipt, metric in zip(rows, cache_rows, baseline):
        path = out / ('cache--' + row['sample_key'] + '.json')
        assert json.loads(path.read_text()) == receipt
        assert receipt['candidate_label_sha256'] == metric['candidate_label_sha256']
        artifact_hashes[path.name] = launch.file_digest(path)
    initial = torch.load(launch.INITIAL, map_location='cpu', weights_only=True)
    delta = torch.load(out / 'pair-delta-000050.pt', map_location='cpu', weights_only=True)
    original = {k.removeprefix('pair.'): v for k, v in initial['model'].items() if k.startswith('pair.')}
    frozen = {k: v for k, v in initial['model'].items() if not k.startswith('pair.')}
    assert tensor_tree_digest(frozen) == report['frozen_state_before'] == report['frozen_state_after']
    assert report['ocr_state_before'] == report['ocr_state_after'] == delta['ocr_state_sha256']
    assert delta['pair_state'].keys() == original.keys()
    assert all(torch.isfinite(v).all() for v in delta['pair_state'].values())
    assert state_hash(original) == report['pair_state_before']
    assert state_hash(delta['pair_state']) == report['pair_state_after']
    assert delta['authorization_sha256'] == launch.file_digest(launch.AUTHORIZATION)
    assert delta['cache_manifest_sha256'] == report['cache_manifest_sha256']
    assert delta['protocol_sha256'] == launch.PROPOSAL_SHA and delta['initial_sha256'] == rules.INITIAL_SHA
    assert delta['pair_supervision'] == rules.RECIPE['pair_supervision']
    assert delta['optimizer_steps'] == 50 and delta['decision'] == decision
    changed = [k for k in original if not torch.equal(original[k], delta['pair_state'][k])]
    assert changed
    for path in (out / 'report.json', out / 'runtime.json', out / 'events.jsonl', out / 'cache-manifest.json',
                 out / 'plan.json', out / 'decision-000050.json', out / 'pair-delta-000050.pt'):
        artifact_hashes[path.name] = launch.file_digest(path)
    rescued = [b['key'] for a, b in zip(baseline, final) if b['sdf_exact'] and not a['sdf_exact']]
    broken = [b['key'] for a, b in zip(baseline, final) if a['sdf_exact'] and not b['sdf_exact']]
    result = dict(schema_version='pair-only-result-postcheck-0.1', passed=True,
        input_hashes_verified=len(report['input_sha256']), implementation_hashes_verified=len(auth['implementation_sha256']),
        artifact_sha256=artifact_hashes, optimizer_intents_and_completions=50,
        sdf_records_recomputed=116, changed_pair_tensors=len(changed), total_pair_tensors=len(original),
        frozen_digest_matches_initial=True, recorded_frozen_and_ocr_hashes_unchanged=True,
        rescued_sdf_keys=rescued, broken_initial_exact_sdf_keys=broken,
        update_groups=dict(Counter(next(row['fit_domain'] for row in rows
            if r['key'] == row['source'] + ':' + row['sample_key']) for r in updates)),
        decision=decision, optimizer_steps_performed_by_postcheck=0, model_forwards=0, gpu_used=False,
        limitations=['Per-pair metric aggregates checked against saved metric rows; not fresh prediction/label replay',
            'Frozen cache tensors were held in memory; persisted receipts bind hashes, not full feature tensors',
            'Only pair delta persisted; non-pair final-state invariance is supported by worker before/after digests'])
    write_json(out / 'postcheck.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifact_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
