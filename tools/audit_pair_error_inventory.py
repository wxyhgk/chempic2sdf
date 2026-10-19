"""Two fixed training-case inventories; CPU saved-data analysis only."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from chemvector.training.curriculum import runtime
from tools import pair_error_inventory as inventory
from tools.preview_pair_false_edges import ROOT, TRIAL, read

OUTPUT = ROOT / 'outputs/audits/chemvector-pair-error-inventory2-20260921'
KEYS = ('curr_4ef506af64e8f0b8__bond_short', 'curr_08d02946e879406f__font_large')
LABELS = ROOT / 'outputs/audits/chemvector-pair-regression-diagnostics58-20260921'


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def worker():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not OUTPUT.is_dir():
        raise ValueError('Fresh budgeted CPU-only audit required')
    hashes = {}
    def data(path, expected=None):
        return json.loads(read(path, hashes, expected))
    manifest = data(ROOT / 'outputs/datasets/chemvector-selector-retention-fit-20260919/manifest.json',
                    'f5e3a7163eef5c49118ad78dcf05778f4ad889dc9b6389097ef20896b79e2111')
    post = data(TRIAL / 'postcheck.json', 'b09b82c98fe5280de0d79a6cca858a2854398fa66dca1d29205997f4cece1f54')
    caches = data(TRIAL / 'cache-manifest.json', '3761e0bf5a1e00d7d816a32d638a54a94a84b5b72b2256013bb65a7f1af03068')
    label_report = data(LABELS / 'report.json', '7bede56f8f8cbf5e7bc03a1e70cf964efb638bebff80840888bab146464789a2')
    for path in (Path(__file__), Path(inventory.__file__), Path(runtime.__file__),
                 ROOT / 'tools/preview_pair_false_edges.py', ROOT / 'tools/pair_local_repair.py',
                 ROOT / 'tests/chemvector/test_pair_error_inventory.py'):
        read(path, hashes)
    write(OUTPUT / 'plan.json', dict(sample_keys=KEYS, phases=[0, 50], wall_seconds=120,
        saved_data_only=True, purpose='Separate selected-node, text, missing/extra-edge, type, and stereo limitations',
        correspondence='Quantization-robust reciprocal local association; not a training matcher',
        membership='Exact sampled endpoints plus aggregate coverage; unique possible radius targets only',
        target_coordinate_contract='round(source_pixel / 767 * 1023) / 1023; matching metric multiplies by 768',
        prediction_quantization_error_bound=inventory.EPSILON, new_training=False, graph_edits=0))
    artifacts, summaries = {}, []
    for key in KEYS:
        selected = [r for r in manifest['training_samples'] if r['sample_key'] == key]
        if len(selected) != 1 or selected[0]['split'] != 'train':
            raise ValueError('Only the two fixed training cases are permitted')
        row = selected[0]
        folder = Path(manifest['sources'][row['source']]) / 'samples'
        meta = data(folder / (key + '.meta.json'), row['meta_sha256'])
        if meta['sample_key'] != key or meta['split'] != 'train':
            raise ValueError('Training source identity mismatch')
        layers = data(folder / (key + '.layers.json'), meta['sha256']['.layers.json'])
        graph = data(folder / (key + '.graph.json'), meta['sha256']['.graph.json'])
        if (layers['sample_key'] != key or (layers['width'], layers['height']) != (768, 768)
                or graph['molecule_id'] != layers['molecule_id']):
            raise ValueError('Source geometry mismatch')
        for kind in ('atoms', 'bonds'):
            if len(graph[kind]) != len(layers[kind]) or any(
                    any(v.get(k) != value for k, value in g.items()) for g, v in zip(graph[kind], layers[kind])):
                raise ValueError('Source graph/layers chemical identity mismatch')
        cache = data(TRIAL / (name := 'cache--' + key + '.json'), post['artifact_sha256'][name])
        if cache != next(c for c in caches if c['key'] == row['source'] + ':' + key):
            raise ValueError('Per-case cache receipt differs from bound manifest')
        for path in ('chemvector/training/curriculum/data.py', 'chemvector/training/curriculum/pair_cached_trial.py',
                     'chemvector/training/graph/relations/train_direct_atom_pair_bond.py',
                     'chemvector/training/curriculum/supervision_contract.py'):
            read(ROOT / path, hashes, cache['implementation_sha256'][path])
        records = [data(TRIAL / (name := f'check-{step:06d}--{key}.json'), post['artifact_sha256'][name])
                   for step in (0, 50)]
        if any(r['metrics']['key'] != cache['key'] or r['metrics']['candidate_label_sha256'] != cache['candidate_label_sha256']
               for r in records):
            raise ValueError('Prediction/cache identity mismatch')
        labels = data(LABELS / (key + '.json'), label_report['case_sha256'][key + '.json'])
        if labels['sample_key'] != key:
            raise ValueError('Sampler receipt sample mismatch')
        sampled = labels['presence_loss_replay']['arms']['initial_logits']['per_sampled_row']
        result = inventory.inventory(layers, records, cache['coverage'], sampled)
        result.update(sample_key=key, source_directory=str(folder))
        path = OUTPUT / (key + '.json')
        write(path, result)
        artifacts[path.name] = sha256(path.read_bytes()).hexdigest()
        summaries.append(dict(sample_key=key, coverage=result['coverage'],
            matched_mapping_inferred_from_receipts=result['receipt_derived_mapping_complete'],
            absent_local_targets=[n['target_id'] for n in result['nodes'] if n['candidate_slot'] is None],
            selected_unmatched_targets=[n['target_id'] for n in result['nodes'] if n['status'] == 'selected_but_unmatched'],
            wrong_element_targets=[n['target_id'] for n in result['nodes'] if 'element' in n.get('stable_chemical_mismatches', [])],
            text_role_miss_targets=[n['target_id'] for n in result['nodes'] if n.get('text_status') == 'visible_text_not_routed_to_ocr'],
            missing_atom_stereo_annotations=[n['target_id'] for n in result['nodes'] if n.get('stereo_annotation_missing')],
            phases={step: dict(counts=p['counts'], components=len(p['components']),
                              extra_or_unresolved_edges=p['extra_or_unresolved_edges']) for step, p in result['phases'].items()}))
    for path, digest in list(hashes.items()):
        read(path, hashes, digest)
    report = dict(schema_version='pair-error-inventory2-0.1', cases=summaries, input_sha256=hashes,
        case_sha256=artifacts, training_performed=False, checkpoint_loads=0, model_forwards=0, gpu_used=False,
        original_predictions_modified=False, production_changed=False, validation_sample_artifacts_read=False,
        sealed_test_read=False, new_identity_queries=0, original_float_cache_loaded=False,
        limitations=['Two selected training failures, not a prevalence or generalization estimate',
            'Receipt-derived membership and unique radius identity do not recover original float coordinates',
            'Unsampled does not mean negative; spatial ownership is not a new training label',
            'A missing selected candidate cannot distinguish raw proposal absence from selector rejection',
            'Raw chemical type, scene type, and whole-molecule exactness are separate',
            'Literal HN/NH orientation differences are not automatically chemical mistakes',
            'Stereo annotation absence is recorded; complete stereochemical equivalence is not certified'])
    write(OUTPUT / 'report.json', report)
    print(json.dumps(summaries, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args(argv)
    if args.worker:
        worker()
        return 0
    receipt = runtime.run_budgeted([sys.executable, '-m', 'tools.audit_pair_error_inventory', '--worker'],
                                  output=OUTPUT, budget=runtime.ResourceBudget(120))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
