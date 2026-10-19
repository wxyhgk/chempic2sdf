"""Bounded metadata-only pair-stage preparation; no images or model inference."""

import argparse
import json
import os
from pathlib import Path
import sys

from . import pair_readiness
from .interface_diagnostic import write_json
from .runtime import ResourceBudget, run_budgeted

RELEASE = Path('outputs/datasets/chemvector-expanded-release-20260915')
OUTPUT = Path('outputs/audits/chemvector-pair-readiness-20260918')


def audit(output):
    from .checkpoints import file_digest
    from .data import validate_records
    from .release import verify_release

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only preparation required')
    output = Path(output)
    if not output.is_dir() or any((output / p).exists() for p in ('plan.json', 'report.json')):
        raise ValueError('Fresh supervised output required')
    manifest, _, bindings = verify_release(RELEASE / 'manifest.json', RELEASE / 'admission.json')
    rows = pair_readiness.reference_training_rows(manifest)
    files = [RELEASE / 'manifest.json', RELEASE / 'admission.json', Path(__file__),
             Path(pair_readiness.__file__), Path(__file__).with_name('pair_stage.py'),
             Path(__file__).with_name('supervision_contract.py')]
    inputs = {str(p): file_digest(p) for p in files}
    for row in rows:
        path = Path(manifest['sources'][row['source']]) / 'samples' / (row['sample_key'] + '.meta.json')
        if path.is_symlink() or file_digest(path) != row['meta_sha256']:
            raise ValueError('Released metadata changed')
        inputs[str(path)] = row['meta_sha256']
    plan = {'schema_version': 'chemvector-pair-readiness-plan-0.1', 'bindings': bindings,
        'input_sha256': inputs, 'reference_training_rows': rows,
        'selection': 'All curriculum/train/reference; target-only 2--7 heavy atoms and no specified stereo; no prediction filtering',
        'inventory_scope': 'Every admitted training manifest row, counted separately by source and molecular identity',
        'required_motifs': pair_readiness.REQUIRED,
        'geometry_rule': 'Confirmed non-edge with shared target neighbor, 10--90% along segment and <=2px perpendicular; diagnostic labels only',
        'cpu_wall_seconds': 180, 'model_forwards': 0, 'images_read': 0, 'optimizer_steps': 0,
        'training_authorized': False, 'external_identity_queries': 0}
    write_json(output / 'plan.json', plan)
    records = []
    for row in rows:
        prefix = Path(manifest['sources'][row['source']]) / 'samples' / row['sample_key']
        meta = json.loads(prefix.with_name(prefix.name + '.meta.json').read_text())
        if (meta['sample_key'] != row['sample_key'] or meta['molecule_id'] != row['molecule_id']
                or meta['canonical_smiles'] != row['canonical_smiles'] or meta['split'] != 'train'
                or meta['training_authorized'] is not True or meta['external_overlap_status'] != 'checked'
                or meta['profile_id'] != 'reference' or meta['profile']['font_size'] != 24
                or meta['profile']['bond_line_width'] != 2):
            raise ValueError('Unexpected reference training metadata')
        payload = []
        for suffix in ('.graph.json', '.layers.json', '.text-records.json'):
            path = prefix.with_name(prefix.name + suffix)
            if path.is_symlink() or file_digest(path) != meta['sha256'][suffix]:
                raise ValueError('Same-source target payload changed')
            inputs[str(path)] = meta['sha256'][suffix]
            payload.append(json.loads(path.read_text()))
        graph, layers, texts = payload
        if (graph['canonical_smiles'] != row['canonical_smiles']
                or graph['molecule_id'] != row['molecule_id'] or layers['sample_key'] != row['sample_key']
                or layers['profile_id'] != 'reference'
                or any(x['split'] != 'train' or x['training_authorized'] is not True for x in (graph, layers))):
            raise ValueError('Released graph/layer identity or lifecycle mismatch')
        for record in (layers, texts):
            if (record['source_png_sha256'] != meta['sha256']['.png']
                    or record['source_svg_sha256'] != meta['sha256']['.svg']):
                raise ValueError('Target source binding mismatch')
        if layers['text_records_sha256'] != meta['sha256']['.text-records.json']:
            raise ValueError('Text record binding mismatch')
        validate_records(graph, layers, texts)
        records.append(pair_readiness.describe_reference(row, graph, layers))
    inventory = pair_readiness.training_inventory(manifest)
    summary = pair_readiness.summarize_references(records)
    for path, digest in inputs.items():
        if file_digest(path) != digest:
            raise ValueError('Preparation changed a bound input')
    write_json(output / 'target-records.json', records)
    write_json(output / 'panel.json', {
        'schema_version': 'chemvector-pair-preparation-panel-0.1', 'bindings': bindings,
        'samples': [r['row'] for r in records if r['eligible_clean_pair_stage']],
        'training_authorized': False, 'missing_required_motifs': summary['missing_required_motifs'],
        'purpose': 'Existing training-side inspection list; NOT a new release or run authorization'})
    report = {'schema_version': 'chemvector-pair-readiness-result-0.1',
        'plan_sha256': file_digest(output / 'plan.json'), 'verified_input_sha256': inputs,
        'summary': summary, 'training_manifest_inventory': inventory,
        'model_forwards': 0, 'images_read': 0, 'gpu_used': False, 'optimizer_steps': 0,
        'external_identity_queries': 0, 'training_performed': False, 'deployed': False,
        'limits': ['Manifest coverage is availability, not historical optimizer exposure or causal attribution',
                   'Reference rendering checks use already admitted metadata, not a new image quality judgment',
                   'Target hard negatives are not candidate reachability or model precision',
                   'Missing clean motifs may exist in larger training molecules',
                   'Pair-stage objective tested only with synthetic tensors; real model not updated']}
    write_json(output / 'report.json', report)
    print(json.dumps(summary), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        audit(OUTPUT)
        return 0
    receipt = run_budgeted([sys.executable, '-m', 'chemvector.training.curriculum.pair_readiness_audit', '--worker'],
                           output=OUTPUT, budget=ResourceBudget(180))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
