"""Score sealed external predictions with ChemVector's unchanged RDKit scorer.

Run with .venv-train Python after the external producer exits. No third-party
model imports, prediction repair, SMILES fallback, confidence filtering or training.
"""

import argparse
import json
from pathlib import Path

from rdkit import rdBase

from chemvector.experiments.profile_comparison.scoring import METRICS, score_sdf
from .common import (COMPARISON, ROOT, SNAPSHOT, selection, sha256, validate_rows,
                     verify_files, write_json)

TRUTH = Path('outputs/experiments/id-graph-roundtrip-20260913')
REFERENCE_REPORT_SHA = 'a7c2cf8a57adfac4610757bd6eeeca7f8f911b19c93f7b2d604d2500fffadeed'


def score_prediction(invocation, block, expected):
    result = score_sdf(block, expected)
    annotations = invocation['annotations']
    completed = invocation['invocation_error'] is None and len(annotations) == 1
    if not completed:
        result.update(dict.fromkeys(METRICS, False))
    placeholder = any(a.get('smi') == 'C' and a.get('conf') == 0 for a in annotations)
    return {'sample_key': invocation['sample_key'], **result,
            'invocation_error': invocation['invocation_error'], 'missing_annotation': not annotations,
            'native_carbon_placeholder': placeholder, 'seconds': invocation['seconds']}


def counts(rows):
    return {'images': len(rows), **{metric: sum(row[metric] for row in rows) for metric in METRICS},
            'valid_but_wrong': sum(row['sdf_valid'] and not row['core_graph_exact'] for row in rows),
            'invalid': sum(not row['sdf_valid'] for row in rows),
            'invocation_errors': sum(row['invocation_error'] is not None for row in rows),
            'native_carbon_placeholders': sum(row['native_carbon_placeholder'] for row in rows)}


def verify_receipt(output, receipt):
    if not receipt['completed'] or receipt['smoke_only'] or receipt['sample_count'] != 200:
        raise ValueError('Require completed 200-image production, not a smoke run')
    if not receipt['models_unchanged'] or receipt['truth_used_for_prediction']:
        raise ValueError('Prediction integrity controls failed')
    for name, digest in receipt['artifacts'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != name:
            raise ValueError('Unsafe artifact path')
        verify_files({str(output / path): digest})
    verify_files({str(output / 'protocol.json'): receipt['protocol_sha256']})


def run(output):
    output = output.resolve()
    receipt = json.loads((output / 'prediction_receipt.json').read_text())
    verify_receipt(output, receipt)
    protocol = json.loads((output / 'protocol.json').read_text())
    original = selection()
    order, metadata = original['sample_order'], original['sample_metadata']
    if protocol['sample_order'] != order:
        raise ValueError('External producer used a different input selection')
    for name in ('chemvector/experiments/profile_comparison/scoring.py',
                 'chemvector/experiments/id_graph/roundtrip.py'):
        verify_files({str(ROOT / name): original['sources'][name]})
    # Ground truth is read only after producer receipt/artifact validation.
    truth_name = str(TRUTH / 'samples.jsonl')
    verify_files({str(SNAPSHOT / truth_name): original['inputs'][truth_name]})
    truth_rows = [json.loads(line) for line in (SNAPSHOT / truth_name).read_text().splitlines()]
    expected = {row['sample_key']: row['expected'] for row in truth_rows}
    if len(expected) != 200 or set(expected) != set(order):
        raise ValueError('Incomplete or unexpected reference truth')
    ref_report_path = SNAPSHOT / COMPARISON / 'report.json'
    verify_files({str(ref_report_path): REFERENCE_REPORT_SHA})
    reference = json.loads(ref_report_path.read_text())
    ref_scores_path = SNAPSHOT / COMPARISON / 'scores.json'
    verify_files({str(ref_scores_path): reference['artifacts']['scores.json']})
    ref_scores = json.loads(ref_scores_path.read_text())
    invocations = json.loads((output / 'invocations.json').read_text())
    validate_rows(invocations, order)
    rows = []
    with rdBase.BlockLogs():
        for invocation in invocations:
            key = invocation['sample_key']
            control_path = SNAPSHOT / TRUTH / 'samples' / f'{key}.sdf'
            verify_files({str(control_path): original['inputs'][str(TRUTH / 'samples' / f'{key}.sdf')]})
            control = score_sdf(control_path.read_text(), expected[key])
            if not all(control[metric] for metric in METRICS):
                raise ValueError('Common SDF scorer failed a truth control')
            name = f'native/{key}.sdf'
            block = (output / name).read_text() if name in receipt['artifacts'] else ''
            row = score_prediction(invocation, block, expected[key])
            annotations = invocation['annotations']
            if len(annotations) == 1:
                native = score_sdf(annotations[0].get('molblock') or '', expected[key])
                exported = score_sdf(block, expected[key])
                row['molblock_sdf_metrics_agree'] = all(native[m] == exported[m] for m in METRICS)
            else:
                row['molblock_sdf_metrics_agree'] = None
            rows.append(row)
    paired = {}
    for arm in ('v01010', 'v01016'):
        baseline = {row['sample_key']: row for row in ref_scores if row['arm'] == arm}
        if len(baseline) != 200 or set(baseline) != set(order):
            raise ValueError('Incomplete baseline rows')
        paired[arm] = {metric: {
            'rescued': [r['sample_key'] for r in rows if r[metric] and not baseline[r['sample_key']][metric]],
            'lost': [r['sample_key'] for r in rows if not r[metric] and baseline[r['sample_key']][metric]],
            'both_correct': sum(r[metric] and baseline[r['sample_key']][metric] for r in rows),
            'both_wrong': sum(not r[metric] and not baseline[r['sample_key']][metric] for r in rows),
        } for metric in METRICS}
    report = {
        'schema': 'external-ocsr-common-sdf-report-0.1', 'denominator': 200, 'molecules': 50,
        'model': 'local MolGrapher gc_no_stereo_model', 'molgrapher': counts(rows),
        'reference_arms': {arm: reference['arms'][arm] for arm in ('v01010', 'v01016')},
        'by_variant': {variant: counts([r for r in rows if metadata[r['sample_key']]['variant'] == variant])
                       for variant in ('original', 'rdkit', 'coordgen', 'stress')},
        'molecule_all_four_core_exact': sum(all(r['core_graph_exact'] for r in rows
                                               if metadata[r['sample_key']]['molecule_id'] == mol)
                                           for mol in {metadata[k]['molecule_id'] for k in order}),
        'paired': paired, 'truth_controls': 200, 'common_scorer_rdkit': rdBase.rdkitVersion,
        'native_molblock_sdf_metric_mismatches': sum(r['molblock_sdf_metrics_agree'] is False for r in rows),
        'prediction_receipt_sha256': sha256(output / 'prediction_receipt.json'),
        'producer_protocol_sha256': sha256(output / 'protocol.json'),
        'reference_report_sha256': REFERENCE_REPORT_SHA,
        'scope': protocol['scope'], 'stereo_caveat': protocol['stereo_caveat'],
        'geometry_caveat': protocol['geometry_caveat'], 'training_performed': False,
        'promotion_authorized': False, 'completed': True,
    }
    write_json(output / 'scores.json', rows)
    report['scores_sha256'] = sha256(output / 'scores.json')
    write_json(output / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('model', 'denominator', 'molgrapher', 'by_variant',
                                            'molecule_all_four_core_exact', 'native_molblock_sdf_metric_mismatches')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    run(parser.parse_args().output_dir)
