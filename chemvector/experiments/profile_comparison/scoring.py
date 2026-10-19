"""Score saved SDFs, never select an action with truth or sanitize success."""

from collections import Counter
import json
from pathlib import Path

from rdkit import Chem, rdBase

from chemvector.experiments.id_graph.roundtrip import require_hash, sha256, signature, write_json
from .protocol import ARMS, REFERENCE, TRUTH, verify

METRICS = ('sdf_valid', 'core_graph_exact', 'stereo_graph_exact')


def score_sdf(block, expected):
    result = {**dict.fromkeys(METRICS, False), 'score_error': None}
    try:
        mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=True, strictParsing=True)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError('Invalid or empty SDF molecule')
        core, stereo = signature(mol, False), signature(mol)
        result.update(sdf_valid=True, core_graph_exact=core == expected['core'],
                      stereo_graph_exact=stereo == expected['stereo'], predicted_core=core, predicted_stereo=stereo)
    except Exception as error:
        result['score_error'] = {'type': type(error).__name__, 'message': str(error)}
    return result


def counts(rows):
    return {'images': len(rows), **{k: sum(r[k] for r in rows) for k in METRICS},
            'valid_but_wrong': sum(r['sdf_valid'] and not r['core_graph_exact'] for r in rows),
            'invalid': sum(not r['sdf_valid'] for r in rows),
            'invocation_errors': sum(r.get('invocation_error') is not None for r in rows),
            'fallback_used': sum(r.get('fallback_used', False) for r in rows),
            'fallback_attempted': sum(r.get('fallback_attempted', False) for r in rows),
            'fallback_valid': sum(r.get('fallback_used', False) and r['sdf_valid'] for r in rows),
            'fallback_core_exact': sum(r.get('fallback_used', False) and r['core_graph_exact'] for r in rows),
            'suppression_executed': sum(r.get('suppression', {}).get('executed', False) for r in rows)}


def summarize(rows, order, metadata):
    lookup = {(r['sample_key'], r['arm']): r for r in rows}
    required = {(key, arm) for key in order for arm in ARMS}
    if len(lookup) != len(rows) or set(lookup) != required or len(set(order)) != len(order):
        raise ValueError('Missing, duplicate, or unexpected scoring row')
    groups = {arm: [lookup[key, arm] for key in order] for arm in ARMS}
    pairs = {}
    for left, right in (('v01010', 'v01016'), ('strict_id_reference', 'v01010'), ('strict_id_reference', 'v01016')):
        pairs[f'{left}->{right}'] = {metric: {
            'rescued': [k for k in order if not lookup[k, left][metric] and lookup[k, right][metric]],
            'harmed': [k for k in order if lookup[k, left][metric] and not lookup[k, right][metric]],
        } for metric in METRICS}
    return {
        'denominator': len(order), 'molecules': len({metadata[k]['molecule_id'] for k in order}),
        'arms': {arm: counts(group) for arm, group in groups.items()}, 'pairs': pairs,
        'by_variant': {arm: {variant: counts([r for r in group if metadata[r['sample_key']]['variant'] == variant])
                            for variant in sorted({m['variant'] for m in metadata.values()})} for arm, group in groups.items()},
        'molecule_all_four_exact': {arm: sum(all(r['core_graph_exact'] for r in group if metadata[r['sample_key']]['molecule_id'] == mol)
                                           for mol in {metadata[k]['molecule_id'] for k in order}) for arm, group in groups.items()},
        'invocation_error_messages': {arm: dict(Counter(str(r['invocation_error']) for r in group if r.get('invocation_error')))
                                      for arm, group in groups.items()},
    }


def score(output, protocol):
    verify(protocol)
    receipt = json.loads((output / 'prediction_receipt.json').read_text())
    require_hash(output / 'protocol.json', receipt['protocol_sha256'])
    for name, digest in receipt['artifacts'].items():
        require_hash(output / name, digest)
    if not receipt['models_unchanged'] or not receipt['cold_parity_passed'] or receipt['invocations'] != 400:
        raise ValueError('Prediction controls did not pass')
    # Only now decode truth signatures and previous per-sample scores.
    expected = {r['sample_key']: r['expected'] for r in map(json.loads, (TRUTH / 'samples.jsonl').read_text().splitlines())}
    reference = {r['sample_key']: r for r in map(json.loads, (REFERENCE / 'samples.jsonl').read_text().splitlines())}
    invocations = {(r['sample_key'], r['arm']): r for r in json.loads((output / 'invocations.json').read_text())}
    rows = []
    with rdBase.BlockLogs():
        for key in protocol['sample_order']:
            # Compiler truth control uses the same SDF scorer as every prediction.
            control = score_sdf((TRUTH / 'samples' / f'{key}.sdf').read_text(), expected[key])
            if not all(control[k] for k in METRICS):
                raise ValueError('Truth SDF control failed under common scorer')
            for arm in ARMS:
                row = {'sample_key': key, 'arm': arm, 'invocation_error': None, 'fallback_used': False}
                if arm == 'strict_id_reference':
                    path = REFERENCE / 'sdf' / f'{key}.sdf'
                    result = score_sdf(path.read_text() if path.exists() else '', expected[key])
                    if any(result[k] != reference[key][k] for k in METRICS):
                        raise ValueError('Historical strict SDF scores not reproduced')
                else:
                    destination = output / 'predictions' / arm / key
                    invocation = invocations[key, arm]
                    row['invocation_error'] = invocation['error']
                    path = destination / 'prediction-repaired.sdf'
                    result = score_sdf(path.read_text() if path.exists() else '', expected[key])
                    if invocation['error'] is not None:
                        # A crashed invocation cannot count as successful even if a partial SDF exists.
                        result.update(dict.fromkeys(METRICS, False))
                    else:
                        metrics = json.loads((destination / 'metrics.json').read_text())
                        if bool(metrics['sdf_valid']) != result['sdf_valid']:
                            raise ValueError('Producer/scorer SDF validity disagree')
                        row['decode_path'] = metrics['diagnostics']['decode_path']
                        row['fallback_used'] = row['decode_path'] == 'direct_atom_pair_carbon_fallback'
                        row['fallback_attempted'] = bool(metrics['errors'])
                        row['compiler_errors'] = metrics['errors']
                        row['suppression'] = metrics['diagnostics']['structural_twin_suppression']
                        row['ocr_labels'] = metrics['diagnostics']['ocr']['labels']
                rows.append({**row, **result})
    report = summarize(rows, protocol['sample_order'], protocol['sample_metadata'])
    report.update(protocol_sha256=sha256(output / 'protocol.json'), truth_controls=200, strict_reference_replays=200,
                  cold_parity_count=receipt['cold_parity_count'], models_unchanged=True,
                  comparison_completed=True, training_performed=False, promotion_authorized=False,
                  strict_reference_caveat=protocol['strict_reference_caveat'], scope=protocol['scope'])
    write_json(output / 'scores.json', rows)
    report['artifacts'] = {'scores.json': sha256(output / 'scores.json'),
                           'prediction_receipt.json': sha256(output / 'prediction_receipt.json')}
    verify(protocol)
    write_json(output / 'report.json', report)
    return report
