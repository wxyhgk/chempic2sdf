"""Bounded, post-hoc oracle diagnosis of the saved 200-image ID baseline."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from rdkit import rdBase

from chemvector.experiments.id_graph.roundtrip import require_hash, sha256, write_json
from chemvector.experiments.id_graph_prediction.scoring import compile_score
from .controls import FIELDS, intervene, representation_check

PREDICTION = Path('outputs/experiments/id-graph-prediction-20260913')
TRUTH = Path('outputs/experiments/id-graph-roundtrip-20260913')
PREDICTION_SHA = '8d54de90db5ead287777e1d6a49bcb1798f360acbe69b8af9c2202851dadc303'
ARMS = [f'{mode}_C{c}' for mode in FIELDS for c in (0, 1)]


def verify(config):
    for group in ('sources', 'inputs'):
        for name, digest in config[group].items():
            require_hash(name, digest)


def freeze(output):
    if output.exists():
        raise FileExistsError('Use a fresh diagnosis directory')
    require_hash(PREDICTION / 'protocol.json', PREDICTION_SHA)
    previous = json.loads((PREDICTION / 'protocol.json').read_text())
    sources = dict(previous['sources'])
    for path in sorted(Path(__file__).parent.glob('*.py')) + [Path('tests/chemvector/test_id_graph_diagnosis.py')]:
        sources[str(path)] = sha256(path)
    inputs = dict(previous['inputs'])
    for root in (PREDICTION, TRUTH):
        inputs[str(root / 'report.json')] = sha256(root / 'report.json')
        for name, digest in json.loads((root / 'report.json').read_text())['artifacts'].items():
            inputs[str(root / name)] = digest
    config = {'schema': 'id-graph-offline-diagnosis-0.1', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
              'sources': sources, 'inputs': inputs, 'sample_order': previous['sample_order'],
              'rdkit_version': rdBase.rdkitVersion, 'arms': ARMS, 'attribute_fields': FIELDS,
              'denominator': 200, 'matching_radius_px': 6.,
              'node_selection': 'only mutually unique radius-eligible pairs; same fixed matching for every arm',
              'C': 'replace induced subgraph among safe matched nodes; preserve all edges incident to other nodes',
              'stereo': 'do not inject stereo; all prediction arms remain unspecified',
              'controls': ['replay baseline all 200', 'truth core exact all 200',
                           'full_C1 exact on complete safe bijection subset; no claims outside this support'],
              'primary': 'canonical SDF core graph exact preserving isotope/charge',
              'secondary': 'SDF valid and paired rescue/harm, not achievable deployment gains',
              'normalization': 'compare raw differing edges after RDKit sanitization at matched IDs; invalid graphs unresolved',
              'scope': 'post-hoc reused-development diagnostic; no network, new data, training, production changes or A-axis',
              'causal_architecture_ranking_authorized': False}
    if len(config['sample_order']) != 200 or len(set(config['sample_order'])) != 200:
        raise ValueError('Require original 200 distinct samples')
    verify(config)
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', config)
    return config


def summarize(rows, controls, representations):
    lookup = {(r['sample_key'], r['arm']): r for r in rows}
    if len(rows) != 200 * len(ARMS) or len(lookup) != len(rows):
        raise ValueError('Incomplete/duplicate diagnostic rows')
    keys = [r['sample_key'] for r in controls]
    if len(keys) != 200 or len(set(keys)) != 200 or any((key, arm) not in lookup for key in keys for arm in ARMS):
        raise ValueError('Incomplete diagnostic keys')
    groups = {arm: [lookup[key, arm] for key in keys] for arm in ARMS}
    counts = {arm: {f: sum(r[f] for r in group) for f in ('sdf_valid', 'core_graph_exact')} for arm, group in groups.items()}
    transitions = {}
    for left in ARMS:
        for right in ARMS:
            same_attribute = left.rsplit('_', 1)[0] == right.rsplit('_', 1)[0]
            same_bonds = left[-1] == right[-1]
            if not ((same_attribute and left.endswith('C0') and right.endswith('C1'))
                    or (same_bonds and left.startswith('none_') and not right.startswith('none_'))):
                continue
            transitions[f'{left}->{right}'] = {f: {
                'rescued': sum(not lookup[k, left][f] and lookup[k, right][f] for k in keys),
                'harmed': sum(lookup[k, left][f] and not lookup[k, right][f] for k in keys)}
                for f in ('sdf_valid', 'core_graph_exact')}
    complete_keys = [r['sample_key'] for r in controls if r['complete_safe_bijection']]
    passed = all(r['baseline_replayed'] and r['truth_core_exact'] for r in controls)
    passed &= bool(complete_keys) and all(lookup[k, 'full_C1']['core_graph_exact'] for k in complete_keys)
    return {'denominator': 200, 'arms': counts, 'transitions': transitions,
            'baseline_replays': sum(r['baseline_replayed'] for r in controls),
            'truth_controls': sum(r['truth_core_exact'] for r in controls),
            'complete_safe_subset': {'count': len(complete_keys), 'arms': {
                arm: {f: sum(lookup[k, arm][f] for k in complete_keys) for f in ('sdf_valid', 'core_graph_exact')}
                for arm in ARMS}},
            'diagnostic_controls_passed': passed,
            'normalization': {f: sum(r[f] for r in representations) for f in ('resolved', 'persistent', 'unresolved')},
            'compile_failures_by_arm': {arm: dict(Counter(r['compile_error']['type'] for r in group if r['compile_error']))
                                       for arm, group in groups.items()},
            'causal_architecture_ranking_authorized': False, 'training_performed': False}


def run(output):
    config = freeze(output)
    baseline = {r['sample_key']: r for r in map(json.loads, (PREDICTION / 'samples.jsonl').read_text().splitlines())}
    expected = {r['sample_key']: r['expected'] for r in map(json.loads, (TRUTH / 'samples.jsonl').read_text().splitlines())}
    rows, controls, representations = [], [], []
    with rdBase.BlockLogs():
        for key in config['sample_order']:
            graph = json.loads((PREDICTION / 'graphs' / f'{key}.json').read_text())
            truth = json.loads((TRUTH / 'samples' / f'{key}.graph.json').read_text())
            protected = json.dumps(graph, sort_keys=True)
            for mode in FIELDS:
                for c in (0, 1):
                    changed, diagnostic = intervene(graph, truth, mode, bool(c))
                    metrics, _ = compile_score(changed, expected[key])
                    row = {'sample_key': key, 'arm': f'{mode}_C{c}', **metrics, **diagnostic}
                    rows.append(row)
                    if mode == 'none' and c == 0:
                        replay = all(row[f] == baseline[key][f] for f in (
                            'sdf_valid', 'core_graph_exact', 'stereo_graph_exact', 'compile_error'))
                        if changed != graph:
                            raise ValueError('No-op intervention changed baseline')
                        controls.append({'sample_key': key, 'baseline_replayed': replay,
                            'truth_core_exact': compile_score(truth, expected[key])[0]['core_graph_exact'],
                            'complete_safe_bijection': diagnostic['complete_safe_bijection']})
            if json.dumps(graph, sort_keys=True) != protected:
                raise ValueError('Diagnostic mutated its prediction')
            representations.append({'sample_key': key, **representation_check(graph, truth)})
            if len(controls) % 25 == 0:
                print(f'Diagnosed {len(controls)}/200', flush=True)
    verify(config)
    for name, records in (('samples.jsonl', rows), ('controls.jsonl', controls), ('representation.jsonl', representations)):
        with (output / name).open('x') as stream:
            for row in records:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
    report = summarize(rows, controls, representations)
    report['protocol_sha256'] = sha256(output / 'protocol.json')
    report['artifacts'] = {str(p.relative_to(output)): sha256(p) for p in output.iterdir() if p.is_file()}
    write_json(output / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('transitions', 'artifacts')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/experiments/id-graph-diagnosis-20260913'))
    report = run(parser.parse_args().output)
    raise SystemExit(0 if report['diagnostic_controls_passed'] else 1)
