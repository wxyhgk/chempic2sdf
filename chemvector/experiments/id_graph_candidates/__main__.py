"""python -m chemvector.experiments.id_graph_candidates --output NEW_DIRECTORY"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

from rdkit import rdBase

from chemvector.experiments.id_graph.roundtrip import sha256, write_json, require_hash
from .decode import Budget, compiled, decode
from .evidence import join_evidence, read_selected_beams
from .protocol import BASELINE_TEXT, BEAMS, PREDICTION, TRUTH, freeze, verify


def generate(output, config):
    selected = set(config['sample_order'])
    baseline_rows = [r for r in map(json.loads, BASELINE_TEXT.read_text().splitlines())
                     if r['sample_key'] in selected and r['arm'] == 'atom_conditioned']
    keys = {r['object_key'][2:] for r in baseline_rows}
    evidence = join_evidence(baseline_rows, read_selected_beams(BEAMS, keys))
    by_sample = {key: {} for key in selected}
    for key, item in evidence.items():
        sample, candidate = key.rsplit(':p', 1)
        by_sample[sample][int(candidate)] = item
    write_json(output / 'ocr_evidence.json', evidence)
    (output / 'candidates').mkdir()
    artifacts = {'ocr_evidence.json': sha256(output / 'ocr_evidence.json')}
    completed = 0
    try:
        with rdBase.BlockLogs():
            for key in config['sample_order']:
                raw = json.loads((PREDICTION / 'raw' / f'{key}.json').read_text())
                graph = json.loads((PREDICTION / 'graphs' / f'{key}.json').read_text())
                started = time.perf_counter()
                result = decode(raw, graph, by_sample[key], config['bond_threshold_logit'], Budget(**config['budget']))
                result['elapsed_seconds'] = time.perf_counter() - started
                path = output / 'candidates' / f'{key}.json'
                write_json(path, result)
                artifacts[str(path.relative_to(output))] = sha256(path)
                completed += 1
                if completed % 25 == 0:
                    print(f'Generated candidate sets {completed}/200', flush=True)
        verify(config)
        receipt = {'samples': completed, 'artifacts': artifacts, 'truth_used': False,
                   'budget': asdict(Budget(**config['budget'])), 'protocol_sha256': sha256(output / 'protocol.json')}
        write_json(output / 'candidate_receipt.json', receipt)
        return receipt
    except Exception as error:
        write_json(output / 'failure.json', {'stage': 'candidate_generation', 'completed': completed,
                                           'denominator': 200, 'message': str(error), 'valid': False})
        raise


def evaluate(output, config, receipt):
    from .evaluate import edge_agreement, score_candidates, summarize

    if receipt['samples'] != 200:
        raise ValueError('Incomplete candidate denominator')
    for name, digest in receipt['artifacts'].items():
        require_hash(output / name, digest)
    baseline = {r['sample_key']: r for r in map(json.loads, (PREDICTION / 'samples.jsonl').read_text().splitlines())}
    expected = {r['sample_key']: r['expected'] for r in map(json.loads, (TRUTH / 'samples.jsonl').read_text().splitlines())}
    rows, edges = [], []
    with rdBase.BlockLogs():
        for key in config['sample_order']:
            result = json.loads((output / 'candidates' / f'{key}.json').read_text())
            graph = json.loads((PREDICTION / 'graphs' / f'{key}.json').read_text())
            truth = json.loads((TRUTH / 'samples' / f'{key}.graph.json').read_text())
            row = score_candidates(result, expected[key])
            old, replay = baseline[key], result['baseline']
            row['baseline_replayed'] = (row['baseline_core_exact'] == old['core_graph_exact']
                and row['baseline_stereo_exact'] == old['stereo_graph_exact']
                and replay['valid'] == old['sdf_valid'] and replay['error'] == old['compile_error'])
            oracle = compiled(truth)
            row['truth_core_exact'] = oracle['valid'] and oracle['core'] == expected[key]['core']
            row['elapsed_seconds'] = result['elapsed_seconds']
            rows.append(row)
            edges.append({'sample_key': key, **edge_agreement(graph, truth)})
    verify(config)
    for name, items in (('samples.jsonl', rows), ('edge_agreement.jsonl', edges)):
        with (output / name).open('x') as stream:
            for row in items:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
    report = summarize(rows)
    totals = Counter()
    for row in edges:
        totals.update({k: v for k, v in row.items() if k != 'sample_key'})
    report.update(edge_agreement=dict(totals), baseline_replays=sum(r['baseline_replayed'] for r in rows),
                  truth_controls=sum(r['truth_core_exact'] for r in rows),
                  controls_passed=all(r['baseline_replayed'] and r['truth_core_exact'] for r in rows),
                  total_generation_seconds=sum(r['elapsed_seconds'] for r in rows))
    report['artifacts'] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'artifacts'}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/experiments/id-graph-candidates-20260913'))
    output = parser.parse_args().output
    config = freeze(output)
    receipt = generate(output, config)
    report = evaluate(output, config, receipt)
    return 0 if report['controls_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
