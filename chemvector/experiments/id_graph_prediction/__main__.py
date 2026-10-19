"""python -m chemvector.experiments.id_graph_prediction --output NEW_DIRECTORY"""

import argparse
from collections import Counter
import json
from pathlib import Path

from chemvector.experiments.id_graph.roundtrip import require_hash, sha256, write_json
from .runtime import TRUTH, freeze, predict, verify
from .scoring import component_errors, compile_score


def evaluate(output, config, receipt):
    if receipt['sample_count'] != 200:
        raise ValueError('Incomplete prediction denominator')
    for name, digest in receipt['artifacts'].items():
        require_hash(output / name, digest)
    truth_scores = {r['sample_key']: r for r in map(json.loads, (TRUTH / 'samples.jsonl').read_text().splitlines())}
    adapter_rows = {r['sample_key']: r for r in receipt['rows']}
    rows = []
    (output / 'sdf').mkdir()
    for key in config['sample_order']:
        graph = json.loads((output / 'graphs' / f'{key}.json').read_text())
        truth = json.loads((TRUTH / 'samples' / f'{key}.graph.json').read_text())
        metrics, sdf = compile_score(graph, truth_scores[key]['expected'])
        row = {**adapter_rows[key], **metrics, 'components': component_errors(graph, truth, config['matching_radius_px'])}
        rows.append(row)
        if sdf is not None:
            with (output / 'sdf' / f'{key}.sdf').open('x') as stream:
                stream.write(sdf)
    verify(config)
    with (output / 'samples.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    sums = Counter()
    attribute_errors, representation = Counter(), Counter()
    for row in rows:
        for key, value in row['components'].items():
            if isinstance(value, int):
                sums[key] += value
        attribute_errors.update(row['components']['attribute_errors'])
        representation.update(row['components']['representation_differences'])
    report = {'schema': config['schema'], 'denominator': 200,
              'counts': {name: sum(r[name] for r in rows) for name in
                         ('adapter_preserved', 'deployed_replay_exact', 'sdf_valid', 'core_graph_exact', 'stereo_graph_exact')},
              'component_totals': dict(sums), 'attribute_errors': dict(attribute_errors),
              'representation_differences_not_error_counts': dict(representation),
              'sample_flags_nonexclusive': {
                  'unmatched_truth_atoms': sum(r['components']['unmatched_truth_atoms'] > 0 for r in rows),
                  'unmatched_predicted_atoms': sum(r['components']['unmatched_predicted_atoms'] > 0 for r in rows),
                  'ambiguous_geometry': sum(r['components']['predicted_ambiguous'] + r['components']['truth_ambiguous'] > 0 for r in rows),
                  'attribute_error_on_unambiguous_match': sum(r['components']['atoms_with_attribute_error'] > 0 for r in rows),
                  'bond_error_on_unambiguous_endpoints': sum(any(r['components'][f] for f in ('missing_bonds', 'extra_bonds', 'wrong_bond_types')) for r in rows)},
              'compile_errors': dict(Counter(r['compile_error']['message'] for r in rows if r['compile_error'])),
              'limitations': [config['scope'], config['compiler_comparison'],
                  'unmatched at 6 px combines missing/extra detections and localization errors',
                  'ambiguous endpoints excluded only from attribute/bond counts, never from the 200-sample graph denominator',
                  'descriptive overlapping errors; no causal ranking or 8-arm intervention results',
                  'stereo is not predicted by this baseline; no truth stereo injected',
                  'raw bond-class errors include alternative aromatic/Kekule encodings; inspect before treating as chemical errors'],
              'models_unchanged': receipt['models_unchanged'], 'training_performed': False,
              'promotion_authorized': False, 'factorial_executed': False}
    report['artifacts'] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('artifacts', 'compile_errors')}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/experiments/id-graph-prediction-20260913'))
    args = parser.parse_args()
    config = freeze(args.output)
    receipt = predict(args.output, config)
    evaluate(args.output, config, receipt)


if __name__ == '__main__':
    main()
