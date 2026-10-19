"""Offline shadow expansion: element alternatives and one active-node deletion."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from rdkit import rdBase

from chemvector.experiments.id_graph.roundtrip import sha256, require_hash, write_json
from chemvector.experiments.id_graph_candidates.protocol import PREDICTION, TRUTH, verify
from chemvector.experiments.id_graph_candidates.decode import compiled
from .decode import ExpansionBudget, extend

PREVIOUS = Path('outputs/experiments/id-graph-candidates-20260913')
PREVIOUS_SHA = 'a1e17b9bc8b63b4bd7d769ac95ffe5b37a90d6df34ce734cada4a38ba6686575'


def freeze(output):
    if output.exists():
        raise FileExistsError('Use a new expansion directory')
    require_hash(PREVIOUS / 'protocol.json', PREVIOUS_SHA)
    old = json.loads((PREVIOUS / 'protocol.json').read_text())
    report = json.loads((PREVIOUS / 'report.json').read_text())
    if not report['controls_passed']:
        raise ValueError('Require passing baseline candidate controls')
    inputs = dict(old['inputs'])
    inputs[str(PREVIOUS / 'report.json')] = sha256(PREVIOUS / 'report.json')
    inputs.update({str(PREVIOUS / name): digest for name, digest in report['artifacts'].items()})
    sources = dict(old['sources'])
    for p in sorted(Path(__file__).parent.glob('*.py')) + [Path('tests/chemvector/test_id_graph_candidate_expansion.py')]:
        sources[str(p)] = sha256(p)
    config = {'schema': 'id-graph-shadow-expansion-0.1', 'sources': sources, 'inputs': inputs,
              'frozen_utc': datetime.now(timezone.utc).isoformat(), 'rdkit_version': rdBase.rdkitVersion,
              'sample_order': old['sample_order'], 'denominator': 200, 'budget': asdict(ExpansionBudget()),
              'bond_threshold_logit': old['bond_threshold_logit'],
              'arms': ['v1', 'v1_element', 'v1_node_delete', 'union'],
              'union': 'union of independent one-edit candidates, NOT combined/multi-step edits',
              'element': 'same existing crop top8; change non-H element and its text-derived charge/H; retain isotope/radicals',
              'node_delete': 'enumerate all active heavy atoms in stable ID order up to128; delete atom and incident bonds',
              'source_evidence': 'all original observations preserved, including rejected proposals; active semantic graph may shrink',
              'limits': 'no new nodes/edges, no neural calls, no new OCR, no truth-dependent generation or candidate filtering',
              'decision': 'KEEP all valid originals; shadow candidate sets only; no ranking or automatic adoption',
              'inventory_partition': 'scorer-only necessary constraint on heavy-element/isotope multiset; not visual matching',
              'controls': 'old baseline and entire old candidate set replay200; truth core200; no old-candidate coverage loss',
              'scope': 'known-development pilot chosen after prior errors; not held-out generalization',
              'training_performed': False, 'promotion_authorized': False}
    if len(config['sample_order']) != 200 or len(set(config['sample_order'])) != 200:
        raise ValueError('Require fixed 200 samples')
    verify(config)
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', config)
    return config


def generate(output, config):
    evidence = json.loads((PREVIOUS / 'ocr_evidence.json').read_text())
    by_sample = {key: {} for key in config['sample_order']}
    for key, item in evidence.items():
        sample, candidate = key.rsplit(':p', 1)
        if sample not in by_sample:
            raise ValueError('Unexpected OCR sample')
        by_sample[sample][int(candidate)] = item
    (output / 'candidates').mkdir()
    artifacts, completed = {}, 0
    try:
        with rdBase.BlockLogs():
            for key in config['sample_order']:
                raw = json.loads((PREDICTION / 'raw' / f'{key}.json').read_text())
                graph = json.loads((PREDICTION / 'graphs' / f'{key}.json').read_text())
                previous = json.loads((PREVIOUS / 'candidates' / f'{key}.json').read_text())
                start = time.perf_counter()
                result = extend(raw, graph, by_sample[key], previous, config['bond_threshold_logit'], ExpansionBudget(**config['budget']))
                result['elapsed_seconds'] = time.perf_counter() - start
                path = output / 'candidates' / f'{key}.json'
                write_json(path, result)
                artifacts[str(path.relative_to(output))] = sha256(path)
                completed += 1
                if completed % 25 == 0:
                    print(f'Expanded {completed}/200', flush=True)
        verify(config)
        receipt = {'samples': completed, 'artifacts': artifacts, 'truth_used': False,
                   'protocol_sha256': sha256(output / 'protocol.json')}
        write_json(output / 'candidate_receipt.json', receipt)
        return receipt
    except Exception as error:
        write_json(output / 'failure.json', {'stage': 'generation', 'completed': completed, 'denominator': 200,
                                           'valid': False, 'error': str(error)})
        raise


def evaluate(output, config, receipt):
    from .evaluate import heavy_inventory, score, summarize

    if receipt['samples'] != 200:
        raise ValueError('Incomplete generation')
    for name, digest in receipt['artifacts'].items():
        require_hash(output / name, digest)
    expected = {r['sample_key']: r['expected'] for r in map(json.loads, (TRUTH / 'samples.jsonl').read_text().splitlines())}
    old_scores = {r['sample_key']: r for r in map(json.loads, (PREVIOUS / 'samples.jsonl').read_text().splitlines())}
    rows, partitions = [], []
    attempt_counts, correct_edit_counts = Counter(), Counter()
    elapsed = 0.
    budget_counts = Counter()
    with rdBase.BlockLogs():
        for key in config['sample_order']:
            result = json.loads((output / 'candidates' / f'{key}.json').read_text())
            previous = json.loads((PREVIOUS / 'candidates' / f'{key}.json').read_text())
            graph = json.loads((PREDICTION / 'graphs' / f'{key}.json').read_text())
            truth = json.loads((TRUTH / 'samples' / f'{key}.graph.json').read_text())
            scores = score(result, expected[key])
            rows.extend(scores)
            p, t = heavy_inventory(graph), heavy_inventory(truth)
            delta = sum(p.values()) - sum(t.values())
            category = ('fewer_heavy_atoms' if delta < 0 else 'more_heavy_atoms' if delta > 0 else
                        'same_count_different_elements_or_isotopes' if p != t else 'same_heavy_inventory')
            truth_status = compiled(truth)
            old_groups = {g['stereo']: g for g in previous['groups']}
            groups = {g['stereo']: g for g in result['groups']}
            all_preserved = all(k in groups and all(a in groups[k]['actions'] for a in g['actions']) for k, g in old_groups.items())
            baseline = next(r for r in scores if r['arm'] == 'v1')
            replayed = (result['baseline'] == previous['baseline']
                        and baseline['covered_core'] == old_scores[key]['baseline_or_candidate_core_coverage']
                        and baseline['legal_groups'] == old_scores[key]['legal_candidate_groups']
                        and baseline['wrong_only'] == old_scores[key]['candidate_only_wrong_core'])
            partitions.append({'sample_key': key, 'baseline_valid': result['baseline']['valid'],
                               'inventory_category': category, 'predicted_heavy_count': sum(p.values()),
                               'truth_heavy_count': sum(t.values()), 'heavy_count_delta': delta,
                               'baseline_replayed': replayed, 'old_candidate_set_preserved': all_preserved,
                               'truth_core_exact': truth_status['valid'] and truth_status['core'] == expected[key]['core']})
            attempt_counts.update(a['action']['kind'] for a in result['extension_attempts'])
            for kind in ('ELEMENT', 'DELETE_NODE'):
                correct_edit_counts[kind] += any(g['core'] == expected[key]['core'] and any(a['kind'] == kind for a in g['actions']) for g in result['groups'])
            for field in ('element_eligible', 'element_truncated', 'deletions_eligible', 'deletions_truncated'):
                budget_counts[field] += result[field]
            elapsed += result['elapsed_seconds']
    verify(config)
    for name, records in (('samples.jsonl', rows), ('partitions.jsonl', partitions)):
        with (output / name).open('x') as stream:
            for row in records:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
    report = summarize(rows, partitions)
    report.update(attempt_counts=dict(attempt_counts), correct_coverage_by_new_edit=dict(correct_edit_counts),
                  budget_counts=dict(budget_counts), total_generation_seconds=elapsed,
                  baseline_replays=sum(p['baseline_replayed'] for p in partitions),
                  old_candidate_set_preserved=sum(p['old_candidate_set_preserved'] for p in partitions),
                  truth_controls=sum(p['truth_core_exact'] for p in partitions),
                  controls_passed=all(p['baseline_replayed'] and p['old_candidate_set_preserved'] and p['truth_core_exact'] for p in partitions))
    report['artifacts'] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'artifacts'}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/experiments/id-graph-candidate-expansion-20260913'))
    output = parser.parse_args().output
    config = freeze(output)
    report = evaluate(output, config, generate(output, config))
    return 0 if report['controls_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
