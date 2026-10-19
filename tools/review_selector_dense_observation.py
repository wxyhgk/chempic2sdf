"""Post-snapshot neighborhood review: no model, forward, or changed decoding."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/audits/chemvector-selector-dense-observation7-20260920'
REPORT_SHA = '7ba69b5f75241fced837611a9dbdef236a1855fdbbd290f01ff50154f78ec1ff'


def neighborhood_category(target):
    """A suppressed GT cell does not rule out other nearby local peaks."""
    near = target['dense_cells_within_6px']
    if (not target['cpu_still_missing'] or any(n['raw_active'] for n in near)
            or any(not 0 <= n['direct_target_distance_px'] <= 6 for n in near)
            or len({n['grid_id'] for n in near}) != len(near)):
        raise ValueError('Unique within-radius missing-target neighborhood required')
    peaks = [n for n in near if n['is_local_peak']]
    if not peaks:
        return 'no_local_peak_within_radius'
    if any(n['in_topk_before_threshold'] and n['passes_threshold'] for n in peaks):
        raise ValueError('Contradictory active-peak exclusion')
    if any(n['passes_threshold'] for n in peaks):
        return 'above_threshold_local_peak_excluded_by_topk'
    return ('local_peaks_below_threshold_and_all_outside_topk'
            if all(not n['in_topk_before_threshold'] for n in peaks)
            else 'local_peaks_below_threshold')


def checked(path, expected):
    path = Path(path).absolute()
    if path.resolve() != path or 'sealed_test' in path.parts:
        raise ValueError('Explicit nonsymlink nonsealed input required')
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != expected:
        raise ValueError('Bound artifact changed: ' + str(path))
    return payload


def review():
    report = json.loads(checked(SOURCE / 'report.json', REPORT_SHA))
    if (report['completed_images'] != 7 or report['focused_targets'] != 10
            or report['model_forwards'] != 7 or report['reproduced_targets'] != 10):
        raise ValueError('Expected complete reproduced fixed scope')
    for path, digest in report['input_sha256'].items():
        checked(path, digest)
    for name, digest in report['snapshot_sha256'].items():
        checked(SOURCE / name, digest)
    counts, categories, rows = Counter(), Counter(), []
    for name, digest in report['case_sha256'].items():
        case = json.loads(checked(SOURCE / name, digest))
        with np.load(SOURCE / (case['sample_key'] + '.npz'), allow_pickle=False) as snapshot:
            # Independent NumPy float64 sigmoid, direct distances and local
            # window comparisons; check membership against the actual float32
            # decoder observations, never replace its decisions or tie policy.
            logits = snapshot['heatmap_logits'][0, 0].astype(np.float64)
            prob = 1 / (1 + np.exp(-logits))
            xy = snapshot['anchors'][0].reshape(2, -1).T.astype(np.float64)
            active = set(snapshot['raw_indices'][0, snapshot['raw_mask'][0]].tolist())
            for target in case['targets']:
                x, y = target['target_cell']
                if not prob[y, x] < prob[max(0, y-1):y+2, max(0, x-1):x+2].max() or y*192+x in active:
                    raise ValueError('GT suppression failed independent verification')
                near = target['dense_cells_within_6px']
                distances = np.linalg.norm(xy - np.array(target['saved_target']), axis=1) * 768
                if set(np.flatnonzero(distances <= 6)) != {n['grid_id'] for n in near}:
                    raise ValueError('Distance precision changes neighborhood membership')
                for node in near:
                    cy, cx = divmod(node['grid_id'], 192)
                    peak = bool(prob[cy, cx] == prob[max(0, cy-1):cy+2, max(0, cx-1):cx+2].max())
                    if (peak != node['is_local_peak'] or node['grid_id'] in active
                            or bool(prob[cy, cx] >= .01) != node['passes_threshold']):
                        raise ValueError('Neighborhood peak/threshold precision discrepancy')
                    counts['near_dense_cells'] += 1
                    counts['near_local_peaks'] += peak
                    counts['near_nonpeak_cells'] += not peak
                    counts['near_above_threshold_cells'] += node['passes_threshold']
                category = neighborhood_category(target)
                categories[category] += 1
                counts['targets'] += 1
                counts['gt_above_threshold'] += target['gt_cell']['passes_threshold']
                rows.append(dict(sample_key=case['sample_key'], target_id=target['target_id'],
                    text=target['text'], category=category,
                    gt_score=target['gt_cell']['score'],
                    gt_anchor_distance_px=target['gt_cell']['direct_target_distance_px'],
                    nearest_raw_distance_px=target['cpu_nearest_cell']['direct_target_distance_px'],
                    alternative_near_peaks=[{k: n[k] for k in ('grid_id', 'score', 'passes_threshold',
                        'in_topk_before_threshold', 'strictly_higher_peak_count', 'direct_target_distance_px')}
                        for n in near if n['is_local_peak']]))
    events = [json.loads(line) for line in (SOURCE / 'events.jsonl').read_text().splitlines()]
    keys = [r['sample_key'] for r in report['cases']]
    expected = [(key, event) for key in keys for event in ('forward_reserved', 'snapshot_saved', 'case_complete')]
    if (len(set(keys)) != 7 or [(e['sample_key'], e['event']) for e in events] != expected
            or counts['targets'] != 10):
        raise ValueError('Incomplete/duplicate forward or persistence order')
    result = dict(schema_version='selector-dense-neighborhood-review-0.1',
        source_report_sha256=REPORT_SHA, review_code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        verified_input_hashes=len(report['input_sha256']), verified_snapshots=7, verified_cases=7,
        counts=counts, target_categories=categories, targets=rows,
        original_forward_count_verified=7, additional_model_forwards=0,
        training_authorized=False, thresholds_changed=False,
        note='Neighborhood categories describe saved outputs, not training causes or measured fixes')
    with (SOURCE / 'neighborhood-review.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    result = review()
    print(json.dumps(dict(counts=result['counts'], target_categories=result['target_categories']), ensure_ascii=False))
