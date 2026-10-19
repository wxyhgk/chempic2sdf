"""Join existing OCR beams by object identity AND exact crop hash, not position."""

from copy import deepcopy
import json

from chemvector.experiments.id_graph_prediction.adapter import finite


def join_evidence(baseline_rows, beam_rows):
    expected = {}
    for row in baseline_rows:
        if row['arm'] != 'atom_conditioned':
            continue
        key = row['object_key']
        if not key.startswith('A:') or key[2:] in expected:
            raise ValueError('Duplicate or unsupported baseline OCR identity')
        expected[key[2:]] = row
    found = {}
    for row in beam_rows:
        key = row['object_key']
        if key not in expected:
            continue
        if key in found:
            raise ValueError('Duplicate OCR beam identity')
        base = expected[key]
        if row['crop_sha256'] != base['crop_sha256']:
            raise ValueError('OCR beam crop differs from frozen greedy crop')
        if row['arm'] != 'ctc_v0109_top8':
            raise ValueError('Unexpected OCR candidate source')
        candidates = deepcopy(row['candidates'])
        if len(candidates) > 8 or [c['rank'] for c in candidates] != list(range(len(candidates))):
            raise ValueError('Expected ordered top-8 ranks')
        for candidate in candidates:
            if candidate['score_kind'] != 'ctc_log_probability' or not isinstance(candidate['text'], str):
                raise ValueError('Unsupported OCR evidence score or text')
            finite(candidate['score'])
        if any(a['score'] < b['score'] for a, b in zip(candidates, candidates[1:])):
            raise ValueError('Beam ranks do not follow CTC scores')
        if row.get('error') and candidates:
            raise ValueError('Errored OCR record cannot supply candidates')
        found[key] = {'object_key': key, 'crop_sha256': row['crop_sha256'], 'model_id': row['model_id'],
                      'raw_greedy_text': base['text'], 'text_role_score': finite(base['score']),
                      'candidates': candidates, 'error': row.get('error')}
    # Missing records remain explicit, never fabricated as empty OCR evidence.
    for key, row in expected.items():
        if key not in found:
            found[key] = {'object_key': key, 'crop_sha256': row['crop_sha256'],
                          'raw_greedy_text': row['text'], 'text_role_score': finite(row['score']),
                          'model_id': None, 'candidates': [], 'error': 'missing_beam_record'}
    return found


def read_selected_beams(path, keys):
    """Stream the old artifact, decoding only object-key-selected JSON rows."""
    import re

    for line in path.open():
        # Artifact object keys are simple ASCII; parsing the key alone avoids
        # decoding unselected candidate payloads in the larger historical bundle.
        match = re.search(r'"object_key"\s*:\s*"([^"\\]+)"', line)
        if match and match.group(1) in keys:
            yield json.loads(line)
