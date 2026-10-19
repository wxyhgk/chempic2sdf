"""File identities for the already exposed development comparison (stdlib only)."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = ROOT / 'outputs/baselines/chemvector-v01016-reference-20260914/repository'
COMPARISON = Path('outputs/experiments/profile-comparison-20260914')
# The comparison identity is independent of the mutable project worktree.
PROTOCOL_SHA = '565a236efd04de73b90f941ef8d19f373ade478bfaf175fd52dab6be6bcd04b7'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(files):
    for name, digest in files.items():
        if sha256(name) != digest:
            raise ValueError(f'Bound file changed: {name}')


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def selection():
    path = SNAPSHOT / COMPARISON / 'protocol.json'
    verify_files({str(path): PROTOCOL_SHA})
    protocol = json.loads(path.read_text())
    keys = protocol['sample_order']
    if len(keys) != 200 or len(set(keys)) != 200:
        raise ValueError('Expected the fixed 200 development inputs')
    if any(Path(key).name != key or not key.startswith('fresh_dev_') for key in keys):
        raise ValueError('Unsafe sample identity')
    return protocol


def validate_rows(rows, order):
    keys = [row['sample_key'] for row in rows]
    if keys != order or len(set(keys)) != len(keys):
        raise ValueError('Missing, duplicate, reordered or unexpected result row')


def artifact_hashes(output):
    return {str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob('*')) if path.is_file()}
