"""Archive the measured v0.10.16 reference without committing or changing the worktree.

Only the existing comparison's hash-bound sources, checkpoints and evidence are
copied. Dataset shards and Python environments remain external dependencies.
This is an integrity-checked local snapshot, not a signed or OS-immutable backup.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys

COMPARISON = 'outputs/experiments/profile-comparison-20260914'
PROTOCOL_SHA = '565a236efd04de73b90f941ef8d19f373ade478bfaf175fd52dab6be6bcd04b7'
REPORT_SHA = 'a7c2cf8a57adfac4610757bd6eeeca7f8f911b19c93f7b2d604d2500fffadeed'
EXTRA_FILES = ('requirements-train.txt', 'tools/freeze_chemvector_baseline.py',
               'tests/chemvector/test_freeze_chemvector_baseline.py',
               'docs/model/training/20260914-frontend-profile-common-sdf-comparison.md',
               'docs/model/baselines/20260914-frozen-reference-and-exploration.md')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_file(root, name):
    path = PurePosixPath(name)
    if not isinstance(name, str) or not name or path.is_absolute() or path.as_posix() != name:
        raise ValueError('Expected canonical repository-relative path')
    if any(part in ('.', '..') for part in path.parts) or '\\' in name:
        raise ValueError('Unsafe snapshot path')
    result = root
    for part in path.parts:
        result = result / part
        if result.is_symlink():
            raise ValueError('Snapshot inputs must not traverse symlinks')
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def require_hash(root, name, digest):
    path = safe_file(root, name)
    if sha256(path) != digest:
        raise ValueError(f'Bound file changed: {name}')
    return path


def add_binding(files, name, digest):
    if name in files and files[name] != digest:
        raise ValueError(f'Conflicting hash bindings: {name}')
    files[name] = digest


def manifest_digest(manifest):
    value = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def copy_bound_files(repo, destination, files):
    """Use independent copies, never hardlinks to mutable workspace files."""
    total_bytes = 0
    for name, digest in sorted(files.items()):
        source = require_hash(repo, name, digest)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        if sha256(target) != digest:
            raise ValueError(f'Copy verification failed: {name}')
        total_bytes += target.stat().st_size
    return total_bytes


def build_bindings(repo):
    protocol_path = require_hash(repo, f'{COMPARISON}/protocol.json', PROTOCOL_SHA)
    report_path = require_hash(repo, f'{COMPARISON}/report.json', REPORT_SHA)
    protocol = json.loads(protocol_path.read_text())
    report = json.loads(report_path.read_text())
    if not report['comparison_completed'] or not report['models_unchanged'] or report['denominator'] != 200:
        raise ValueError('Comparison is not a completed frozen reference')
    if report['arms']['v01010']['core_graph_exact'] != 55 or report['arms']['v01016']['core_graph_exact'] != 59:
        raise ValueError('Unexpected reference scores')
    receipt_binding = report['artifacts']['prediction_receipt.json']
    receipt = json.loads(require_hash(repo, f'{COMPARISON}/prediction_receipt.json', receipt_binding).read_text())
    if not receipt['cold_parity_passed'] or not receipt['models_unchanged'] or receipt['invocations'] != 400:
        raise ValueError('Prediction controls failed')
    files = {}
    for group in ('sources', 'checkpoints', 'inputs'):
        for name, digest in protocol[group].items():
            add_binding(files, name, digest)
    for bundle in (report, receipt):
        for name, digest in bundle['artifacts'].items():
            add_binding(files, f'{COMPARISON}/{name}', digest)
    add_binding(files, f'{COMPARISON}/report.json', REPORT_SHA)
    add_binding(files, f'{COMPARISON}/protocol.json', PROTOCOL_SHA)
    for name in EXTRA_FILES:
        add_binding(files, name, sha256(safe_file(repo, name)))
    for name, digest in protocol['shards'].items():
        require_hash(repo, name, digest)
    return files, protocol, report


def create(repo, output):
    relative = output.resolve().relative_to(repo.resolve())
    if len(relative.parts) != 3 or relative.parts[:2] != ('outputs', 'baselines'):
        raise ValueError('Snapshot must be outputs/baselines/<new-name>')
    if output.exists():
        raise FileExistsError('Do not overwrite an existing baseline snapshot')
    files, protocol, report = build_bindings(repo)
    git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True)
    distributions = sorted({(d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata['Name']})
    output.mkdir(parents=True, exist_ok=False)
    total_bytes = copy_bound_files(repo, output / 'repository', files)
    manifest = {
        'schema': 'chemvector-measured-reference-snapshot-0.1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'frozen_research_reference_not_deployment_promotion',
        'reference_profile': 'structure-v01016-structural-twin-ocr',
        'unchanged_deployment_profile': 'structure-v01010-direct-aromatic-ocr',
        'reference_metrics': report['arms']['v01016'], 'deployment_metrics': report['arms']['v01010'],
        'sample_count': 200, 'molecule_count': 50, 'evaluation_scope': 'reused_development_not_unseen_confirmation',
        'git_head_context': git_head, 'tracked_worktree_dirty': bool(dirty), 'tracked_status_context': dirty,
        'code_identity': 'copied file hashes, not git HEAD alone; uncommitted bound files are included',
        'files': files, 'file_count': len(files), 'total_bytes': total_bytes,
        'external_dataset_shards': protocol['shards'],
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'packages': [{'name': n, 'version': v} for n, v in distributions]},
        'reproduction_dependencies': 'dataset shards and compatible Python/CUDA environment are not bundled',
        'copy_policy': 'independent copies; no hardlinks, deletion, chmod, git commits or tags',
        'training_performed': False, 'promotion_authorized': False,
        'integrity_boundary': 'hash-checked local copy, not a digital signature or filesystem write lock',
    }
    manifest['manifest_sha256'] = manifest_digest(manifest)
    write_json(output / 'manifest.json', manifest)
    result = verify(output)
    write_json(output / 'verification.json', result)
    return result


def verify(output):
    manifest = json.loads(safe_file(output, 'manifest.json').read_text())
    if manifest.get('manifest_sha256') != manifest_digest(manifest):
        raise ValueError('Snapshot manifest digest differs')
    if manifest['file_count'] != len(manifest['files']):
        raise ValueError('Snapshot file count differs')
    root = output / 'repository'
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if actual != set(manifest['files']):
        raise ValueError('Missing or unexpected snapshot file')
    total_bytes = 0
    for name, digest in manifest['files'].items():
        total_bytes += require_hash(root, name, digest).stat().st_size
    if total_bytes != manifest['total_bytes']:
        raise ValueError('Snapshot size differs')
    return {'verified': True, 'manifest_sha256': manifest['manifest_sha256'],
            'files': len(manifest['files']), 'total_bytes': total_bytes,
            'external_dataset_shards_verified': False, 'environment_recreated': False,
            'reference_profile': manifest['reference_profile']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('create', 'verify'))
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    result = create(repo, output) if args.action == 'create' else verify(output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
