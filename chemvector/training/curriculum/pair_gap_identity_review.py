"""One authorized fixed private review, with a durable no-retry reservation."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from . import private_review
from .pair_gap_review_request import OUTPUT as REQUEST, COMMITMENT
from .runtime import ResourceBudget, run_budgeted

OUTPUT = Path('outputs/audits/chemvector-pair-gap-private-identity-20260918')
AUTHORIZATION = Path('docs/model/training/releases/20260918-pair-gap-identity-authorization.json')
RUNNER_SHA = 'af0d6bd2808a6d595289abd28ac28b6b0510bf40ec72529899f9bb7d87b034d0'
WORKER_SHA = '779b01bdf3dd9bd024a6678b540575ae42adaf7c60ce3f82fd24c36e412d360e'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def reserve_once(path, record):
    """Consume permission before starting any data-side process.

    A crash may leave only this reservation. It is deliberately not reclaimed:
    absence of a receipt does not prove absence of a completed private query.
    """
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def invoke_once(request_dir, evaluation_root, receipt_dir, authorization, *, authorized=False):
    if authorized is not True:
        raise ValueError('Explicit fixed-batch identity-only authorization required')
    request_dir, receipt_dir, authorization = map(Path, (request_dir, receipt_dir, authorization))
    attempt = request_dir / 'private-attempt.json'
    if attempt.exists() or receipt_dir.exists():
        raise FileExistsError('Review already reserved/executed; inspect saved artifacts, never retry')
    auth = json.loads(authorization.read_bytes())
    request_path, candidates_path = request_dir / 'request.json', request_dir / 'candidate-identities.json'
    request = json.loads(request_path.read_bytes())
    if (auth.get('schema_version') != 'chemvector-fixed-identity-authorization-record-0.1'
            or auth.get('maximum_real_private_invocations') != 1 or auth.get('automatic_retries') != 0
            or any(auth.get(k) is not False for k in ('training_authorized', 'model_side_private_sample_access_authorized',
                'images_sdf_or_tar_access_authorized', 'source_permission_changes_authorized', 'evaluation_split_changes_authorized'))
            or Path(auth['request_directory']).resolve() != request_dir.resolve()
            or Path(auth['output_directory']).resolve() != receipt_dir.resolve()
            or Path(auth['evaluation_root']).resolve() != Path(evaluation_root).resolve()
            or auth['request_sha256'] != digest(request_path)
            or auth['candidate_identities_sha256'] != digest(candidates_path)
            or request['candidate_identities_sha256'] != auth['candidate_identities_sha256']
            or request.get('candidate_molecules') != 18 or request.get('candidate_images') != 18
            or request.get('training_authorized') is not False
            or request['evaluation_commitment_sha256'] != auth['evaluation_commitment_sha256']
            or auth.get('candidate_molecules') != 18 or auth.get('candidate_images') != 18
            or auth['runner_sha256'] != RUNNER_SHA or digest(private_review.__file__) != RUNNER_SHA
            or auth['worker_sha256'] != WORKER_SHA or digest(private_review.WORKER) != WORKER_SHA
            or auth['wrapper_sha256'] != digest(__file__)):
        raise ValueError('Fixed request/authorization/implementation binding mismatch')
    # Only public commitment/cache and explicit candidate-side evidence are
    # present in this generated request. Private manifests are not listed here.
    for path, expected in request['input_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Request input changed before the single authorized invocation')
    reserve_once(attempt, {'schema_version': 'chemvector-private-review-reservation-0.1',
        'request_sha256': auth['request_sha256'], 'candidate_identities_sha256': auth['candidate_identities_sha256'],
        'authorization_sha256': digest(authorization), 'reserved_at_utc': datetime.now(timezone.utc).isoformat(),
        'maximum_invocations': 1, 'automatic_retries': 0, 'status': 'reserved_no_replay',
        'training_authorized': False})
    # Keep the old, reviewed data-side isolation and receipt allowlist intact.
    return private_review.run(request_dir, evaluation_root, receipt_dir, authorized=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorized-identity-only', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.authorized_identity_only:
        parser.error('Explicit identity-only authorization is required')
    if args.worker:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('CPU-only identity review required')
        result = invoke_once(REQUEST, COMMITMENT.parent, OUTPUT / 'receipt', AUTHORIZATION, authorized=True)
        print(json.dumps({k: result[k] for k in ('status', 'training_authorized', 'failure_code',
            'candidate_molecules', 'candidate_images', 'candidate_overlap_counts_by_rule') if k in result}), flush=True)
        return 0 if result['status'] in {'checked', 'checked_conservative'} else 1
    receipt = run_budgeted([sys.executable, '-m', 'chemvector.training.curriculum.pair_gap_identity_review',
                           '--authorized-identity-only', '--worker'], output=OUTPUT, budget=ResourceBudget(240))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
