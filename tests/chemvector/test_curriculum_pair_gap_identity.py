"""Synthetic request/reservation tests; never call a private data custodian."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from chemvector.training.curriculum.admission import identities
from chemvector.training.curriculum.pair_gap_data import candidate_entries
from chemvector.training.curriculum.pair_gap_review_request import assemble_candidates
from chemvector.training.curriculum import pair_gap_identity_review as once


def candidate_fixture():
    entries = candidate_entries()
    samples = [dict(molecule_id=e['molecule_id'], sample_key=e['molecule_id'] + '__reference',
                    profile_id='reference') for e in entries]
    handoff = dict(status='not_submitted', training_authorized=False, external_overlap_status='unchecked',
        candidates=[dict(molecule_id=e['molecule_id'], canonical_smiles=e['canonical_smiles'],
                         **identities(e['canonical_smiles']), released_overlap_fields=[]) for e in entries])
    return entries, samples, handoff


def test_request_preserves_all_fixed_identities_and_variants_without_reading_files(monkeypatch):
    values = candidate_fixture()
    before = deepcopy(values)
    def denied(*args, **kwargs):
        raise AssertionError('Pure candidate assembly cannot read an external index or pixels')
    monkeypatch.setattr(Path, 'read_bytes', denied)
    monkeypatch.setattr(Path, 'read_text', denied)
    result = assemble_candidates(*values)
    assert len(result) == 18 and sum(r['image_count'] for r in result) == 18
    assert all(r['source'] == 'curriculum' for r in result)
    assert {r['molecule_id'] for r in result} == {r['molecule_id'] for r in values[0]}
    assert values == before


@pytest.mark.parametrize('mutation', ['short', 'duplicate', 'style', 'key', 'training', 'identity',
                                    'overlap', 'submitted', 'handoff_duplicate'])
def test_changed_handoff_or_candidate_scope_fails_closed(mutation):
    entries, samples, handoff = candidate_fixture()
    if mutation == 'short': samples.pop()
    elif mutation == 'duplicate': entries[1] = deepcopy(entries[0])
    elif mutation == 'style': samples[0]['profile_id'] = 'font_small'
    elif mutation == 'key': samples[0]['sample_key'] = 'different__reference'
    elif mutation == 'training': entries[0]['training_authorized'] = True
    elif mutation == 'identity': handoff['candidates'][0]['inchikey'] = 'wrong'
    elif mutation == 'overlap': handoff['candidates'][0]['released_overlap_fields'] = ['group_key']
    elif mutation == 'submitted': handoff['status'] = 'submitted'
    else: handoff['candidates'].append(deepcopy(handoff['candidates'][0]))
    with pytest.raises(ValueError): assemble_candidates(entries, samples, handoff)


def write(path, value):
    path.write_text(json.dumps(value))


def request_fixture(tmp_path):
    root = tmp_path / 'request'
    root.mkdir()
    candidates = root / 'candidate-identities.json'
    write(candidates, ['synthetic'] * 18)
    request = dict(schema_version='chemvector-identity-review-request-0.1', training_authorized=False,
        candidate_molecules=18, candidate_images=18, input_sha256={}, evaluation_commitment_sha256='synthetic',
        candidate_identities_sha256=once.digest(candidates))
    write(root / 'request.json', request)
    output, evaluation = tmp_path / 'receipt', tmp_path / 'synthetic-evaluation-not-opened'
    auth = dict(schema_version='chemvector-fixed-identity-authorization-record-0.1',
        maximum_real_private_invocations=1, automatic_retries=0, training_authorized=False,
        model_side_private_sample_access_authorized=False, images_sdf_or_tar_access_authorized=False,
        source_permission_changes_authorized=False, evaluation_split_changes_authorized=False,
        request_directory=str(root), output_directory=str(output), evaluation_root=str(evaluation),
        request_sha256=once.digest(root / 'request.json'), candidate_identities_sha256=once.digest(candidates),
        candidate_molecules=18, candidate_images=18, evaluation_commitment_sha256='synthetic',
        runner_sha256=once.RUNNER_SHA, worker_sha256=once.WORKER_SHA, wrapper_sha256=once.digest(once.__file__))
    authorization = tmp_path / 'authorization.json'
    write(authorization, auth)
    return root, evaluation, output, authorization


def test_one_successful_call_reserves_before_execution_and_never_replays(monkeypatch, tmp_path):
    args = request_fixture(tmp_path)
    calls = []
    def fake(*inputs, **kwargs):
        reservation = json.loads((args[0] / 'private-attempt.json').read_text())
        assert reservation['maximum_invocations'] == 1 and reservation['automatic_retries'] == 0
        assert kwargs['authorized'] is True
        calls.append(inputs)
        return {'status': 'checked', 'training_authorized': False}
    monkeypatch.setattr(once.private_review, 'run', fake)
    result = once.invoke_once(*args, authorized=True)
    assert result['status'] == 'checked' and len(calls) == 1
    with pytest.raises(FileExistsError): once.invoke_once(*args, authorized=True)
    assert len(calls) == 1


def test_interruption_after_reservation_still_consumes_attempt(monkeypatch, tmp_path):
    args = request_fixture(tmp_path)
    calls = []
    def interrupted(*inputs, **kwargs):
        calls.append(inputs)
        raise TimeoutError('Synthetic interruption, no private data')
    monkeypatch.setattr(once.private_review, 'run', interrupted)
    with pytest.raises(TimeoutError): once.invoke_once(*args, authorized=True)
    assert (args[0] / 'private-attempt.json').is_file() and not args[2].exists()
    with pytest.raises(FileExistsError): once.invoke_once(*args, authorized=True)
    assert len(calls) == 1


@pytest.mark.parametrize('mutation', ['authorization', 'retry', 'training', 'request', 'candidate', 'wrapper', 'output'])
def test_bad_authority_or_binding_cannot_reserve_or_start_review(monkeypatch, tmp_path, mutation):
    args = request_fixture(tmp_path)
    def forbidden(*a, **kw):
        raise AssertionError('No data-side execution allowed')
    monkeypatch.setattr(once.private_review, 'run', forbidden)
    auth = json.loads(args[3].read_text())
    if mutation == 'retry': auth['automatic_retries'] = 1
    elif mutation == 'training': auth['training_authorized'] = True
    elif mutation == 'request': auth['request_sha256'] = 'changed'
    elif mutation == 'candidate': auth['candidate_identities_sha256'] = 'changed'
    elif mutation == 'wrapper': auth['wrapper_sha256'] = 'changed'
    elif mutation == 'output': auth['output_directory'] = str(tmp_path / 'elsewhere')
    write(args[3], auth)
    with pytest.raises(ValueError): once.invoke_once(*args, authorized=mutation != 'authorization')
    assert not (args[0] / 'private-attempt.json').exists()


def test_exclusive_reservation_does_not_overwrite_old_attempt(tmp_path):
    path = tmp_path / 'attempt.json'
    once.reserve_once(path, {'attempt': 1})
    saved = path.read_bytes()
    with pytest.raises(FileExistsError): once.reserve_once(path, {'attempt': 2})
    assert path.read_bytes() == saved


def test_launcher_requires_explicit_permission_and_uses_only_cpu(monkeypatch):
    calls = []
    monkeypatch.setattr(once, 'run_budgeted', lambda command, **kwargs: calls.append((command, kwargs)) or {'status': 'completed'})
    with pytest.raises(SystemExit): once.main([])
    assert not calls
    assert once.main(['--authorized-identity-only']) == 0
    command, kwargs = calls[0]
    assert command[-2:] == ['--authorized-identity-only', '--worker']
    assert kwargs['budget'].wall_seconds == 240 and kwargs['budget'].gpu_devices == ()
    assert 'training' not in kwargs
