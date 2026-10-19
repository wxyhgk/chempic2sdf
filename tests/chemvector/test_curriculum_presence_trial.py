"""Synthetic CPU fitting/publication tests; no real data or checkpoints."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from chemvector.training.curriculum import presence_probe as probe, presence_trial as trial, presence_release as release
from chemvector.training.curriculum.runtime import StepLimit
from tests.chemvector.test_curriculum_pair_presence_stage import setup


def admission():
    return {'schema_version': 'chemvector-curriculum-admission-0.1', 'training_authorized': True,
            'external_overlap_status': 'checked', 'coverage_gaps': [], 'approved_manifest_sha256': 'synthetic'}


def caches():
    return [dict(sample_key='synthetic', purpose='training', training_authorized=True, manifest_sha256='synthetic',
                 hidden=torch.arange(32).reshape(4, 8).float() / 32,
                 targets=torch.tensor([True, False, False, False]), known=torch.tensor([True, True, False, False]))]


def test_cached_fit_updates_only_presence_and_logs_reservation_before_steps(monkeypatch):
    stage, *_ = setup(monkeypatch)
    before = probe.frozen_state_digest(stage.components)
    events = []
    result = probe.fit(stage, caches(), admission=admission(), manifest_sha256='synthetic',
        step_limit=StepLimit(3), emit=lambda event, **row: events.append((event, row)))
    assert result['optimizer_steps'] == 3 and result['known_pairs'] == 2
    assert result['positive_pairs'] == result['negative_pairs'] == 1
    assert result['final_loss'] < result['initial_loss']
    assert [e for e, _ in events] == ['step_reserved', 'step'] * 3
    assert [v['step'] for e, v in events if e == 'step_reserved'] == [1, 2, 3]
    assert before == probe.frozen_state_digest(stage.components)


@pytest.mark.parametrize('bad', ['validation', 'unapproved', 'binding', 'nan', 'grad', 'unknown_positive', 'empty', 'duplicate'])
def test_bad_caches_cannot_reach_optimizer(bad):
    rows = caches()
    if bad == 'validation': rows[0]['purpose'] = 'diagnostic'
    elif bad == 'unapproved': rows[0]['training_authorized'] = False
    elif bad == 'binding': rows[0]['manifest_sha256'] = 'changed'
    elif bad == 'nan': rows[0]['hidden'][0, 0] = float('nan')
    elif bad == 'grad': rows[0]['hidden'].requires_grad_(True)
    elif bad == 'unknown_positive': rows[0]['targets'][2] = True
    elif bad == 'empty': rows[0]['known'].zero_(); rows[0]['targets'].zero_()
    else: rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError): probe.fitting_tensors(rows, 'synthetic')


@pytest.mark.parametrize('bad', ['admission', 'manifest', 'budget', 'resume'])
def test_fit_rejects_bad_authority_before_any_parameter_update(monkeypatch, bad):
    stage, *_ = setup(monkeypatch)
    original = {k: v.clone() for k, v in stage.state_dict().items()}
    approved, limit = admission(), StepLimit(2)
    if bad == 'admission': approved['training_authorized'] = False
    elif bad == 'manifest': approved['approved_manifest_sha256'] = 'changed'
    elif bad == 'budget': limit = StepLimit(201)
    else: limit.consume()
    with pytest.raises(ValueError):
        probe.fit(stage, caches(), admission=approved, manifest_sha256='synthetic', step_limit=limit, emit=lambda *a, **k: None)
    assert all(torch.equal(original[k], v) for k, v in stage.state_dict().items())


def test_interrupted_reserved_step_does_not_allow_resume(monkeypatch):
    stage, *_ = setup(monkeypatch)
    limit = StepLimit(2)
    def stop(event, **record):
        raise TimeoutError('synthetic interruption after reservation')
    with pytest.raises(TimeoutError):
        probe.fit(stage, caches(), admission=admission(), manifest_sha256='synthetic', step_limit=limit, emit=stop)
    assert limit.consumed == 1
    with pytest.raises(ValueError, match='Fresh bounded'):
        probe.fit(stage, caches(), admission=admission(), manifest_sha256='synthetic', step_limit=limit, emit=stop)


def test_exclusive_durable_invocation_is_not_reclaimed(tmp_path):
    auth = tmp_path / 'auth.json'; auth.write_text('{}')
    reservation = tmp_path / 'attempt.json'
    trial.reserve_once(reservation, auth)
    before = reservation.read_bytes()
    with pytest.raises(FileExistsError): trial.reserve_once(reservation, auth)
    assert reservation.read_bytes() == before


def test_launcher_check_only_never_reserves_or_runs_and_normal_run_is_cpu(monkeypatch, tmp_path):
    auth = tmp_path / 'auth.json'; auth.write_text('{}')
    monkeypatch.setattr(trial, 'AUTHORIZATION', auth)
    monkeypatch.setattr(trial, 'ATTEMPT', tmp_path / 'attempt.json')
    monkeypatch.setattr(trial, 'OUTPUT', tmp_path / 'output')
    monkeypatch.setattr(trial, 'authorize', lambda: ({}, admission(), {}))
    calls = []
    monkeypatch.setattr(trial, 'run_budgeted', lambda command, **kw: calls.append((command, kw)) or {'status': 'completed'})
    assert trial.main(['--check-only']) == 0 and not calls and not trial.ATTEMPT.exists()
    assert trial.main([]) == 0 and len(calls) == 1
    assert calls[0][1]['budget'].wall_seconds == 300 and calls[0][1]['budget'].gpu_devices == ()
    assert calls[0][1]['training'] is True
    with pytest.raises(FileExistsError): trial.main([])
    for arg in ('--gpu', '--max-steps', '--learning-rate'):
        with pytest.raises(SystemExit): trial.main([arg, '1'])


@pytest.mark.parametrize('train,valid', [('CCO', 'CCO'), ('CN', 'C[NH3+]'), ('Cc1ccccc1', 'Nc1ccccc1')])
def test_training_cannot_overlap_existing_validation_by_identity_connectivity_or_group(train, valid):
    with pytest.raises(ValueError):
        release.disjoint_training_validation([{'molecule_id': 'a', 'canonical_smiles': train}],
                                             [{'molecule_id': 'b', 'canonical_smiles': valid}])


@pytest.fixture
def publication(tmp_path, monkeypatch):
    """Opaque synthetic bundles; targets/images mocked, real copy/binding checks."""
    source = tmp_path / 'source'; (source / 'samples').mkdir(parents=True)
    rows, artifacts = [], {}
    for i in range(24):
        key = f'm{i}__reference'; new = i < 17
        flags = dict(split='quarantine' if new else 'train', training_authorized=not new,
                     external_overlap_status='unchecked' if new else 'checked')
        payloads = {s: b'synthetic' for s in release.SUFFIXES}
        for suffix in ('.graph.json', '.layers.json'):
            payloads[suffix] = json.dumps({**flags, 'canonical_smiles': 'CCO'}).encode()
        meta = dict(sample_key=key, molecule_id=f'm{i}', **flags,
                    sha256={s: sha256(v).hexdigest() for s, v in payloads.items() if s != '.meta.json'})
        payloads['.meta.json'] = json.dumps(meta).encode()
        for suffix, value in payloads.items():
            (source / 'samples' / (key + suffix)).write_bytes(value)
        rows.append(dict(root=str(source), sample_key=key, molecule_id=f'm{i}', canonical_smiles='CCO',
            meta_sha256=sha256(payloads['.meta.json']).hexdigest(),
            origin='new_quarantine' if new else 'existing_training_control'))
        artifacts[key] = {s: sha256(v).hexdigest() for s, v in payloads.items()}
    rights = tmp_path / 'rights.json'; rights.write_text('{}')
    plan = dict(samples=rows, validation_controls=[dict(sample_key=f'v{i}', root=str(source)) for i in range(16)],
                source_artifacts=artifacts, source_use_review_sha256=release.file_digest(rights),
                release_checks={k: 'checked' for k in release.CHECKS}, limitations=['synthetic only'])
    plan_path = tmp_path / 'plan.json'; release.write(plan_path, plan)
    review = dict(schema_version='chemvector-presence-promotion-review-0.1', decision='approve_local_presence_probe',
        approved_plan_sha256=release.file_digest(plan_path), reviewer='synthetic', rationale='test only',
        release_checks=plan['release_checks'], source_use_review_sha256=plan['source_use_review_sha256'])
    review_path = tmp_path / 'review.json'; review_path.write_text(json.dumps(review))
    for name, path in (('PLAN', plan_path), ('REVIEW', review_path), ('RIGHTS', rights), ('OUTPUT', tmp_path / 'published')):
        monkeypatch.setattr(release, name, path)
    monkeypatch.setattr(release, 'build_plan', lambda: deepcopy(plan))
    def load(root, key, *, purpose):
        assert purpose == 'training'
        meta = json.loads((Path(root) / 'samples' / (key + '.meta.json')).read_text())
        assert meta['training_authorized'] and meta['split'] == 'train'
        return {'meta': meta}
    monkeypatch.setattr(release, 'load_sample', load)
    monkeypatch.setattr(release, 'CurriculumCollator', lambda **kw: lambda samples: {})
    return source, plan


def test_independent_small_release_copies_exact_transitions_and_does_not_enter_mixed_launcher(publication):
    source, plan = publication
    before = {p: p.read_bytes() for p in (source / 'samples').iterdir()}
    report = release.publish()
    manifest, _ = release.verify()
    assert len(manifest['samples']) == 24 and len(manifest['validation_controls']) == 16
    assert report['byte_identical_copied_artifacts'] == 17 * 8 + 7 * 11
    assert all(p.read_bytes() == data for p, data in before.items())
    for row in manifest['samples']:
        for suffix in release.SUFFIXES - {'.meta.json', '.graph.json', '.layers.json'}:
            old = source / 'samples' / (row['sample_key'] + suffix)
            new = release.OUTPUT / 'samples' / old.name
            assert new.read_bytes() == old.read_bytes() and new.stat().st_ino != old.stat().st_ino
    from chemvector.training.curriculum.release import verify_release
    with pytest.raises(ValueError, match='Unsupported release'):
        verify_release(release.OUTPUT / 'manifest.json', release.OUTPUT / 'admission.json')
    with pytest.raises(FileExistsError): release.publish()


@pytest.mark.parametrize('mutation', ['review', 'source', 'final_verify'])
def test_failed_publication_does_not_leave_admission(publication, monkeypatch, mutation):
    source, plan = publication
    if mutation == 'review':
        review = json.loads(release.REVIEW.read_text()); review['decision'] = 'pending'
        release.REVIEW.write_text(json.dumps(review))
    elif mutation == 'source': (source / 'samples' / 'm0__reference.png').write_bytes(b'changed')
    else:
        def fail(): raise ValueError('synthetic final verification failure')
        monkeypatch.setattr(release, 'verify', fail)
    with pytest.raises(ValueError): release.publish()
    assert not (release.OUTPUT / 'admission.json').exists()


@pytest.mark.parametrize('mutation', ['pixels', 'labels', 'manifest', 'admission'])
def test_published_evidence_tampering_is_rejected(publication, mutation):
    release.publish()
    if mutation == 'pixels': (release.OUTPUT / 'samples/m0__reference.png').write_bytes(b'changed')
    elif mutation == 'labels': (release.OUTPUT / 'samples/m0__reference.graph.json').write_text('{}')
    elif mutation == 'manifest': (release.OUTPUT / 'manifest.json').write_text('{}')
    else:
        path = release.OUTPUT / 'admission.json'
        value = json.loads(path.read_text()); value['training_authorized'] = False
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError): release.verify()
