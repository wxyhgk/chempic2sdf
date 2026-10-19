"""CPU launcher guards; no real candidate rendering or model access in tests."""

import pytest

from chemvector.training.curriculum import pair_gap_generation as generation


def test_extending_input_hashes_cannot_replace_the_original_commitment(tmp_path):
    path = tmp_path / 'existing.meta.json'
    inputs = {str(path): 'original'}
    generation.bind_digest(inputs, path, 'original')
    generation.bind_digest(inputs, tmp_path / 'new.json', 'new')
    with pytest.raises(ValueError, match='Previously bound'):
        generation.bind_digest(inputs, path, 'changed')
    assert inputs == {str(path): 'original', str(tmp_path / 'new.json'): 'new'}


def test_fixed_cpu_generation_budget_and_no_training_authorization(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return {'status': 'completed'}
    monkeypatch.setattr(generation, 'run_budgeted', run)
    assert generation.main([]) == 0
    command, kwargs = calls[0]
    assert command[-2:] == ['chemvector.training.curriculum.pair_gap_generation', '--worker']
    assert kwargs['budget'].wall_seconds == 300 and kwargs['budget'].gpu_devices == ()
    assert 'training' not in kwargs and 'admission' not in kwargs
    assert kwargs['output'] == generation.OUTPUT


@pytest.mark.parametrize('option', ['--train', '--samples', '--font-size', '--resume'])
def test_no_training_expansion_or_render_retry_knobs(option):
    with pytest.raises(SystemExit): generation.main([option])


def test_failed_worker_is_not_retried(monkeypatch):
    calls = []
    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {'status': 'worker_failed'}
    monkeypatch.setattr(generation, 'run_budgeted', run)
    assert generation.main([]) == 1
    assert len(calls) == 1


@pytest.mark.parametrize('bad', ['cuda_visible', 'cuda_initialized', 'dataset_exists', 'plan_exists', 'missing_output'])
def test_scope_and_freshness_guard_runs_before_release_access(monkeypatch, tmp_path, bad):
    import torch
    from chemvector.training.curriculum import release
    output = tmp_path / 'audit'
    output.mkdir()
    dataset = tmp_path / 'dataset'
    monkeypatch.setattr(generation, 'DATA', dataset)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: bad == 'cuda_initialized')
    def deny(*args, **kwargs):
        raise AssertionError('Guard must precede corpus access')
    monkeypatch.setattr(release, 'verify_release', deny)
    if bad == 'cuda_visible': monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    elif bad == 'dataset_exists': dataset.mkdir()
    elif bad == 'plan_exists': (output / 'plan.json').touch()
    elif bad == 'missing_output': output = tmp_path / 'absent'
    with pytest.raises(ValueError): generation.run(output)
