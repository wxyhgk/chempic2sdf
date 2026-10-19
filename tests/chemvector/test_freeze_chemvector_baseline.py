import json

import pytest

from tools.freeze_chemvector_baseline import (
    add_binding, copy_bound_files, create, manifest_digest, safe_file, sha256, verify, write_json,
)


@pytest.mark.parametrize('name', ['/etc/passwd', '../outside', 'a/../../b', 'a//b', './a', 'a\\b', ''])
def test_unsafe_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError): safe_file(tmp_path, name)


def test_symlink_input_rejected(tmp_path):
    source = tmp_path / 'source'
    write_json(source, {'x': 1})
    (tmp_path / 'link').symlink_to(source)
    with pytest.raises(ValueError, match='symlink'): safe_file(tmp_path, 'link')


def test_conflicting_binding_rejected():
    files = {}
    add_binding(files, 'a', 'one')
    add_binding(files, 'a', 'one')
    with pytest.raises(ValueError): add_binding(files, 'a', 'two')


def fixture_snapshot(tmp_path):
    repo, output = tmp_path / 'source', tmp_path / 'snapshot'
    repo.mkdir()
    output.mkdir()
    write_json(repo / 'a.json', {'value': 7})
    files = {'a.json': sha256(repo / 'a.json')}
    size = copy_bound_files(repo, output / 'repository', files)
    manifest = {'files': files, 'file_count': 1, 'total_bytes': size, 'reference_profile': 'test'}
    manifest['manifest_sha256'] = manifest_digest(manifest)
    write_json(output / 'manifest.json', manifest)
    return repo, output


def test_independent_copy_not_hardlink_and_verifies_without_workspace(tmp_path):
    repo, output = fixture_snapshot(tmp_path)
    assert (repo / 'a.json').stat().st_ino != (output / 'repository/a.json').stat().st_ino
    (repo / 'a.json').unlink()
    assert verify(output)['verified']


@pytest.mark.parametrize('change', ['content', 'missing', 'extra', 'manifest'])
def test_snapshot_mutations_detected(tmp_path, change):
    _, output = fixture_snapshot(tmp_path)
    path = output / 'repository/a.json'
    if change == 'content':
        path.unlink()
        write_json(path, {'value': 8})
    elif change == 'missing': path.unlink()
    elif change == 'extra': write_json(output / 'repository/b.json', {})
    else:
        manifest = json.loads((output / 'manifest.json').read_text())
        manifest['reference_profile'] = 'other'
        (output / 'manifest.json').unlink()
        write_json(output / 'manifest.json', manifest)
    with pytest.raises((ValueError, FileNotFoundError)): verify(output)


def test_changed_source_not_copied_under_old_hash(tmp_path):
    write_json(tmp_path / 'a', {})
    with pytest.raises(ValueError, match='changed'):
        copy_bound_files(tmp_path, tmp_path / 'snapshot', {'a': '0' * 64})


def test_create_cannot_overwrite_existing_snapshot(tmp_path):
    output = tmp_path / 'outputs/baselines/existing'
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError): create(tmp_path, output)


def test_create_rejects_broad_output_directory(tmp_path):
    with pytest.raises(ValueError): create(tmp_path, tmp_path / 'outputs')
