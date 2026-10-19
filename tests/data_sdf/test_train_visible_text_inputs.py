import csv
import hashlib
import io
import json
import tarfile

import pytest

from Data.sdf import train_visible_text_inputs as inputs


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":")) + "\n").encode()


def make_dataset(tmp_path, monkeypatch, *, bad_member=None, layer_split="train"):
    root = tmp_path / "style_train_100k"
    root.mkdir()
    (root / "shards").mkdir()
    monkeypatch.setattr(inputs, "APPROVED_ROOT", root)
    rows = []
    archive_path = root / "shards/train-000000.tar"
    with tarfile.open(archive_path, "w") as archive:
        for index in range(3):
            for variant in inputs.VARIANTS:
                key = f"train_{index:06d}_{variant}"
                molecule = f"synthetic:{index}"
                row = {"sample_key": key, "molecule_id": molecule, "split": "train",
                       "variant": variant, "output_shard": "shards/train-000000.tar"}
                for kind, suffix in inputs.SUFFIXES.items():
                    document = {"molecule_id": molecule, "variant": variant,
                                "sample_key": key, "split": layer_split, "atoms": []}
                    payload = b"opaque PNG bytes" if kind == "png" else encode(document)
                    row[kind + "_sha256"] = digest(payload)
                    member = tarfile.TarInfo(key + suffix)
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
                rows.append(row)
        if bad_member:
            member = tarfile.TarInfo(bad_member)
            archive.addfile(member, io.BytesIO(b""))
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (root / "samples.csv").write_text(csv_buffer.getvalue())
    commitment = {"schema_version": "chemvector-joint-commitment-1.0",
                  "dataset": "style_train_100k", "samples": 12, "molecules": 3,
                  "split_counts": {"train": 12},
                  "samples_csv": [record(root, "samples.csv")],
                  "tars": [record(root, "shards/train-000000.tar")]}
    commit(root, commitment)
    return root


def record(root, relative):
    payload = (root / relative).read_bytes()
    return {"path": relative, "bytes": len(payload), "sha256": digest(payload)}


def commit(root, value):
    value.pop("commitment_sha256", None)
    value["commitment_sha256"] = digest(encode(value))
    (root / "commitment.json").write_bytes(encode(value))


def test_selection_deterministic_complete_groups_and_readonly_payloads(tmp_path, monkeypatch):
    root = make_dataset(tmp_path, monkeypatch)
    chosen = inputs.freeze_selection(root, 2)
    assert chosen == inputs.freeze_selection(root, 2)
    assert len(chosen["selection"]) == 8
    expected = sorted((f"synthetic:{i}" for i in range(3)),
                      key=lambda value: digest((inputs.SELECTION_SALT + value).encode()))[:2]
    assert chosen["molecule_ids"] == expected
    samples = list(inputs.iter_selected_samples(root, chosen))
    assert len(samples) == 8
    assert all(set(sample["payloads"]) == set(inputs.SUFFIXES) for sample in samples)
    assert all(sample["layers"]["split"] == "train" for sample in samples)
    assert sorted(path.name for path in root.iterdir()) == ["commitment.json", "samples.csv", "shards"]


@pytest.mark.parametrize("count", [0, -1, True, 4, None])
def test_bad_counts(tmp_path, monkeypatch, count):
    root = make_dataset(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        inputs.freeze_selection(root, count)


def test_unapproved_root_rejected_before_read(tmp_path, monkeypatch):
    root = make_dataset(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="approved"):
        inputs.freeze_selection(root.parent / "sealed_test", 1)


@pytest.mark.parametrize("relative", ["samples.csv", "shards/train-000000.tar"])
def test_changed_source_rejected(tmp_path, monkeypatch, relative):
    root = make_dataset(tmp_path, monkeypatch)
    chosen = inputs.freeze_selection(root, 1)
    path = root / relative
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        list(inputs.iter_selected_samples(root, chosen))


def test_commitment_self_hash_and_selection_tamper(tmp_path, monkeypatch):
    root = make_dataset(tmp_path, monkeypatch)
    chosen = inputs.freeze_selection(root, 1)
    chosen["selection"].pop()
    with pytest.raises(ValueError, match="selection differs"):
        list(inputs.iter_selected_samples(root, chosen))
    value = json.loads((root / "commitment.json").read_bytes())
    value["samples"] = 99
    (root / "commitment.json").write_bytes(encode(value))
    with pytest.raises(ValueError, match="self-hash"):
        inputs.freeze_selection(root, 1)


@pytest.mark.parametrize("relative", ["../sealed_test/data.tar", "/tmp/train-000000.tar",
                                     "shards/fresh_dev-000000.tar"])
def test_commitment_path_boundary(tmp_path, monkeypatch, relative):
    root = make_dataset(tmp_path, monkeypatch)
    value = json.loads((root / "commitment.json").read_bytes())
    value["tars"][0]["path"] = relative
    commit(root, value)
    with pytest.raises(ValueError):
        inputs.freeze_selection(root, 1)


def test_symlink_manifest_rejected(tmp_path, monkeypatch):
    root = make_dataset(tmp_path, monkeypatch)
    original = root / "samples.csv"
    target = root / "copied.csv"
    original.rename(target)
    original.symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        inputs.freeze_selection(root, 1)


@pytest.mark.parametrize("name", ["../train_000000_original.png", "sealed_test.png",
                                 "train_000000_original.png"])
def test_unsafe_and_duplicate_tar_members(tmp_path, monkeypatch, name):
    root = make_dataset(tmp_path, monkeypatch, bad_member=name)
    chosen = inputs.freeze_selection(root, 1)
    with pytest.raises(ValueError):
        list(inputs.iter_selected_samples(root, chosen))


def test_selected_layers_cannot_be_eval_even_with_valid_hashes(tmp_path, monkeypatch):
    root = make_dataset(tmp_path, monkeypatch, layer_split="fresh_dev")
    chosen = inputs.freeze_selection(root, 1)
    with pytest.raises(ValueError, match="selected train"):
        list(inputs.iter_selected_samples(root, chosen))


@pytest.mark.parametrize("field,value", [("split", "sealed_test"),
                                       ("sample_key", "../train_000000_original"),
                                       ("output_shard", "../sealed_test/train-000000.tar"),
                                       ("variant", "stress")])
def test_manifest_row_boundaries(tmp_path, monkeypatch, field, value):
    root = make_dataset(tmp_path, monkeypatch)
    rows = list(csv.DictReader(io.StringIO((root / "samples.csv").read_text())))
    rows[0][field] = value
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (root / "samples.csv").write_text(buffer.getvalue())
    commitment = json.loads((root / "commitment.json").read_bytes())
    commitment["samples_csv"] = [record(root, "samples.csv")]
    commit(root, commitment)
    with pytest.raises(ValueError):
        inputs.freeze_selection(root, 1)


def test_tar_link_refused_without_following_target(tmp_path, monkeypatch):
    root = make_dataset(tmp_path, monkeypatch)
    with tarfile.open(root / "shards/train-000000.tar", "a") as archive:
        member = tarfile.TarInfo("train_999999_original.png")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../sealed_test/missing.png"
        archive.addfile(member)
    commitment = json.loads((root / "commitment.json").read_bytes())
    commitment["tars"] = [record(root, "shards/train-000000.tar")]
    commit(root, commitment)
    chosen = inputs.freeze_selection(root, 1)
    with pytest.raises(ValueError, match="Unsafe/nontrain"):
        list(inputs.iter_selected_samples(root, chosen))
