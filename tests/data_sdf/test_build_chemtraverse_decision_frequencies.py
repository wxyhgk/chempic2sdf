from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from Data.sdf.build_chemtraverse_decision_frequencies import (
    FREQUENCY_SCHEMA_VERSION,
    build_decision_frequencies,
)
from tests.data_sdf.test_chemmol_sequence import make_annotations
from tests.data_sdf.test_chemscene_v0_1 import make_layers


def _add_json(archive: tarfile.TarFile, name: str, payload: dict) -> None:
    content = json.dumps(payload).encode("utf-8")
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def _fixture(root: Path) -> tuple[Path, Path]:
    shard = root / "train-000000.tar"
    chemmol, render = make_annotations("C1CC1")
    layers = make_layers(chemmol, render)
    with tarfile.open(shard, "w") as archive:
        for key in ("sample-a", "sample-b"):
            _add_json(archive, key + ".chemmol.json", chemmol)
            _add_json(archive, key + ".render.json", render)
            _add_json(archive, key + ".layers.json", layers)
    manifest = root / "train.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"sample_key": key, "shard": str(shard.resolve())})
            for key in ("sample-a", "sample-b")
        )
        + "\n",
        encoding="utf-8",
    )
    return shard, manifest


def test_builder_is_deterministic_and_binds_train_manifest(tmp_path: Path) -> None:
    shard, manifest = _fixture(tmp_path)

    first = build_decision_frequencies([shard], manifest, "sdf")
    second = build_decision_frequencies([shard], manifest, "sdf")

    assert first == second
    assert first["schema_version"] == FREQUENCY_SCHEMA_VERSION
    assert first["sequence_schema_version"] == "chemtraverse-0.1"
    assert first["target_molecule_id"] == "sdf"
    assert first["samples"] == {"manifest": 2, "audited": 2}
    assert first["sample_manifest"]["sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert first["positions"]["valid"] == (
        first["positions"]["decision"] + first["positions"]["forced"]
    )
    assert first["positions"]["decision"] > 0
    assert first["decision_frequencies"]["[DOWN]"] > 0
    assert first["decision_frequencies"]["[LINK]"] > 0


def test_builder_rejects_manifest_shards_outside_declared_inputs(
    tmp_path: Path,
) -> None:
    shard, manifest = _fixture(tmp_path)
    other = tmp_path / "other.tar"
    with tarfile.open(other, "w"):
        pass

    with pytest.raises(ValueError, match="outside the declared"):
        build_decision_frequencies([other], manifest, "sdf")

    assert shard.is_file()
