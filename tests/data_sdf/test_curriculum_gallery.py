from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from Data.sdf.chemvector_curriculum_v0_1.gallery import write_gallery


def _inputs(tmp_path: Path, *, profile: str = "reference") -> tuple[list[dict], list[dict]]:
    samples_path = tmp_path / "samples"
    samples_path.mkdir()
    image = Image.new("RGB", (768, 768), "white")
    draw = ImageDraw.Draw(image)
    draw.line((350, 384, 405, 355), fill="black", width=3)
    image.save(samples_path / "m001_reference.png")
    for suffix in ("svg", "sdf", "graph.json", "layers.json", "meta.json"):
        (samples_path / f"m001_reference.{suffix}").write_text("{}", encoding="utf-8")
    molecules = [{
        "molecule_id": "m001",
        "canonical_smiles": "CC",
        "stages": ["atoms_bonds", "functional_groups"],
        "concepts": ["single_bond"],
        "names": ["ethane"],
    }]
    samples = [{
        "sample_key": "m001_reference",
        "molecule_id": "m001",
        "profile_id": profile,
        "stages": ["atoms_bonds", "functional_groups"],
        "measured": {"font_size": 18},
    }]
    return molecules, samples


def test_gallery_is_offline_and_keeps_originals_unchanged(tmp_path: Path) -> None:
    molecules, samples = _inputs(tmp_path)
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (tmp_path / "samples").iterdir()}
    write_gallery(tmp_path, molecules, samples, {"status": "quarantine"})
    document = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "quarantine" in document and "未进入训练" in document
    assert "768 × 768" in document and "不等于训练输入" in document
    assert '<select id="stage">' in document and '<select id="profile">' in document
    assert '<option value="reference" selected>' in document
    assert "http://" not in document and "https://" not in document
    assert "fetch(" not in document and "XMLHttpRequest" not in document
    for suffix in ("png", "svg", "sdf", "graph.json", "layers.json", "meta.json"):
        assert f'href="samples/m001_reference.{suffix}"' in document
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (tmp_path / "samples").iterdir()}
    assert before == after
    with Image.open(tmp_path / "reference-contact-01.png") as image:
        assert image.size == (1200, 362)
    assert not (tmp_path / "reference-contact-02.png").exists()


def test_user_metadata_is_html_escaped(tmp_path: Path) -> None:
    molecules, samples = _inputs(tmp_path, profile="small_font")
    malicious = '<script>alert("x")</script>'
    molecules[0]["names"] = [malicious]
    samples[0]["measured"] = {"note": malicious}
    write_gallery(tmp_path, molecules, samples, {"note": malicious})
    document = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert malicious not in document
    assert "&lt;script&gt;" in document
    assert not list(tmp_path.glob("reference-contact-*.png"))


@pytest.mark.parametrize("key", ["../outside", "/absolute", "x/y", "x\\y", 'x\" onclick=\"oops', ""])
def test_unsafe_keys_fail_before_writing(tmp_path: Path, key: str) -> None:
    molecules, samples = _inputs(tmp_path)
    samples[0]["sample_key"] = key
    with pytest.raises(ValueError, match="Unsafe sample_key"):
        write_gallery(tmp_path, molecules, samples, {})
    assert not (tmp_path / "index.html").exists()


def test_missing_artifact_fails_before_writing(tmp_path: Path) -> None:
    molecules, samples = _inputs(tmp_path)
    (tmp_path / "samples" / "m001_reference.sdf").unlink()
    with pytest.raises(ValueError, match="Missing or symlinked"):
        write_gallery(tmp_path, molecules, samples, {})
    assert not (tmp_path / "index.html").exists()


def test_existing_gallery_is_not_overwritten(tmp_path: Path) -> None:
    molecules, samples = _inputs(tmp_path)
    (tmp_path / "index.html").write_text("existing review", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_gallery(tmp_path, molecules, samples, {})
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "existing review"


def test_duplicate_ids_and_unknown_molecule_fail(tmp_path: Path) -> None:
    molecules, samples = _inputs(tmp_path)
    with pytest.raises(ValueError, match="Duplicate molecule_id"):
        write_gallery(tmp_path, molecules + molecules, samples, {})
    with pytest.raises(ValueError, match="Duplicate sample_key"):
        write_gallery(tmp_path, molecules, samples + samples, {})
    samples[0]["molecule_id"] = "unknown"
    with pytest.raises(ValueError, match="Unknown molecule_id"):
        write_gallery(tmp_path, molecules, samples, {})


def test_empty_gallery_is_valid(tmp_path: Path) -> None:
    write_gallery(tmp_path, [], [], {"status": "empty draft"})
    document = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "共 0 个去重结构、0 张样式变体" in document
    assert not list(tmp_path.glob("reference-contact-*.png"))


def test_contact_sheets_paginate_without_dropping_structures(tmp_path: Path) -> None:
    molecules, samples = _inputs(tmp_path)
    for index in range(2, 22):
        molecule_id = f"m{index:03d}"
        key = f"{molecule_id}_reference"
        molecules.append({**molecules[0], "molecule_id": molecule_id})
        samples.append({**samples[0], "molecule_id": molecule_id, "sample_key": key})
        for suffix in ("png", "svg", "sdf", "graph.json", "layers.json", "meta.json"):
            shutil.copyfile(
                tmp_path / "samples" / f"m001_reference.{suffix}",
                tmp_path / "samples" / f"{key}.{suffix}",
            )
    write_gallery(tmp_path, molecules, samples, {})
    assert len(list(tmp_path.glob("reference-contact-*.png"))) == 2
    with Image.open(tmp_path / "reference-contact-01.png") as first:
        assert first.size == (1200, 1442)
    with Image.open(tmp_path / "reference-contact-02.png") as second:
        assert second.size == (1200, 362)
    document = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert document.count('<article class="sample"') == 21
    assert 'href="reference-contact-02.png"' in document


def test_symlinked_sample_directory_is_rejected(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "samples").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinked sample directory"):
        write_gallery(tmp_path, [], [], {})
    assert not (tmp_path / "index.html").exists()
