"""Synthetic-only integration checks for the quarantined curriculum pilot."""

from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cairosvg
import numpy as np
from PIL import Image
import pytest
from rdkit import Chem

from Data.sdf.chemvector_curriculum_v0_1 import atoms_bonds, functional_groups
import Data.sdf.chemvector_curriculum_v0_1.build as build_module
from Data.sdf.chemvector_curriculum_v0_1.build import build
from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from Data.sdf.chemvector_curriculum_v0_1.rendering import (
    make_molecule,
    profiles,
    render,
    sdf_bytes,
)


def _entry(smiles):
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    return next(row for row in merged_records() if row["canonical_smiles"] == canonical)


def _pixels(data, mode=None):
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert(mode) if mode else image).copy()


def _mol_from_sdf(data):
    mol = Chem.MolFromMolBlock(
        data.decode().split("$$$$")[0], sanitize=True, removeHs=False, strictParsing=True
    )
    assert mol is not None
    return mol


@pytest.fixture(scope="module")
def paired_samples():
    # One small molecule with a single O glyph avoids conflating font size
    # with the separately rejected OH-letter-spacing cases in the pilot.
    return {profile.profile_id: render(_entry("CC=O"), profile) for profile in profiles()}


@pytest.fixture(scope="module")
def chemical_samples(paired_samples):
    # Charge, two-letter labels and aromatic chemical bonds supplement the
    # simple carbonyl without opening any existing training/evaluation data.
    return {
        "carbonyl": paired_samples["reference"],
        **{smiles: render(_entry(smiles), profiles()[0]) for smiles in ("C[O-]", "CCl", "c1ccncc1")},
    }


def test_catalogue_merges_108_teaching_records_into_101_unique_molecules():
    sources = atoms_bonds.records() + functional_groups.records()
    entries = merged_records()
    assert len(sources) == 108
    assert len(entries) == 101
    assert len({entry["canonical_smiles"] for entry in entries}) == 101
    assert sum(len(entry["source_record_ids"]) for entry in entries) == 108
    assert Counter(stage for entry in entries for stage in entry["stages"]) == {
        "atoms_bonds": 44, "functional_groups": 64
    }
    assert sum(len(entry["stages"]) == 2 for entry in entries) == 7
    by_id = {entry["molecule_id"]: entry for entry in entries}
    for entry in entries:
        expected_id = "curr_" + sha256(entry["canonical_smiles"].encode()).hexdigest()[:16]
        assert entry["molecule_id"] == expected_id
        assert entry["split"] == "quarantine"
        assert entry["training_authorized"] is False
        assert entry["external_overlap_status"] == "unchecked"
        assert entry["source_record_ids"] == sorted(set(entry["source_record_ids"]))
    for source in sources:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(source["smiles"]))
        destination = by_id["curr_" + sha256(smiles.encode()).hexdigest()[:16]]
        assert source["record_id"] in destination["source_record_ids"]
        assert source["stage"] in destination["stages"]
        assert set(source["concepts"]) <= set(destination["concepts"])


def test_catalogue_order_does_not_change_identity_or_metadata():
    sources = atoms_bonds.records() + functional_groups.records()
    assert merged_records(list(reversed(sources))) == merged_records(sources)
    assert merged_records() == merged_records()
    duplicate_id = [sources[0], {**sources[0], "smiles": "CCC"}]
    with pytest.raises(ValueError, match="Duplicate catalog record ID"):
        merged_records(duplicate_id)


def test_nine_profiles_change_only_the_named_axis():
    variants = profiles()
    reference = asdict(variants[0])
    expected_axes = {
        "font_small": "font_size", "font_large": "font_size",
        "bond_short": "bond_length_cap", "bond_long": "bond_length_cap",
        "line_thin": "bond_line_width", "line_thick": "bond_line_width",
        "serif": "font_family", "rotated": "rotation_degrees",
    }
    assert len(variants) == 9
    assert {profile.profile_id for profile in variants} == {"reference", *expected_axes}
    for profile in variants[1:]:
        changed = {key for key, value in asdict(profile).items() if value != reference[key]}
        assert changed == {"profile_id", expected_axes[profile.profile_id]}


def test_all_catalogue_sdf_roundtrips_preserve_atom_and_bond_indices():
    # Cheap chemistry-only loop: no rendering all 101 structures in this test.
    for entry in merged_records():
        original = make_molecule(entry["canonical_smiles"])
        restored = _mol_from_sdf(sdf_bytes(original))
        assert Chem.MolToSmiles(restored) == entry["canonical_smiles"]
        assert restored.GetNumAtoms() == original.GetNumAtoms()
        assert restored.GetNumBonds() == original.GetNumBonds()
        for left, right in zip(original.GetAtoms(), restored.GetAtoms()):
            assert (left.GetIdx(), left.GetAtomicNum(), left.GetIsotope(), left.GetFormalCharge()) == (
                right.GetIdx(), right.GetAtomicNum(), right.GetIsotope(), right.GetFormalCharge()
            )
        for left, right in zip(original.GetBonds(), restored.GetBonds()):
            assert (left.GetIdx(), {left.GetBeginAtomIdx(), left.GetEndAtomIdx()}, left.GetBondType()) == (
                right.GetIdx(), {right.GetBeginAtomIdx(), right.GetEndAtomIdx()}, right.GetBondType()
            )


def test_chiral_sdf_roundtrip_preserves_stereo_and_undirected_edge_indices():
    original = make_molecule("F[C@H](Cl)Br")
    restored = _mol_from_sdf(sdf_bytes(original))
    assert Chem.MolToSmiles(restored, isomericSmiles=True) == Chem.MolToSmiles(original, isomericSmiles=True)
    assert Chem.MolToSmiles(restored, isomericSmiles=True) != Chem.MolToSmiles(
        Chem.MolFromSmiles("F[C@@H](Cl)Br"), isomericSmiles=True
    )
    for left, right in zip(original.GetBonds(), restored.GetBonds()):
        assert left.GetIdx() == right.GetIdx()
        assert {left.GetBeginAtomIdx(), left.GetEndAtomIdx()} == {right.GetBeginAtomIdx(), right.GetEndAtomIdx()}
        assert left.GetBondType() == right.GetBondType()


@pytest.mark.parametrize("sample_name", ["carbonyl", "C[O-]", "CCl", "c1ccncc1"])
def test_graph_sdf_and_layer_indices_refer_to_the_same_chemistry(chemical_samples, sample_name):
    files, meta = chemical_samples[sample_name]
    graph = json.loads(files[".graph.json"])
    layers = json.loads(files[".layers.json"])
    mol = _mol_from_sdf(files[".sdf"])
    assert Chem.MolToSmiles(mol) == graph["canonical_smiles"] == meta["canonical_smiles"]
    assert graph["atom_id_convention"] == "zero-based SDF atom order"
    assert len(graph["atoms"]) == len(layers["atoms"]) == mol.GetNumAtoms()
    assert len(graph["bonds"]) == len(layers["bonds"]) == mol.GetNumBonds()
    for atom, row, layer in zip(mol.GetAtoms(), graph["atoms"], layers["atoms"]):
        assert atom.GetIdx() == row["atom_id"] == layer["atom_id"]
        assert atom.GetSymbol() == row["element"] == layer["element"]
        assert atom.GetFormalCharge() == row["formal_charge"] == layer["formal_charge"]
        assert atom.GetIsotope() == row["isotope"] == layer["isotope"]
        assert atom.GetIsAromatic() == row["aromatic"] == layer["aromatic"]
    for bond, row, layer in zip(mol.GetBonds(), graph["bonds"], layers["bonds"]):
        assert bond.GetIdx() == row["bond_id"] == layer["bond_id"]
        assert bond.GetBeginAtomIdx() == row["begin_atom"] == layer["begin_atom"]
        assert bond.GetEndAtomIdx() == row["end_atom"] == layer["end_atom"]
        assert str(bond.GetBondType()) == row["chemical_type"] == layer["chemical_type"]
    assert graph["functional_groups"] == functional_groups.annotate_groups(mol)
    assert graph["functional_groups"] == meta["functional_groups"]
    assert graph["training_authorized"] is False


@pytest.mark.parametrize("profile_id", [profile.profile_id for profile in profiles()])
def test_same_final_svg_produces_png_masks_and_character_boxes(paired_samples, profile_id):
    files, meta = paired_samples[profile_id]
    layers = json.loads(files[".layers.json"])
    records = json.loads(files[".text-records.json"])
    source_png = _pixels(files[".png"], "RGB")
    independent_png = _pixels(cairosvg.svg2png(bytestring=files[".svg"]), "RGB")
    np.testing.assert_array_equal(source_png, independent_png)
    text, bond, auxiliary = [
        _pixels(files[suffix]) > 0 for suffix in (".text.png", ".bond.png", ".auxiliary.png")
    ]
    foreground = source_png.min(axis=2) < 255
    np.testing.assert_array_equal(foreground, text | bond | auxiliary)
    assert not (text & (bond | auxiliary)).any()
    semantic = bond.astype(np.uint8) | (text.astype(np.uint8) << 1) | (auxiliary.astype(np.uint8) << 2)
    np.testing.assert_array_equal(_pixels(files[".semantic.png"]), semantic)
    assert layers["semantic_encoding"]["kind"] == "bitset"
    glyph_union = np.zeros_like(text)
    for glyph in records["glyphs"]:
        root = ET.fromstring(files[".svg"])
        node = root[glyph["node_index"]]
        assert node.attrib == glyph["svg_attributes"]
        assert node.text == glyph["text"]
        assert node.attrib["id"] == glyph["node_id"]
        for child in list(root):
            if child is not node:
                root.remove(child)
        # Independent isolated rasterization, retaining only the exact final
        # SVG text node, checks the pixel bbox rather than a font metric guess.
        isolated = cairosvg.svg2png(bytestring=ET.tostring(root))
        with Image.open(io.BytesIO(isolated)) as image:
            alpha = image.convert("RGBA").getchannel("A")
            assert list(alpha.getbbox()) == glyph["bbox"]
            glyph_union |= np.asarray(alpha) > 0
        atom = layers["atoms"][glyph["atom_id"]]
        assert glyph["node_id"] in atom["text_node_ids"]
    np.testing.assert_array_equal(glyph_union, text)
    for atom in layers["atoms"]:
        glyphs = [glyph for glyph in records["glyphs"] if glyph["atom_id"] == atom["atom_id"]]
        if not glyphs:
            assert atom["visible_text"] is False
            assert atom["bbox"] is None
        else:
            expected_box = [
                min(glyph["bbox"][0] for glyph in glyphs), min(glyph["bbox"][1] for glyph in glyphs),
                max(glyph["bbox"][2] for glyph in glyphs), max(glyph["bbox"][3] for glyph in glyphs),
            ]
            assert atom["bbox"] == expected_box
    assert meta["same_source_raster"] is records["same_source_raster"] is True


def test_serif_is_consistent_across_final_svg_glyph_record_and_provenance(paired_samples):
    files, meta = paired_samples["serif"]
    layers = json.loads(files[".layers.json"])
    records = json.loads(files[".text-records.json"])
    provenance = layers["renderer"]
    assert meta["profile"]["font_family"] == "DejaVu Serif"
    assert provenance["config"] == meta["profile"]
    assert provenance["font"]["family"] == "DejaVu Serif"
    assert sha256(Path(provenance["font"]["path"]).read_bytes()).hexdigest() == provenance["font"]["sha256"]
    for glyph in records["glyphs"]:
        assert "font-family:DejaVu Serif" in glyph["svg_attributes"]["style"]
        assert "DejaVu Sans" not in glyph["svg_attributes"]["style"]
    assert b"font-family:DejaVu Serif" in files[".svg"]
    assert files[".png"] != paired_samples["reference"][0][".png"]
    assert provenance["resize_after_render"] is False
    assert provenance["legacy_pixel_equivalent"] is False


def test_font_and_bond_length_controls_change_actual_pixels_on_separate_axes(paired_samples):
    measured = {name: meta["measured"] for name, (_, meta) in paired_samples.items()}
    reference = measured["reference"]
    assert max(measured["font_small"]["glyph_ink_heights_px"]) < min(reference["glyph_ink_heights_px"])
    assert min(measured["font_large"]["glyph_ink_heights_px"]) > max(reference["glyph_ink_heights_px"])
    for name, size in (("font_small", 14), ("reference", 24), ("font_large", 32)):
        assert set(measured[name]["glyph_declared_font_sizes_px"]) == {size}
        assert measured[name]["bond_anchor_distances_px"] == pytest.approx(reference["bond_anchor_distances_px"])
        assert measured[name]["stroke_widths_px"] == reference["stroke_widths_px"]
    assert max(measured["bond_short"]["bond_anchor_distances_px"]) < min(reference["bond_anchor_distances_px"])
    assert min(measured["bond_long"]["bond_anchor_distances_px"]) > max(reference["bond_anchor_distances_px"])
    for name in ("bond_short", "bond_long"):
        assert measured[name]["glyph_ink_heights_px"] == reference["glyph_ink_heights_px"]
        assert measured[name]["glyph_declared_font_sizes_px"] == reference["glyph_declared_font_sizes_px"]
        assert measured[name]["stroke_widths_px"] == reference["stroke_widths_px"]
    assert measured["line_thin"]["stroke_widths_px"] == [1.0]
    assert measured["line_thick"]["stroke_widths_px"] == [3.2]
    for name in ("line_thin", "line_thick", "serif", "rotated"):
        assert measured[name]["bond_anchor_distances_px"] == pytest.approx(reference["bond_anchor_distances_px"])
    for files, _ in paired_samples.values():
        assert files[".sdf"] == paired_samples["reference"][0][".sdf"]
        assert files[".graph.json"] == paired_samples["reference"][0][".graph.json"]


def test_repeated_rendering_is_byte_deterministic(paired_samples):
    for profile in (profiles()[0], next(profile for profile in profiles() if profile.profile_id == "serif")):
        assert render(_entry("CC=O"), profile) == paired_samples[profile.profile_id]


def test_sample_metadata_hashes_bind_every_other_artifact(paired_samples):
    for files, meta in paired_samples.values():
        assert json.loads(files[".meta.json"]) == meta
        assert set(meta["sha256"]) == set(files) - {".meta.json"}
        for suffix, expected in meta["sha256"].items():
            assert sha256(files[suffix]).hexdigest() == expected
        layers = json.loads(files[".layers.json"])
        records = json.loads(files[".text-records.json"])
        for document in (layers, records):
            assert document["source_svg_sha256"] == sha256(files[".svg"]).hexdigest()
            assert document["source_png_sha256"] == sha256(files[".png"]).hexdigest()
        assert layers["text_records_sha256"] == sha256(files[".text-records.json"]).hexdigest()
        assert meta["split"] == layers["split"] == "quarantine"
        assert meta["training_authorized"] is layers["training_authorized"] is False


def test_small_fresh_build_uses_all_nine_profiles_for_each_stage_and_rejects_overwrite(tmp_path):
    # Two molecules, not a material-size data build: one exclusive to each stage.
    entries = [_entry("CC"), _entry("CC(=O)C")]
    assert entries[0]["stages"] == ["atoms_bonds"]
    assert entries[1]["stages"] == ["functional_groups"]
    output = tmp_path / "fresh-pilot"
    summary = build(output, workers=1, entries=entries, selected_profiles=profiles(), gallery=False)
    assert summary["planned_images"] == summary["accepted_images"] == 18
    assert summary["rejected_images"] == 0
    assert summary["molecules_with_all_profiles"] == 2
    assert summary["images_by_stage_membership"] == {"atoms_bonds": 9, "functional_groups": 9}
    assert summary["status"] == "quarantine_preview_only"
    assert summary["training_authorized"] is False
    assert summary["external_corpora_read"] is False
    assert summary["external_overlap_status"] == "unchecked"
    assert summary["internal_split"] == "quarantine_only_no_train_val_test"
    samples = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    for entry in entries:
        assert {sample["profile_id"] for sample in samples if sample["molecule_id"] == entry["molecule_id"]} == {
            profile.profile_id for profile in profiles()
        }
    hashes = json.loads((output / "sample-sha256.json").read_text())
    assert set(hashes) == {str(path.relative_to(output)) for path in (output / "samples").iterdir()}
    for relative, expected in hashes.items():
        assert sha256((output / relative).read_bytes()).hexdigest() == expected
    for sample in samples:
        prefix = output / "samples" / sample["sample_key"]
        meta = json.loads(Path(str(prefix) + ".meta.json").read_text())
        assert meta["training_authorized"] is False
        assert meta["split"] == "quarantine"
        for suffix, expected in meta["sha256"].items():
            assert sha256(Path(str(prefix) + suffix).read_bytes()).hexdigest() == expected
    summary_before = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        build(output, workers=1, entries=entries, selected_profiles=profiles(), gallery=False)
    assert (output / "summary.json").read_bytes() == summary_before
    assert json.loads((output / "sample-sha256.json").read_text()) == hashes


@pytest.mark.parametrize("reason", sorted(build_module.QUALITY_REJECTIONS))
def test_expected_contract_rejection_is_recorded_without_fallback_or_fake_sample(tmp_path, monkeypatch, reason):
    def reject(_entry, _profile):
        raise ValueError(reason)

    monkeypatch.setattr(build_module, "render", reject)
    profile = profiles()[0]
    output = tmp_path / "rejected-pilot"
    summary = build(output, workers=1, entries=[_entry("CC")], selected_profiles=[profile], gallery=False)
    assert summary["planned_images"] == summary["rejected_images"] == 1
    assert summary["accepted_images"] == 0
    assert summary["training_authorized"] is False
    assert not list((output / "samples").iterdir())
    assert (output / "samples.jsonl").read_bytes() == b""
    rejected = json.loads((output / "rejected.jsonl").read_text())
    assert rejected["profile_id"] == profile.profile_id
    assert rejected["reason"] == reason
    assert summary["molecules_with_no_images"] == [_entry("CC")["molecule_id"]]


@pytest.mark.parametrize("reason", [
    "SDF roundtrip changed chemical identity",
    "Attributed text/bond/auxiliary ink does not exactly cover source foreground",
    "Unexpected implementation error",
])
def test_unknown_or_invariant_errors_abort_without_completion_summary(tmp_path, monkeypatch, reason):
    def fail(_entry, _profile):
        raise ValueError(reason)

    monkeypatch.setattr(build_module, "render", fail)
    output = tmp_path / "incomplete-pilot"
    with pytest.raises(ValueError, match=reason):
        build(output, workers=1, entries=[_entry("CC")], selected_profiles=[profiles()[0]], gallery=False)
    assert not (output / "summary.json").exists()
    assert not (output / "rejected.jsonl").exists()
    assert not list((output / "samples").iterdir())


@pytest.mark.parametrize("field", ["molecule_id", "profile_id"])
def test_path_like_sample_identifiers_are_hard_errors(field):
    entry, profile = _entry("CC"), profiles()[0]
    if field == "molecule_id":
        entry["molecule_id"] = "../outside"
    else:
        profile = replace(profile, profile_id="../outside")
    with pytest.raises(ValueError, match="Unsafe curriculum sample identifier"):
        render(entry, profile)


@pytest.mark.parametrize("case", ["empty_entries", "empty_profiles", "duplicate_entries", "duplicate_profiles", "bad_workers"])
def test_invalid_build_requests_fail_before_creating_output(tmp_path, case):
    entries, selected, workers = [_entry("CC")], [profiles()[0]], 1
    if case == "empty_entries":
        entries = []
    elif case == "empty_profiles":
        selected = []
    elif case == "duplicate_entries":
        entries *= 2
    elif case == "duplicate_profiles":
        selected *= 2
    elif case == "bad_workers":
        workers = 0
    output = tmp_path / case
    with pytest.raises(ValueError):
        build(output, workers=workers, entries=entries, selected_profiles=selected, gallery=False)
    assert not output.exists()
