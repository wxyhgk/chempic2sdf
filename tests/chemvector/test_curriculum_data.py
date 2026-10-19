"""Synthetic CPU checks; no checkpoints, evaluation corpora or GPU access."""

from copy import deepcopy
from hashlib import sha256
import json
import shutil

import pytest
import torch

from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile, render
from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.bond_object import BOND_TYPE_TO_ID, DIRECTION_TO_ID
from chemvector.direct_atom_pair_bond import direct_atom_pair_loss, enumerate_unordered_atom_pairs
from chemvector.spatial_atom_object import SpatialAtomObjectConfig, SpatialAtomObjectDetector, spatial_atom_object_loss
from chemvector.train_atom_bond_scene import TARGET_FIELDS
from chemvector.train_direct_atom_pair_bond import build_pair_targets
from chemvector.training.curriculum.data import CurriculumCollator, load_sample, validate_records
from chemvector.training.curriculum.preflight import run


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("own-curriculum")
    (root / "samples").mkdir()
    rows, hashes, keys = [], {}, {}
    for smiles in ("CC", "CC=O", "CCO", "c1ccccc1", "C[O-]", "F[C@H](Cl)Br"):
        entry = merged_records([dict(record_id="test", name="test", stage="atoms_bonds", concepts=[], smiles=smiles)])[0]
        files, meta = render(entry, Profile())
        key = meta["sample_key"]
        keys[smiles] = key
        for suffix, data in files.items():
            relative = "samples/" + key + suffix
            (root / relative).write_bytes(data)
            hashes[relative] = sha256(data).hexdigest()
        rows.append({k: meta[k] for k in ("sample_key", "molecule_id", "profile_id")})
    (root / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (root / "sample-sha256.json").write_text(json.dumps(hashes))
    (root / "summary.json").write_text(json.dumps(dict(accepted_images=len(rows), planned_images=len(rows), rejected_images=0, external_overlap_status="unchecked")))
    return root, keys


def sample(corpus, smiles="CC=O"):
    root, keys = corpus
    return load_sample(root, keys[smiles], purpose="diagnostic")


def collate(*samples):
    return CurriculumCollator(purpose="diagnostic")(list(samples))


def test_targets_match_existing_names_and_retain_chemical_graph(corpus):
    s = sample(corpus)
    before = deepcopy(s["graph"])
    b = collate(s)
    assert set(TARGET_FIELDS) <= b.keys()
    assert b["pixel_values"].shape == (1, 3, 768, 768)
    assert b["pixel_values"].device.type == "cpu"
    assert b["pixel_values"].min() >= -1 and b["pixel_values"].max() <= 1
    assert b["atom_mask"].sum() == 3 and b["bond_mask"].sum() == 2
    assert b["chemical_graphs"][0] == before == s["graph"]
    assert not b["training_authorized"]
    assert "semantic" not in b and "ocr_targets" not in b
    assert b["spatial_cell_collision_pairs"] == [0]


def test_visible_text_is_not_a_formula_and_bbox_center_is_not_anchor(corpus):
    hydrocarbon, alcohol = sample(corpus, "CC"), sample(corpus, "CCO")
    b = collate(hydrocarbon, alcohol)
    assert b["visible_text"][0] == ["", ""]
    assert b["atom_text_mask"][0].sum() == 0
    assert b["atom_roles"][0, :2].tolist() == [0, 0]
    assert b["visible_text"][1] == [a["text"] for a in alcohol["layers"]["atoms"]]
    text = b["atom_text_mask"][1]
    assert text.sum() == 1
    assert not torch.equal(b["atom_text_center"][1, text], b["atom_anchors"][1, text])
    assert b["chemical_graphs"][0]["atoms"][0]["implicit_h_count"] == 3


def test_aromatic_chemistry_remains_separate_from_kekule_drawing(corpus):
    b = collate(sample(corpus, "c1ccccc1"))
    mask = b["bond_mask"]
    assert set(b["bond_chemical"][mask].tolist()) == {BOND_TYPE_TO_ID["AROMATIC"]}
    assert set(b["bond_display"][mask].tolist()) == {BOND_TYPE_TO_ID["SINGLE"], BOND_TYPE_TO_ID["DOUBLE"]}


def test_charge_and_stereo_are_preserved_and_wedge_endpoints_follow_drawing(corpus):
    charged, stereo = sample(corpus, "C[O-]"), sample(corpus, "F[C@H](Cl)Br")
    b = collate(charged, stereo)
    assert any(a["formal_charge"] == -1 for a in b["chemical_graphs"][0]["atoms"])
    assert any(a["chiral_tag"] != "CHI_UNSPECIFIED" for a in b["chemical_graphs"][1]["atoms"])
    wedge_found = False
    for index, bond in enumerate(stereo["layers"]["bonds"]):
        drawing = bond["drawing"]
        expected = [b["atom_ids"][1].index(drawing[k]) for k in ("begin_atom", "end_atom")]
        assert b["bond_atom_refs"][1, index].tolist() == expected
        assert b["bond_direction"][1, index] == DIRECTION_TO_ID[drawing["direction"]]
        wedge_found |= drawing["direction"] in {"BEGINWEDGE", "BEGINDASH"}
    assert wedge_found


def test_same_coordinates_do_not_merge_semantic_atoms(corpus):
    s = sample(corpus, "CC")
    s["layers"]["atoms"][1]["center"] = s["layers"]["atoms"][0]["center"][:]
    for bond in s["layers"]["bonds"]:
        bond["end"] = bond["begin"][:]
    b = collate(s)
    assert b["atom_mask"].sum() == 2 and b["bond_mask"].sum() == 1
    assert b["bond_atom_refs"][0, 0].tolist() == [0, 1]
    assert b["spatial_cell_collision_pairs"] == [1]


def test_array_order_is_not_used_as_semantic_identity(corpus):
    s = sample(corpus)
    original = collate(s)
    for owner in (s["graph"], s["layers"]):
        owner["atoms"].reverse(); owner["bonds"].reverse()
    changed = collate(s)
    for field in TARGET_FIELDS:
        assert torch.equal(original[field], changed[field])


@pytest.mark.parametrize("mutation", ["text", "chemistry", "edge", "bbox", "direction", "foreign_glyph", "nan", "duplicate_id"])
def test_invalid_labels_fail_closed(corpus, mutation):
    s = sample(corpus)
    if mutation == "text": s["layers"]["atoms"][-1]["text"] = "CH2"
    elif mutation == "chemistry": s["layers"]["atoms"][0]["formal_charge"] = 1
    elif mutation == "edge": s["layers"]["bonds"][0]["drawing"]["end_atom"] = 99
    elif mutation == "bbox": s["layers"]["atoms"][-1]["bbox"][0] += 1
    elif mutation == "direction": s["layers"]["bonds"][0]["drawing"]["direction"] = "UNKNOWN"
    elif mutation == "foreign_glyph": s["text_records"]["glyphs"][0]["atom_id"] = 99
    elif mutation == "nan": s["layers"]["atoms"][0]["center"][0] = float("nan")
    elif mutation == "duplicate_id": s["graph"]["atoms"][1]["atom_id"] = 0
    with pytest.raises(ValueError): collate(s)


def test_quarantine_is_not_a_training_switch(corpus):
    root, keys = corpus
    with pytest.raises(ValueError, match="not admitted"):
        load_sample(root, keys["CC"])
    s = sample(corpus)
    with pytest.raises(ValueError, match="Diagnostic"):
        CurriculumCollator()([s])
    s["purpose"] = "training"
    with pytest.raises(ValueError, match="admission"):
        CurriculumCollator()([s])


@pytest.mark.parametrize("key", ["../sample", "a/b", "", "/tmp/sample"])
def test_unsafe_paths_fail_before_reads(tmp_path, key):
    with pytest.raises(ValueError, match="Unsafe"):
        load_sample(tmp_path, key, purpose="diagnostic")


def test_capacity_is_rejected_not_silently_truncated(corpus):
    with pytest.raises(ValueError, match="capacity"):
        CurriculumCollator(AtomBondSceneConfig(atom_queries=2), purpose="diagnostic")([sample(corpus)])


def test_existing_atom_and_pair_losses_accept_adapter_targets_on_cpu(corpus):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        b = collate(sample(corpus))
        head = SpatialAtomObjectDetector(SpatialAtomObjectConfig(input_channels=4, hidden_channels=8))
        features = torch.zeros(1, 4, 48, 48)
        losses = spatial_atom_object_loss(head(features), b)
        assert torch.isfinite(losses["loss"])
        losses["loss"].backward()
        assert head.heatmap.weight.grad is not None
        pairs, rows = enumerate_unordered_atom_pairs(b["atom_mask"])
        mapping = torch.full_like(b["atom_roles"], -1)
        mapping[0, :3] = torch.arange(3)
        targets = build_pair_targets(pairs, rows, mapping, b, {"atom_mask": b["atom_mask"]})
        assert targets["presence"].sum() == 2
        output = {"presence_logits": torch.zeros(len(pairs), requires_grad=True),
                  "chemical_logits": torch.zeros(len(pairs), 4, requires_grad=True)}
        pair_loss = direct_atom_pair_loss(output, targets["presence"], targets["chemical"])["loss"]
        assert torch.isfinite(pair_loss)
        pair_loss.backward()
        assert output["chemical_logits"].grad is not None
    finally:
        torch.set_num_threads(previous_threads)


def test_preflight_runs_without_cuda_and_refuses_overwrite(corpus, tmp_path, monkeypatch):
    def no_cuda(*args, **kwargs): raise AssertionError("GPU access is forbidden in data preflight")
    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    root, keys = corpus
    report = run(root, tmp_path / "audit")
    assert report["target_compatible_images"] == len(keys)
    assert report["failed_images"] == 0
    assert not report["gpu_used"] and not report["training_authorized"]
    assert report["route"] == "self_developed_chemvector"
    assert not any("render rejections" in blocker for blocker in report["blockers_before_training"])
    assert len(json.loads((tmp_path / "audit/samples.json").read_text())) == len(keys)
    with pytest.raises(FileExistsError): run(root, tmp_path / "audit")


def test_preflight_retains_actual_renderer_rejection_gap(corpus, tmp_path):
    root, _ = corpus
    copied = tmp_path / "corpus"
    shutil.copytree(root, copied)
    summary = json.loads((copied / "summary.json").read_text())
    summary.update(rejected_images=1, planned_images=summary["accepted_images"] + 1)
    (copied / "summary.json").write_text(json.dumps(summary))
    report = run(copied, tmp_path / "audit")
    assert report["renderer_rejected_images"] == 1
    assert any("render rejections" in blocker for blocker in report["blockers_before_training"])
    assert not report["training_authorized"]


def test_tampered_image_fails_hash_check_and_keeps_preflight_denominator(corpus, tmp_path):
    root, keys = corpus
    copied = tmp_path / "corpus"
    shutil.copytree(root, copied)
    (copied / "samples" / (keys["CC"] + ".png")).write_bytes(b"corrupted image")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_sample(copied, keys["CC"], purpose="diagnostic")
    report = run(copied, tmp_path / "audit")
    assert report["checked_images"] == len(keys)
    assert report["failed_images"] == 1
    assert report["target_compatible_images"] == len(keys) - 1
    assert not report["training_authorized"]


def test_loader_refuses_symlinked_sample_directory(corpus, tmp_path):
    root, keys = corpus
    (tmp_path / "samples").symlink_to(root / "samples", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinked"):
        load_sample(tmp_path, keys["CC"], purpose="diagnostic")
