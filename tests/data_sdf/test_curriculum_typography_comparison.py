"""Small paired pilots; no historical data, checkpoints, or GPU."""

from copy import deepcopy
import json
import shutil
import xml.etree.ElementTree as ET

import pytest

from Data.sdf.chemvector_curriculum_v0_1.build import build
from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from Data.sdf.chemvector_curriculum_v0_1.compare_typography import compare, _verify_reference
from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile, render

BRIDGED = "O[C@@H]1C[C@]2(O)N[C@H]1C[C@@H](O)[C@@H]2O"


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired-typography")
    entries = merged_records([dict(record_id=str(i), name=str(i), stage="full_molecule", concepts=[], smiles=s)
                              for i, s in enumerate(("CCO", BRIDGED))])
    for name, mode in (("before", "ink-aware-v1"), ("after", "stereo-aware-v1")):
        build(root / name, workers=1, entries=entries, selected_profiles=[Profile()], gallery=False, typography=mode)
    return root, entries


def test_paired_coverage_and_original_geometry(paired):
    root, _ = paired
    r = compare(root / "before", root / "after")
    assert (r["planned_images"], r["before_accepted"], r["after_accepted"], r["recovered"], r["regressed"]) == (2, 1, 2, 1, 0)
    assert r["common_chemical_graph_and_sdf_unchanged"] == 1
    assert r["reference_checked_images"] == 2 and r["unchanged_protected_paths"] > 0
    assert r["verified_artifact_hashes"] == {"before": 11, "after": 22}
    assert not r["training_authorized"] and not r["gpu_used"]


def test_different_plan_rejected(paired, tmp_path):
    root, _ = paired
    copied = tmp_path / "after"
    shutil.copytree(root / "after", copied)
    p = copied / "profiles.json"
    records = json.loads(p.read_text()); records[0]["font_size"] = 23
    p.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="exact same"):
        compare(root / "before", copied)


def test_missing_denominator_rejected(paired, tmp_path):
    root, _ = paired
    copied = tmp_path / "before"
    shutil.copytree(root / "before", copied)
    (copied / "rejected.jsonl").write_text("")
    with pytest.raises(ValueError, match="partition"):
        compare(copied, root / "after")


def test_artifact_corruption_rejected(paired, tmp_path):
    root, _ = paired
    copied = tmp_path / "after"
    shutil.copytree(root / "after", copied)
    path = next((copied / "samples").glob("*.svg"))
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact hash"):
        compare(root / "before", copied)


def test_reference_detects_stereo_and_anchor_changes_and_restores_hook(paired):
    from Data.sdf.chemvector_curriculum_v0_1 import stereo_typography
    _, entries = paired
    entry = next(e for e in entries if e["canonical_smiles"] != "CCO")
    files, _ = render(entry, Profile(), typography="stereo-aware-v1")
    graph, layers = [json.loads(files[s]) for s in (".graph.json", ".layers.json")]
    hook = stereo_typography.adjust
    modified = ET.fromstring(files[".svg"])
    stereo = next(b for b in layers["bonds"] if b["drawing"]["direction"] == "BEGINDASH")
    modified[stereo["svg_node_indices"][0]].set("d", "M 1,1 L 2,2")
    with pytest.raises(ValueError, match="protected stereo"):
        _verify_reference(entry, Profile(), ET.tostring(modified), graph, layers)
    changed = deepcopy(layers); changed["atoms"][0]["center"][0] += 1
    with pytest.raises(ValueError, match="anchors"):
        _verify_reference(entry, Profile(), files[".svg"], graph, changed)
    assert stereo_typography.adjust is hook


def test_regressed_samples_are_reported_not_hidden(paired, tmp_path, monkeypatch):
    import Data.sdf.chemvector_curriculum_v0_1.build as builder
    root, entries = paired
    original = builder._render_job
    def drop_ethanol(job):
        entry, profile, *_ = job
        if entry["canonical_smiles"] == "CCO":
            return None, None, dict(molecule_id=entry["molecule_id"], profile_id=profile.profile_id,
                                    canonical_smiles="CCO", reason="synthetic regression")
        return original(job)
    monkeypatch.setattr(builder, "_render_job", drop_ethanol)
    after = tmp_path / "regressed"
    build(after, workers=1, entries=entries, selected_profiles=[Profile()], gallery=False, typography="stereo-aware-v1")
    r = compare(root / "before", after)
    assert r["recovered"] == 1 and r["regressed"] == 1
    assert len(r["regressed_keys"]) == 1 and r["common_chemical_graph_and_sdf_unchanged"] == 0
