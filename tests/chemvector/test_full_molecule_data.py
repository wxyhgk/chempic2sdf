"""Synthetic source fixtures only: no real train/eval files or GPU."""

from copy import deepcopy
import csv
from hashlib import sha256
import io
import json
import tarfile

import pytest
from rdkit import Chem

from Data.sdf.chemvector_curriculum_v0_1.build import build
from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.full_molecule_data import (
    _legacy_json, _set_hash, file_hash, prepare, read_train_source,
    select_molecules, verify_selected_assets,
)


def molecule(smiles, index, source="chembl"):
    mol = Chem.MolFromSmiles(smiles)
    return dict(molecule_id=f"{source}:example{index}", source=source, source_id=f"example{index}",
        inchikey=Chem.MolToInchiKey(mol), scaffold="", canonical_isomeric_smiles=Chem.MolToSmiles(mol),
        split="train", heavy_atoms=str(mol.GetNumHeavyAtoms()), image_count="4")


def exclusion(*smiles):
    return {"schema_version": "chemvector-joint-overlap-index-0.1", "audit": {"unresolved": 11},
            "benchmark_inchikeys": [Chem.MolToInchiKey(Chem.MolFromSmiles(s)) for s in (smiles or ("CC",))] + [""]}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def recommit(root, commitment):
    commitment = {k: v for k, v in commitment.items() if k != "commitment_sha256"}
    commitment["commitment_sha256"] = sha256(_legacy_json(commitment)).hexdigest()
    (root / "commitment.json").write_bytes(_legacy_json(commitment))
    return commitment


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "style_train"
    root.mkdir(); (root / "shards").mkdir()
    molecules = [molecule("CCCCCCCC", 0), molecule("CCCCCCCO", 1, "coconut")]
    samples = []
    shard = root / "shards/train-000000.tar"
    with tarfile.open(shard, "w") as archive:
        for index, mol in enumerate(molecules):
            graph = _legacy_json({k: mol[k] for k in ("molecule_id", "inchikey", "canonical_isomeric_smiles")})
            sdf = Chem.MolToMolBlock(Chem.MolFromSmiles(mol["canonical_isomeric_smiles"])).encode()
            for variant in ("original", "rdkit", "coordgen", "stress"):
                key = f"train_{index:06d}_{variant}"
                for suffix, data in ((".chemmol.json", graph), (".sdf", sdf)):
                    member = tarfile.TarInfo(key + suffix); member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
                samples.append({**{k: mol[k] for k in ("molecule_id", "inchikey", "scaffold", "source", "source_id", "split")},
                    "sample_key": key, "variant": variant, "output_shard": "shards/train-000000.tar",
                    "chemmol_sha256": sha256(graph).hexdigest(), "sdf_sha256": sha256(sdf).hexdigest()})
    write_csv(root / "molecules.csv", molecules); write_csv(root / "samples.csv", samples)
    commitment = dict(schema_version="chemvector-joint-commitment-1.0", dataset="style_train_100k",
        sealed_access="not_applicable", samples=8, molecules=2, split_counts={"train": 8},
        samples_csv=[dict(path="samples.csv", bytes=(root / "samples.csv").stat().st_size, sha256=file_hash(root / "samples.csv"))],
        tars=[dict(path="shards/train-000000.tar", bytes=shard.stat().st_size, sha256=file_hash(shard))],
        **{field + "_set_sha256": _set_hash(r[field] for r in molecules) for field in ("molecule_id", "inchikey", "scaffold")})
    recommit(root, commitment)
    (root / "validation.json").write_text(json.dumps(dict(dataset="style_train_100k", status="passed", errors=[])))
    index = tmp_path / "public-cache.json"; index.write_text(json.dumps(exclusion()))
    return root, index


def test_selection_is_order_independent_and_keeps_distinct_scaffolds():
    rows = [molecule("C" * n, n, src) for n, src in ((8, "chembl"), (9, "coconut"), (22, "chembl"), (42, "coconut"))]
    selected, report = select_molecules(rows, exclusion(), count=4, seed=8)
    assert (selected, report) == select_molecules(list(reversed(rows)), exclusion(), count=4, seed=8)
    assert len({r["group_key"] for r in selected}) == 4
    assert not report["training_authorized"] and report["coverage_gaps"]
    assert "old train" in report["validation_use"]


def test_public_connectivity_excludes_other_stereoisomer():
    rows = [molecule("CCCC[C@H](F)CCO", 0), molecule("CCCCCCCO", 1)]
    selected, report = select_molecules(rows, exclusion("CCCC[C@@H](F)CCO"), count=1, seed=1)
    assert selected[0]["molecule_id"] == rows[1]["molecule_id"]
    assert report["screening_skips"]["known_public_identity_or_connectivity"] == 1


def test_same_scaffold_is_not_padded_to_requested_size():
    rows = [molecule("CCc1ccccc1", 0), molecule("CCCc1ccccc1", 1)]
    with pytest.raises(ValueError, match="Not enough"):
        select_molecules(rows, exclusion(), count=2, seed=1)


def test_preparation_binds_real_train_payloads_and_does_not_read_eval(source, monkeypatch):
    root, index = source
    from pathlib import Path
    previous = Path.open
    def guarded(self, *args, **kwargs):
        assert "sealed" not in str(self) and "fresh" not in str(self)
        return previous(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    before = {p: file_hash(p) for p in root.rglob("*") if p.is_file()}
    entries, provenance = prepare(root, index, count=2)
    assert len(entries) == 2 and provenance["selected_source_images"] == 8
    assert not provenance["sealed_test_read"] and not provenance["fresh_eval_read"]
    assert not provenance["original_pixels_preserved"] and not provenance["training_authorized"]
    assert all(e["stages"] == ["full_molecule"] and e["split"] == "quarantine" for e in entries)
    assert {p: file_hash(p) for p in before} == before


@pytest.mark.parametrize("mutation", ["commitment", "molecule_id", "sample_bytes", "nontrain", "unsafe_path", "variant"])
def test_invalid_manifests_rejected(source, mutation):
    root, _ = source
    commitment = json.loads((root / "commitment.json").read_text())
    if mutation == "commitment":
        commitment["seed"] = 7
        (root / "commitment.json").write_text(json.dumps(commitment))
    elif mutation == "molecule_id":
        text = (root / "molecules.csv").read_text().replace("example0", "changed0")
        (root / "molecules.csv").write_text(text)
    elif mutation == "sample_bytes":
        with (root / "samples.csv").open("a") as stream: stream.write("\n")
    else:
        rows = list(csv.DictReader(io.StringIO((root / "samples.csv").read_text())))
        if mutation == "nontrain": rows[0]["split"] = "sealed_test"
        if mutation == "unsafe_path": rows[0]["output_shard"] = "../sealed_test/data.tar"
        if mutation == "variant": rows[0]["variant"] = rows[1]["variant"]
        write_csv(root / "samples.csv", rows)
        commitment["samples_csv"][0].update(bytes=(root / "samples.csv").stat().st_size, sha256=file_hash(root / "samples.csv"))
        recommit(root, commitment)
    with pytest.raises(ValueError): read_train_source(root)


def test_tar_corruption_rejected_before_any_render(source):
    root, index = source
    with (root / "shards/train-000000.tar").open("ab") as stream: stream.write(b"bad")
    with pytest.raises(ValueError, match="shard mismatch"): prepare(root, index, count=1)


def test_nontrain_commitment_rejected_before_molecule_manifest(source, monkeypatch):
    from pathlib import Path
    root, _ = source
    commitment = json.loads((root / "commitment.json").read_text())
    commitment["dataset"] = "fresh_eval_12k"
    recommit(root, commitment)
    previous = Path.open
    def guarded(self, *args, **kwargs):
        assert self.name not in {"molecules.csv", "samples.csv"}
        return previous(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="train-only"):
        read_train_source(root)


def test_symlinked_train_shard_is_rejected(source, tmp_path):
    root, index = source
    shard = root / "shards/train-000000.tar"
    moved = tmp_path / "moved.tar"
    shard.rename(moved); shard.symlink_to(moved)
    with pytest.raises(ValueError, match="shard mismatch"):
        prepare(root, index, count=1)


def test_manifest_smiles_substitution_detected_even_when_identity_set_unchanged(source):
    root, index = source
    molecules, samples, commitment, _ = read_train_source(root)
    selected = deepcopy(molecules)
    selected[0]["canonical_isomeric_smiles"] = "CCCCCCCCO"
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_selected_assets(root, selected, samples, commitment)


def test_new_render_remains_quarantine_and_carries_provenance(source, tmp_path):
    root, index = source
    entries, provenance = prepare(root, index, count=1)
    out = tmp_path / "new-render"
    report = build(out, entries=entries, workers=1, selected_profiles=[Profile()],
                   typography="ink-aware-v1", gallery=False, source_provenance=provenance)
    assert report["accepted_images"] == 1 and report["external_corpora_read"]
    assert report["source_provenance_sha256"] == file_hash(out / "source-provenance.json")
    key = json.loads((out / "samples.jsonl").read_text())["sample_key"]
    sample = load_sample(out, key, purpose="diagnostic")
    assert sample["meta"]["source_provenance"]["source_split"] == "train"
    assert not sample["meta"]["source_provenance"]["original_pixels_preserved"]
    with pytest.raises(ValueError, match="not admitted"): load_sample(out, key)
    with pytest.raises(FileExistsError):
        build(out, entries=entries, workers=1, selected_profiles=[Profile()], source_provenance=provenance)


def test_subpixel_auxiliary_geometry_rejects_whole_image_not_ink():
    from Data.sdf.chemvector_curriculum_v0_1.build import _render_job
    from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
    from Data.sdf.chemvector_curriculum_v0_1.rendering import render
    # Reproduction from an explicitly train-sourced structure; no corpus reads.
    smiles = "CCCCCCCCCCCC[C@@H](O)[C@H]1CC[C@H]([C@H](O)CCC[C@@H]2OC(C)(C)O[C@@H]2CCCCC[C@@H](O)CC2=C[C@H](C)OC2=O)O1"
    entry = merged_records([dict(record_id="crowded_train_geometry", name="crowded geometry",
        stage="full_molecule", concepts=[], smiles=smiles)])[0]
    profile = Profile(profile_id="font_large", font_size=32)
    with pytest.raises(ValueError, match="Degenerate auxiliary junction"):
        render(entry, profile, typography="ink-aware-v1")
    files, meta, rejection = _render_job((entry, profile, "ink-aware-v1"))
    assert files is None and meta is None
    assert rejection["reason"] == "Degenerate auxiliary junction"


def test_chemical_ownership_failures_still_abort_builder(monkeypatch):
    import Data.sdf.chemvector_curriculum_v0_1.build as module
    def fail(*args, **kwargs):
        raise ValueError("Auxiliary junction has no unique incident-bond geometric witness")
    monkeypatch.setattr(module, "render", fail)
    with pytest.raises(ValueError, match="no unique"):
        module._render_job(({}, Profile(), "ink-aware-v1"))
