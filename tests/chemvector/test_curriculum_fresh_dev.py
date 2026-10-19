"""Temporary legacy fixtures only; no real eval or sealed payload access."""

from copy import deepcopy
import csv
from hashlib import sha256
import io
from itertools import islice
import json
from pathlib import Path
import tarfile

from PIL import Image
import pytest
from rdkit import Chem

from chemvector.training.curriculum.admission import identities
from chemvector.training.curriculum.fresh_dev import FORMAT, FreshDevSource, prepare_plan
from chemvector.training.curriculum.release import MIXED_FORMAT, RecordLoader, training_order, verify_release
from tests.chemvector.test_curriculum_release import bind, released
from tests.chemvector.test_curriculum_continuation import tiny


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n").encode()


def legacy(tmp_path, *, smiles=("CCl",), mutate=None, mode="RGB", size=(768, 768)):
    root = tmp_path / "legacy"
    (root / "shards").mkdir(parents=True)
    pixels = io.BytesIO()
    Image.new(mode, size, "white").save(pixels, format="PNG")
    rows, payloads = [], {}
    for i, smi in enumerate(smiles):
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        ids = identities(smi)
        for variant in ("original", "rdkit", "coordgen", "stress"):
            key = f"fresh_dev_{i:06d}_{variant}"
            sdf = (Chem.MolToMolBlock(Chem.MolFromSmiles(smi)) + "$$$$\n").encode()
            payloads[key + ".png"] = pixels.getvalue()
            payloads[key + ".sdf"] = sdf
            rows.append(dict(sample_key=key, molecule_id=f"mol-{i}", inchikey=ids["inchikey"],
                scaffold=ids["legacy_scaffold"], split="fresh_dev", variant=variant,
                output_shard="shards/fresh_dev-000000.tar", meta_sha256="a" * 64,
                png_sha256=sha256(pixels.getvalue()).hexdigest(), sdf_sha256=sha256(sdf).hexdigest()))
    if mutate:
        mutate(rows, payloads)
    shard = root / "shards/fresh_dev-000000.tar"
    with tarfile.open(shard, "w") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            if isinstance(payload, tarfile.TarInfo):
                archive.addfile(payload)
            else:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    with (root / "samples.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def file_record(path):
        data = (root / path).read_bytes()
        return {"path": path, "bytes": len(data), "sha256": sha256(data).hexdigest()}
    commitment = {"schema_version": "chemvector-joint-commitment-1.0", "dataset": "fresh_eval_12k",
        "sealed_access": "locked_until_checkpoint_gate", "samples": len(smiles) * 4 + 4,
        "molecules": len(smiles) + 1, "split_counts": {"fresh_dev": len(smiles) * 4, "sealed_test": 4},
        "samples_csv": [file_record("samples.csv"),
            {"path": "sealed_test/samples.csv", "bytes": 100, "sha256": "b" * 64}],
        "tars": [file_record("shards/fresh_dev-000000.tar"),
            {"path": "sealed_test/sealed_test-000000.tar", "bytes": 100, "sha256": "b" * 64}],
        "renderer_style_schema": {"sha256": "c" * 64},
        **{k: "d" * 64 for k in ("molecule_id_set_sha256", "inchikey_set_sha256", "scaffold_set_sha256",
                                 "font_family_groups_sha256")}}
    commitment["commitment_sha256"] = sha256(encoded(commitment)).hexdigest()
    (root / "commitment.json").write_bytes(encoded(commitment))
    (root / "validation.json").write_bytes(encoded({"dataset": "fresh_eval_12k", "status": "passed", "errors": []}))
    # Sealed files intentionally do not exist. A correct loader never needs them.
    plan = prepare_plan(root, molecules=len(smiles), seed=20260914)
    plan_path = tmp_path / "access-plan.json"
    plan_path.write_bytes(encoded(plan))
    binding = {"format": FORMAT, "access_plan": str(plan_path), "access_plan_sha256": sha256(encoded(plan)).hexdigest()}
    return root, binding


@pytest.fixture
def original(tmp_path):
    return legacy(tmp_path)


def test_plan_only_reads_aggregate_files(original, monkeypatch):
    root, _ = original
    read = Path.read_bytes
    paths = []
    def aggregate_only(path):
        assert path.name in {"commitment.json", "validation.json"}
        paths.append(path.name)
        return read(path)
    monkeypatch.setattr(Path, "read_bytes", aggregate_only)
    plan = prepare_plan(root, molecules=1, seed=12)
    assert paths == ["commitment.json", "validation.json"]
    assert not plan["training_authorized"] and not plan["sealed_access"]


@pytest.mark.parametrize("smiles", ["CCl", "F[C@H](Cl)Br", "[13CH3]CO", "C[NH3+]", "[Na+].[Cl-]"])
def test_original_pixels_and_identity_survive_without_training_annotations(tmp_path, smiles):
    root, binding = legacy(tmp_path, smiles=(smiles,))
    source = FreshDevSource(root, binding)
    rows = source.candidate_rows("fresh")
    assert len(rows) == 4 and len(source._indexes) == 1
    expected = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    for row in rows:
        sample = source.load(row)
        assert set(sample) == {"image", "target_molblock", "purpose"}
        assert sample["purpose"] == "diagnostic" and row["canonical_smiles"] == expected
        assert sample["image"].tobytes() == Image.new("RGB", (768, 768), "white").tobytes()


def test_fixed_selection_is_deterministic_and_keeps_four_variants(tmp_path):
    root, binding = legacy(tmp_path, smiles=("CCl", "CCF", "CCBr"))
    plan = prepare_plan(root, molecules=2, seed=123)
    Path(binding["access_plan"]).write_bytes(encoded(plan))
    binding["access_plan_sha256"] = sha256(encoded(plan)).hexdigest()
    first, second = FreshDevSource(root, binding), FreshDevSource(root, binding)
    assert first.rows == second.rows and len(first.rows) == 8
    assert len({r["molecule_id"] for r in first.rows.values()}) == 2


@pytest.mark.parametrize("file", ["access-plan.json", "legacy/commitment.json", "legacy/validation.json", "legacy/samples.csv"])
def test_source_pins_plan_aggregates_and_manifest(original, tmp_path, file):
    root, binding = original
    path = tmp_path / file
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        FreshDevSource(root, binding)


def test_changed_tar_rejected_before_and_after_index(original):
    root, binding = original
    source = FreshDevSource(root, binding)
    rows = source.candidate_rows("fresh")
    path = root / "shards/fresh_dev-000000.tar"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="changed"):
        source.load(rows[0])
    with pytest.raises(ValueError, match="shard hash/size"):
        FreshDevSource(root, binding).candidate_rows("fresh")


def test_duplicate_tar_members_rejected_even_with_a_matching_commitment(original):
    root, binding = original
    path = root / "shards/fresh_dev-000000.tar"
    with tarfile.open(path, "a") as archive:
        info = tarfile.TarInfo("fresh_dev_000000_original.png")
        info.size = 4
        archive.addfile(info, io.BytesIO(b"fake"))
    commitment_path = root / "commitment.json"
    commitment = json.loads(commitment_path.read_bytes())
    commitment["tars"][0].update(bytes=path.stat().st_size, sha256=sha256(path.read_bytes()).hexdigest())
    commitment.pop("commitment_sha256")
    commitment["commitment_sha256"] = sha256(encoded(commitment)).hexdigest()
    commitment_path.write_bytes(encoded(commitment))
    plan = prepare_plan(root, molecules=1, seed=20260914)
    Path(binding["access_plan"]).write_bytes(encoded(plan))
    binding["access_plan_sha256"] = sha256(encoded(plan)).hexdigest()
    with pytest.raises(ValueError, match="duplicate"):
        FreshDevSource(root, binding).candidate_rows("fresh")


@pytest.mark.parametrize("mutation", ["split", "shard", "variant", "molecule", "key"])
def test_bad_manifest_is_rejected_before_any_tar_access(tmp_path, monkeypatch, mutation):
    def mutate(rows, payloads):
        if mutation == "split":
            rows[0]["split"] = "sealed_test"
        elif mutation == "shard":
            rows[0]["output_shard"] = "sealed_test/sealed_test-000000.tar"
        elif mutation == "variant":
            rows[0]["variant"] = "rdkit"
        elif mutation == "molecule":
            rows[0]["molecule_id"] = "different"
        else:
            rows[0]["sample_key"] = rows[1]["sample_key"]
    root, binding = legacy(tmp_path, mutate=mutate)
    def forbidden(*args, **kwargs):
        raise AssertionError("Manifest failure must precede tar access")
    monkeypatch.setattr(tarfile, "open", forbidden)
    with pytest.raises(ValueError):
        FreshDevSource(root, binding)


@pytest.mark.parametrize("field,value", [("split", "train"), ("domain", "curriculum"),
    ("molecule_id", "other"), ("sample_key", "sealed_test_000000_original"), ("png_sha256", "b" * 64)])
def test_wrong_row_cannot_open_tar(original, monkeypatch, field, value):
    root, binding = original
    source = FreshDevSource(root, binding)
    row = dict(next(iter(source.rows.values())), source="fresh", split="validation", domain="full_molecule")
    row[field] = value
    def forbidden(*args):
        raise AssertionError("Invalid row cannot open tar")
    monkeypatch.setattr(source, "_read_payloads", forbidden)
    with pytest.raises(ValueError):
        source.load(row)


@pytest.mark.parametrize("bad", ["../escape.png", "/absolute.png", "sealed_test_000000_original.png", "link", "missing", "hash", "multi_sdf", "wrong_identity"])
def test_invalid_tar_or_truth_is_rejected(tmp_path, bad):
    def mutate(rows, payloads):
        if bad in {"../escape.png", "/absolute.png", "sealed_test_000000_original.png"}:
            payloads[bad] = b"bad"
        elif bad == "link":
            name = rows[0]["sample_key"] + ".png"
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
            payloads[name] = info
        elif bad == "missing":
            payloads.pop(rows[0]["sample_key"] + ".png")
        elif bad == "hash":
            payloads[rows[0]["sample_key"] + ".png"] = b"changed"
        else:
            sdf = Chem.MolToMolBlock(Chem.MolFromSmiles("CCN")) + "$$$$\n"
            if bad == "multi_sdf":
                sdf += sdf
            for row in rows:
                payloads[row["sample_key"] + ".sdf"] = sdf.encode()
                row["sdf_sha256"] = sha256(sdf.encode()).hexdigest()
    root, binding = legacy(tmp_path, mutate=mutate)
    with pytest.raises(ValueError):
        FreshDevSource(root, binding).candidate_rows("fresh")


@pytest.mark.parametrize("mode,size", [("L", (768, 768)), ("RGBA", (768, 768)), ("RGB", (100, 100))])
def test_original_image_is_never_silently_converted(tmp_path, mode, size):
    root, binding = legacy(tmp_path, mode=mode, size=size)
    with pytest.raises(ValueError, match="shape/mode"):
        FreshDevSource(root, binding).candidate_rows("fresh")


@pytest.mark.parametrize("target", ["root", "shards", "tar", "manifest"])
def test_symlinked_source_components_are_rejected(original, tmp_path, target):
    root, binding = original
    if target == "root":
        link = tmp_path / "root-link"
        link.symlink_to(root, target_is_directory=True)
        root = link
    else:
        path = root / {"shards": "shards", "tar": "shards/fresh_dev-000000.tar", "manifest": "samples.csv"}[target]
        moved = tmp_path / (target + "-moved")
        path.rename(moved)
        path.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(ValueError, match="symlinked"):
        FreshDevSource(root, binding).candidate_rows("fresh")


@pytest.fixture
def mixed(released, original):
    manifest, admission, mp, ap = released
    root, binding = original
    source = FreshDevSource(root, binding)
    manifest["target_format"] = MIXED_FORMAT
    manifest["sources"]["fresh"] = str(root)
    manifest["validation_sources"] = {"fresh": binding}
    manifest["samples"] = [r for r in manifest["samples"] if r["split"] == "train"] + source.candidate_rows("fresh")
    admission["checked_images"] = len(manifest["samples"])
    bind(manifest, admission, mp, ap)
    return manifest, admission, mp, ap


def test_mixed_release_dispatches_and_never_samples_dev(mixed):
    m, a, mp, ap = mixed
    checked, _, _ = verify_release(mp, ap)
    loader = RecordLoader(checked)
    for row in checked["samples"]:
        sample = loader(checked, row)
        assert ("target_molblock" in sample) == (row["split"] == "validation")
        assert sample["purpose"] == ("diagnostic" if row["split"] == "validation" else "training")
    assert all(r["source"] != "fresh" for r in islice(training_order(checked), 100))


@pytest.mark.parametrize("mutation", ["train", "omit", "format", "group", "row", "binding"])
def test_mixed_release_cannot_repurpose_or_cherry_pick_validation(mixed, mutation):
    m, a, mp, ap = mixed
    if mutation == "train":
        for row in m["samples"][2:]:
            row["split"] = "train"
    elif mutation == "omit":
        m["samples"].pop()
        a["checked_images"] -= 1
    elif mutation == "format":
        m["target_format"] = "curriculum-same-svg"
    elif mutation == "group":
        for row in m["samples"][2:]:
            row.update(canonical_smiles="CCO", group_key=identities("CCO")["group_key"])
    elif mutation == "row":
        m["samples"][-1]["sdf_sha256"] = "b" * 64
    else:
        m["validation_sources"]["fresh"]["access_plan_sha256"] = "b" * 64
    bind(m, a, mp, ap)
    with pytest.raises(ValueError):
        verify_release(mp, ap)


def test_unlisted_row_is_rejected(mixed):
    m, _, _, _ = mixed
    loader = RecordLoader(m)
    row = deepcopy(m["samples"][-1])
    row["split"] = "train"
    with pytest.raises(ValueError, match="fixed release"):
        loader(m, row)


def test_mixed_cpu_job_uses_original_validation_only_for_scoring(mixed, tiny, tmp_path, monkeypatch):
    import torch
    from chemvector.training.curriculum.data import CurriculumCollator
    from chemvector.training.curriculum.launcher import run_training
    from tests.chemvector.test_curriculum_bundle import configure

    model, config, _ = tiny
    configure(model)
    _, _, mp, ap = mixed
    output = tmp_path / "mixed-run"
    output.mkdir()
    purposes, predictions = [], []
    old_collate = CurriculumCollator.__call__
    def collate(self, samples):
        purposes.extend(s["purpose"] for s in samples)
        assert all("target_molblock" not in s for s in samples)
        return old_collate(self, samples)
    def predict(model, config, pixels, image):
        assert not model.training and not torch.is_grad_enabled()
        assert pixels.shape == (1, 3, 768, 768)
        predictions.append(True)
        return {"molblock": None, "fallback": False, "error": None}
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU-only synthetic integration")
    monkeypatch.setattr(CurriculumCollator, "__call__", collate)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    provenance = {"checkpoints": {}, "atom_threshold": -1e6, "bond_threshold": -.5}
    report = run_training(mp, ap, output, max_steps=1, validate_every=1, patience=1,
        component_loader=lambda: (model, config, provenance), predictor_factory=lambda *args: predict)
    assert purposes == ["training"] and len(predictions) == 8
    assert report["baseline"]["images"] == 4 and report["optimizer_steps"] == 1
    assert not report["gpu_used"] and not report["deployed"]
    assert report["selected"]["path"].endswith("initial.pt")
