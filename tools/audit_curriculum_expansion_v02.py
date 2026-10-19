"""Independent, CPU-only audit of a completed quarantined v0.2 dataset.

Run as ``python -m tools.audit_curriculum_expansion_v02 --dataset ... --output ...``.
The receipt checks artifacts, not the renderer's exit status, model reachability,
external identity clearance, recognition accuracy, or permission to train.
"""

from collections import Counter, defaultdict
from hashlib import sha256
import argparse
import io
import json
from pathlib import Path
import re
import time

from PIL import Image
from rdkit import Chem

from Data.sdf.text_source_renderer.renderer import chemistry_record
from chemvector.training.curriculum.data import validate_records


REPO = Path(__file__).resolve().parents[1]
KEY = re.compile(r"[A-Za-z0-9_-]+\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
SUFFIXES = (".auxiliary.png", ".bond.png", ".graph.json", ".layers.json", ".meta.json",
            ".png", ".sdf", ".semantic.png", ".svg", ".text-records.json", ".text.png")
MANIFESTS = ("plan.json", "summary.json", "molecules.json", "profiles.json", "samples.jsonl",
             "rejected.jsonl", "sample-sha256.json", "image-duplicates.json", "progress.jsonl",
             "visible-coverage.jsonl", "planned-coverage.json", "accepted-coverage.json")


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def digest(data):
    return sha256(data).hexdigest()


def read(path):
    require(not path.is_symlink() and path.is_file(), f"Missing or symlinked artifact: {path}")
    return path.read_bytes()


def decode(data):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def no_constant(value):
        raise AuditError(f"Nonfinite JSON value: {value}")
    return json.loads(data, object_pairs_hook=unique_pairs, parse_constant=no_constant)


def lines(data):
    return [decode(line) for line in data.splitlines() if line.strip()]


def source_path(name):
    """A dataset manifest cannot grant permission to read an arbitrary file."""
    path = Path(name)
    path = path if path.is_absolute() else REPO / path
    allowed_roots = [REPO / "Data/sdf" / name for name in
                     ("chemvector_curriculum_v0_1", "chemvector_curriculum_v0_2", "text_source_renderer")]
    allowed_files = {REPO / "chemvector/training/curriculum" / name for name in ("runtime.py", "admission.py")}
    allowed_files.update(Path("/usr/share/fonts/truetype/dejavu") / name
                         for name in ("DejaVuSans.ttf", "DejaVuSerif.ttf"))
    require(".." not in path.parts and (path in allowed_files or
            (path.suffix == ".py" and any(path.is_relative_to(root) for root in allowed_roots))),
            f"Source binding outside allowed code/font scope: {name}")
    require(path.resolve() == path.absolute(), f"Symlinked source binding: {name}")
    return path


def verify_sources(bindings):
    require(isinstance(bindings, dict) and bindings, "Missing frozen source bindings")
    for name, expected in bindings.items():
        require(isinstance(expected, str) and SHA.fullmatch(expected), f"Invalid source SHA: {name}")
        require(digest(read(source_path(name))) == expected, f"Frozen source SHA mismatch: {name}")


def validate_sdf(payload, graph, canonical):
    """Allow only representational endpoint/stereo differences in SDF storage."""
    text = payload.decode("utf-8")
    blocks = text.split("$$$$")
    require(len(blocks) == 2 and not blocks[1].strip(), "Expected exactly one terminated SDF molecule")
    mol = Chem.MolFromMolBlock(blocks[0], sanitize=True, removeHs=False, strictParsing=True)
    require(mol is not None, "Invalid SDF")
    require(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) == canonical,
            "SDF strict isomeric identity mismatch")
    expected = Chem.MolFromSmiles(canonical)
    require(expected is not None, "Invalid canonical SMILES")
    record = chemistry_record(expected)
    for row, bond in zip(record["bonds"], expected.GetBonds()):
        row["stereo_atoms"] = list(bond.GetStereoAtoms())
    require(record["atoms"] == graph["atoms"] and record["bonds"] == graph["bonds"],
            "Graph chemistry/IDs differ from canonical source")
    require(mol.GetNumAtoms() == len(graph["atoms"]) and mol.GetNumBonds() == len(graph["bonds"]),
            "SDF object count mismatch")
    for atom, row in zip(mol.GetAtoms(), graph["atoms"]):
        observed = (atom.GetIdx(), atom.GetSymbol(), atom.GetIsotope(), atom.GetFormalCharge(),
                    atom.GetIsAromatic(), atom.GetNumRadicalElectrons(), atom.GetTotalNumHs())
        wanted = (row["atom_id"], row["element"], row["isotope"], row["formal_charge"],
                  row["aromatic"], row["radical_electrons"], row["explicit_h_count"] + row["implicit_h_count"])
        require(observed == wanted, "SDF atom ID/chemistry mismatch")
        # SDF can choose a different equivalent explicit/implicit-H encoding.
        if expected.GetAtomWithIdx(atom.GetIdx()).HasProp("_CIPCode"):
            require(atom.HasProp("_CIPCode") and atom.GetProp("_CIPCode") ==
                    expected.GetAtomWithIdx(atom.GetIdx()).GetProp("_CIPCode"),
                    "SDF specified atom stereo mismatch")
    for bond, row in zip(mol.GetBonds(), graph["bonds"]):
        require((bond.GetIdx(), {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}, str(bond.GetBondType()),
                 bond.GetIsAromatic()) ==
                (row["bond_id"], {row["begin_atom"], row["end_atom"]}, row["chemical_type"], row["aromatic"]),
                "SDF bond ID/chemistry mismatch")
        stereo = str(bond.GetStereo())
        allowed = {"STEREONONE", "STEREOANY"} if row["stereo"] == "STEREONONE" else {row["stereo"]}
        require(stereo in allowed, "SDF specified bond stereo mismatch")
    return mol.GetNumAtoms(), mol.GetNumBonds()


def check_dataset(dataset, *, progress=print):
    root = Path(dataset)
    require(root.is_dir() and not root.is_symlink(), "Dataset must be a real directory")
    require(not (root / "samples").is_symlink(), "Symlinked samples directory")
    raw = {name: read(root / name) for name in MANIFESTS}
    manifest_sha = {name: digest(data) for name, data in raw.items()}
    plan, summary, molecules, profiles, hashes, duplicates = (
        decode(raw[name]) for name in ("plan.json", "summary.json", "molecules.json", "profiles.json",
                                      "sample-sha256.json", "image-duplicates.json"))
    require(plan.get("schema_version") == "chemvector-curriculum-v0.2-plan" and
            summary.get("schema_version") == "chemvector-curriculum-v0.2-result", "Wrong dataset schemas")
    require(summary.get("generation_complete") is True, "Dataset generation is incomplete")
    require(summary.get("status") == "quarantine_preview_only", "Dataset is not quarantined")
    for owner in (plan, summary):
        require(owner.get("training_authorized") is False and owner.get("external_overlap_status") == "unchecked"
                and owner.get("external_identity_queries") == 0 and owner.get("external_corpora_read") is False
                and owner.get("model_forwards") == 0 and owner.get("optimizer_steps") == 0,
                "Dataset scope/admission flags disagree")
    require(summary.get("gpu_used") is False, "Unexpected GPU claim")
    require(summary["plan_sha256"] == manifest_sha["plan.json"], "Summary plan SHA mismatch")
    verify_sources(plan["input_sha256"])
    require(molecules and profiles == plan["profiles"], "Plan/profile mismatch")
    by_id = {row["molecule_id"]: row for row in molecules}
    by_profile = {row["profile_id"]: row for row in profiles}
    require(len(by_id) == len(molecules) and len(by_profile) == len(profiles) and by_profile,
            "Duplicate molecule/profile ID or empty profiles")
    require(len({row["canonical_smiles"] for row in molecules}) == len(molecules), "Duplicate chemical identity")
    nonstereo_groups = defaultdict(list)
    for identity, row in by_id.items():
        smiles = row["canonical_smiles"]
        mol = Chem.MolFromSmiles(smiles)
        require(mol is not None and Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) == smiles,
                "Noncanonical candidate identity")
        require(identity == "curr_" + digest(smiles.encode())[:16], "Molecule identity hash mismatch")
        require(row.get("split") == "quarantine" and row.get("training_authorized") is False
                and row.get("external_overlap_status") == "unchecked", "Candidate admission flag mismatch")
        nonstereo = Chem.Mol(mol)
        Chem.RemoveStereochemistry(nonstereo)
        nonstereo_groups[Chem.MolToSmiles(nonstereo, canonical=True, isomericSmiles=True)].append(identity)
    require(all(KEY.fullmatch(key) for key in [*by_id, *by_profile]), "Unsafe molecule/profile ID")
    samples, rejected, journal = (lines(raw[name]) for name in ("samples.jsonl", "rejected.jsonl", "progress.jsonl"))
    expected = {(identity, profile) for identity in by_id for profile in by_profile}
    actual = [(row["molecule_id"], row["profile_id"]) for row in [*samples, *rejected]]
    require(len(actual) == len(set(actual)) and set(actual) == expected, "Incomplete/duplicate render denominator")
    require(len(journal) == len(expected), "Incomplete progress journal")
    journal_map = {(row["molecule_id"], row["profile_id"]): row for row in journal}
    require(len(journal_map) == len(journal) and set(journal_map) == expected, "Duplicate/wrong journal denominator")
    visibility = lines(raw["visible-coverage.jsonl"])
    visible_map = {row["sample_key"]: row for row in visibility}
    require(len(visible_map) == len(visibility) == len(samples), "Duplicate/incomplete visibility denominator")
    keys = [row["sample_key"] for row in samples]
    require(len(keys) == len(set(keys)), "Duplicate sample key")
    require(set(visible_map) == set(keys), "Visibility sample IDs differ")
    expected_files = {f"samples/{key}{suffix}" for key in keys for suffix in SUFFIXES}
    require(set(hashes) == expected_files, "Sample SHA manifest membership mismatch")
    actual_files = {"samples/" + path.name for path in (root / "samples").iterdir()}
    require(actual_files == expected_files, "Missing or unmanifested sample files")
    for row in rejected:
        require(row["canonical_smiles"] == by_id[row["molecule_id"]]["canonical_smiles"] and row.get("reason"),
                "Invalid rejected identity/reason")
        require(journal_map[(row["molecule_id"], row["profile_id"])] == {"status": "quality_rejected", **row},
                "Rejected journal row mismatch")
    byte_groups, pixel_groups, totals, drawing = defaultdict(list), defaultdict(list), Counter(), Counter()
    from Data.sdf.chemvector_curriculum_v0_2.coverage import summarize, visible_coverage
    require(decode(raw["planned-coverage.json"]) == summarize(molecules), "Planned coverage mismatch")
    for index, row in enumerate(samples, 1):
        key, identity, profile = row["sample_key"], row["molecule_id"], row["profile_id"]
        require(key == f"{identity}__{profile}" and KEY.fullmatch(key), "Sample key does not match identity/profile")
        payload = {suffix: read(root / "samples" / (key + suffix)) for suffix in SUFFIXES}
        checked_sha = {suffix: digest(data) for suffix, data in payload.items()}
        for suffix, observed in checked_sha.items():
            require(observed == hashes[f"samples/{key}{suffix}"], f"Sample artifact SHA mismatch: {key}{suffix}")
        meta, graph, layers, text = (decode(payload[suffix]) for suffix in
                                   (".meta.json", ".graph.json", ".layers.json", ".text-records.json"))
        require(meta["sha256"] == {suffix: value for suffix, value in checked_sha.items() if suffix != ".meta.json"},
                f"Internal metadata SHA mismatch: {key}")
        require(all(meta.get(field) == row[field] for field in row), f"Sample/metadata mismatch: {key}")
        require(all(meta.get(field) == value for field, value in by_id[identity].items()),
                f"Metadata/candidate provenance mismatch: {key}")
        require(meta["profile"] == by_profile[profile], f"Metadata profile mismatch: {key}")
        require(meta["canonical_smiles"] == graph["canonical_smiles"] == by_id[identity]["canonical_smiles"],
                f"Metadata graph identity mismatch: {key}")
        require(graph["molecule_id"] == layers["molecule_id"] == identity and layers["sample_key"] == key
                and layers["profile_id"] == profile, f"Bundle identity mismatch: {key}")
        require(all(owner.get("training_authorized") is False for owner in (meta, graph, layers)) and
                all(owner.get("split") == "quarantine" and owner.get("external_overlap_status") == "unchecked"
                    for owner in (meta, layers)), f"Bundle admission flags mismatch: {key}")
        for owner in (layers, text):
            require(owner["source_png_sha256"] == checked_sha[".png"] and
                    owner["source_svg_sha256"] == checked_sha[".svg"], f"Same-source binding mismatch: {key}")
        require(layers["text_records_sha256"] == checked_sha[".text-records.json"], f"Text binding mismatch: {key}")
        validate_records(graph, layers, text)
        na, nb = validate_sdf(payload[".sdf"], graph, meta["canonical_smiles"])
        totals.update(atoms=na, bonds=nb, verified_files=len(payload))
        visible = visible_coverage(layers)
        require(visible_map[key] == {"sample_key": key, **visible}, f"Visible coverage mismatch: {key}")
        require(journal_map[(identity, profile)] ==
                {"status": "accepted", **row, "visible_coverage": visible, "sha256": checked_sha},
                f"Accepted journal row mismatch: {key}")
        drawing.update(visible["drawing_directions"])
        description = {"sample_key": key, "molecule_id": identity, "canonical_smiles": meta["canonical_smiles"]}
        byte_groups[checked_sha[".png"]].append(description)
        with Image.open(io.BytesIO(payload[".png"])) as image:
            require(image.format == "PNG" and image.mode == "RGB" and image.size == (768, 768),
                    f"Unexpected original PNG format: {key}")
            image.load()
            pixel_groups[digest(image.tobytes())].append(description)
        if index % 1000 == 0 or index == len(samples):
            progress(f"Audited {index}/{len(samples)} accepted images", flush=True)
    def repeated(groups):
        return {key: rows for key, rows in groups.items() if len(rows) > 1}
    def conflicts(groups):
        return {key: rows for key, rows in groups.items() if len({r["canonical_smiles"] for r in rows}) > 1}
    require(duplicates["duplicate_png_groups"] == repeated(byte_groups) and
            duplicates["conflicting_chemical_identity_groups"] == conflicts(byte_groups), "PNG duplicate ledger mismatch")
    by_molecule = Counter(row["molecule_id"] for row in samples)
    expected_summary = {
        "unique_molecules": len(molecules), "planned_images": len(expected), "accepted_images": len(samples),
        "rejected_images": len(rejected), "source_records": sum(len(row["source_record_ids"]) for row in molecules),
        "distinct_png_files": len(byte_groups), "same_png_extra_rows": len(samples) - len(byte_groups),
        "cross_identity_same_png_groups": len(conflicts(byte_groups)),
        "cross_identity_same_png_rows": sum(len(rows) for rows in conflicts(byte_groups).values()),
        "molecules_with_images": len(by_molecule),
        "molecules_with_all_profiles": sum(count == len(profiles) for count in by_molecule.values()),
        "molecules_with_no_images": [row["molecule_id"] for row in molecules if row["molecule_id"] not in by_molecule],
        "images_by_profile": dict(Counter(row["profile_id"] for row in samples)),
        "images_by_stage_membership": dict(Counter(stage for row in samples for stage in row["stages"])),
        "rejections_by_reason": dict(Counter(row["reason"] for row in rejected)),
        "all_sample_files_sha256_rechecked": len(hashes), "frozen_sources_verified": len(plan["input_sha256"]),
    }
    for field, expected_value in expected_summary.items():
        require(summary.get(field) == expected_value, f"Summary count mismatch: {field}")
    require(plan["selected_molecules"] == len(molecules) and plan["planned_images"] == len(expected)
            and plan["candidate_molecules_before_preview_limit"] >= len(molecules), "Plan denominator mismatch")
    expected_coverage = summarize(molecules, samples)
    accepted_coverage = decode(raw["accepted-coverage.json"])
    require(all(accepted_coverage.get(key) == value for key, value in expected_coverage.items()), "Accepted coverage mismatch")
    actual_drawing = {}
    for field in ("drawing_directions", "drawing_bond_types", "chemical_bond_types", "visible_element_labels"):
        counts = Counter()
        for row in visibility:
            counts.update(row[field])
        actual_drawing[field] = dict(sorted(counts.items()))
    require(accepted_coverage["actual_drawing_instances"] == actual_drawing and
            accepted_coverage["nearby_target_non_edges"] == sum(row["nearby_target_non_edges"] for row in visibility),
            "Actual drawing coverage mismatch")
    verify_sources(plan["input_sha256"])
    for name, expected_sha in manifest_sha.items():
        require(digest(read(root / name)) == expected_sha, f"Manifest changed during audit: {name}")
    return {
        "status": "passed_artifact_integrity_only", "dataset": str(root.resolve()), "manifest_sha256": manifest_sha,
        "checked": expected_summary, "totals": dict(totals), "drawing_directions": dict(drawing),
        "stereo_removed_isotope_preserved_identities": len(nonstereo_groups),
        "multiple_stereo_variants_per_connectivity": {key: rows for key, rows in nonstereo_groups.items() if len(rows) > 1},
        "connectivity_identity_definition": "RemoveStereochemistry on a molecule copy, then canonical isomeric SMILES; isotope, charge and bond orders retained. Not scaffold grouping or a train/evaluation split.",
        "decoded_rgb_distinct_images": len(pixel_groups),
        "decoded_rgb_cross_identity_conflicts": conflicts(pixel_groups),
        "byte_identical_cross_identity_conflicts": conflicts(byte_groups),
        "blocking_cross_identity_pixel_conflicts": bool(conflicts(pixel_groups) or conflicts(byte_groups)),
        "training_authorized": False, "external_identity_queries": 0, "gpu_used": False, "model_forwards": 0,
        "predicted_candidate_reachability": "not_measured", "formal_training_admission": "not_granted",
        "renderer_exit_status": "not_inferred_from_artifacts",
        "limitations": ["Artifact integrity does not prove model accuracy, chemical stability or independent evaluation eligibility.",
                        "Cross-identity pixel conflicts, if present, remain blockers; passing integrity does not resolve them.",
                        "No full target tensor collation, endpoint reachability, optimizer or OCR inference was performed."],
    }


def audit(dataset, output):
    output = Path(output)
    require(not output.resolve().is_relative_to(Path(dataset).resolve()), "Audit output must be outside the dataset")
    source_sha = digest(read(Path(__file__)))
    dependencies = {str(path.relative_to(REPO)): digest(read(path)) for path in
                    (REPO / "chemvector/training/curriculum/data.py",
                     REPO / "Data/sdf/text_source_renderer/renderer.py",
                     REPO / "Data/sdf/chemvector_curriculum_v0_2/coverage.py")}
    started = time.monotonic()
    # Exclusive reservation happens before reading data. Failures retain a
    # non-success receipt, never overwrite a previous audit, and propagate.
    with output.open("x", encoding="utf-8") as stream:
        receipt = {"schema_version": "chemvector-curriculum-v0.2-independent-audit", "status": "failed",
                   "auditor_source_sha256": source_sha, "auditor_dependency_sha256": dependencies,
                   "training_authorized": False, "gpu_used": False}
        try:
            receipt.update(check_dataset(dataset))
            require(digest(read(Path(__file__))) == source_sha, "Auditor source changed during audit")
            for name, expected in dependencies.items():
                require(digest(read(REPO / name)) == expected, f"Auditor dependency changed: {name}")
            return receipt
        except Exception as exc:
            receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            receipt["elapsed_seconds"] = time.monotonic() - started
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = audit(args.dataset, args.output)
    print(json.dumps({key: receipt[key] for key in ("status", "checked", "blocking_cross_identity_pixel_conflicts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
