"""CPU-only new-render pilot from committed STYLE TRAIN identities.

No legacy builder import, benchmark payload access, fresh-eval reads, training,
or admission switch. Existing images and their annotations are never relabeled.
"""

import argparse
from collections import Counter, defaultdict, deque
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import tarfile

from rdkit import Chem

from Data.sdf.chemvector_curriculum_v0_1.build import build
from Data.sdf.chemvector_curriculum_v0_1.catalog import merged_records
from .admission import _cached_keys, identities

VARIANTS = {"original", "rdkit", "coordgen", "stress"}


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(root, name):
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlinked train source: {name}")
    return path.read_bytes()


def _set_hash(values):
    digest = sha256()
    for value in sorted(set(values)):
        data = value.encode()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _legacy_json(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def read_train_source(root):
    """Only fixed train manifests; never follow manifest paths into eval data."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Symlinked train root")
    # Reject non-train commitments before opening any per-molecule manifest.
    raw = {name: _read(root, name) for name in ("commitment.json", "validation.json")}
    commitment, validation = (json.loads(raw[name]) for name in ("commitment.json", "validation.json"))
    if (commitment.get("schema_version") != "chemvector-joint-commitment-1.0"
            or commitment.get("dataset") != "style_train_100k"
            or commitment.get("sealed_access") != "not_applicable"
            or set(commitment.get("split_counts", {})) != {"train"}
            or validation.get("dataset") != "style_train_100k"
            or validation.get("status") != "passed" or validation.get("errors")):
        raise ValueError("Expected passed, train-only style dataset commitment")
    unsigned = {k: v for k, v in commitment.items() if k != "commitment_sha256"}
    if sha256(_legacy_json(unsigned)).hexdigest() != commitment.get("commitment_sha256"):
        raise ValueError("Train commitment self-hash mismatch")
    raw["molecules.csv"] = _read(root, "molecules.csv")
    molecules = list(csv.DictReader(io.StringIO(raw["molecules.csv"].decode())))
    if (len(molecules) != commitment["molecules"] or not molecules
            or len({r["molecule_id"] for r in molecules}) != len(molecules)
            or any(r["split"] != "train" or r["source"] not in {"chembl", "coconut"} for r in molecules)):
        raise ValueError("Invalid train-only molecule manifest")
    for field in ("molecule_id", "inchikey", "scaffold"):
        if _set_hash(r[field] for r in molecules) != commitment[field + "_set_sha256"]:
            raise ValueError(f"Committed train {field} set mismatch")
    csv_records = commitment["samples_csv"]
    if len(csv_records) != 1 or csv_records[0]["path"] != "samples.csv":
        raise ValueError("Only the explicit train samples.csv is supported")
    payload = _read(root, "samples.csv")
    if (len(payload) != csv_records[0]["bytes"]
            or sha256(payload).hexdigest() != csv_records[0]["sha256"]):
        raise ValueError("Train sample manifest commitment mismatch")
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    if (len(rows) != commitment["samples"] or commitment["split_counts"]["train"] != len(rows)
            or len({r["sample_key"] for r in rows}) != len(rows)):
        raise ValueError("Train image count or sample key mismatch")
    by_id = {r["molecule_id"]: r for r in molecules}
    variants = defaultdict(set)
    for row in rows:
        if (row["split"] != "train" or row["variant"] not in VARIANTS
                or not re.fullmatch(r"train_[0-9]+_(original|rdkit|coordgen|stress)", row["sample_key"])
                or not re.fullmatch(r"shards/train-[0-9]{6}\.tar", row["output_shard"])
                or row["molecule_id"] not in by_id):
            raise ValueError("Non-train or unsafe source sample")
        if any(row[k] != by_id[row["molecule_id"]][k] for k in ("inchikey", "scaffold", "source", "source_id")):
            raise ValueError("Sample/molecule manifest identity mismatch")
        if row["variant"] in variants[row["molecule_id"]]:
            raise ValueError("Duplicate train molecular variant")
        variants[row["molecule_id"]].add(row["variant"])
    if set(variants) != set(by_id) or any(v != VARIANTS for v in variants.values()):
        raise ValueError("Incomplete four-variant train molecules")
    evidence = {"root": str(root.resolve()), "dataset": "style_train_100k",
                "commitment_sha256": commitment["commitment_sha256"],
                "input_sha256": {name: sha256(data).hexdigest() for name, data in raw.items()},
                "samples_csv_sha256": sha256(payload).hexdigest(),
                "source_molecules": len(molecules), "source_images": len(rows)}
    return molecules, rows, commitment, evidence


def size_bucket(heavy_atoms):
    for lower, upper in ((8, 20), (21, 40), (41, 60)):
        if lower <= heavy_atoms <= upper:
            return f"{lower}-{upper}"
    return None


def select_molecules(molecules, index, *, count, seed):
    """Deterministic source/size round-robin, one molecule per core/scaffold.

    Cached positives exclude; negatives never constitute complete admission.
    Selection is independent of renderer success and model predictions.
    """
    if type(count) is not int or not 1 <= count <= 2000 or type(seed) is not int:
        raise ValueError("Expected a bounded pilot size (1..2000) and integer seed")
    if index.get("schema_version") != "chemvector-joint-overlap-index-0.1":
        raise ValueError("Unsupported cached public identity index")
    keys, key_audit = _cached_keys(index, "benchmark_inchikeys")
    if not keys:
        raise ValueError("Missing valid public identity keys")
    prefixes = {k[:14] for k in keys}
    pools, skipped = defaultdict(list), Counter()
    for row in molecules:
        if row["split"] != "train":
            raise ValueError("Non-train source molecule")
        key = row["inchikey"]
        if not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", key):
            raise ValueError("Unresolved train molecule identity")
        if key in keys or key[:14] in prefixes:
            skipped["known_public_identity_or_connectivity"] += 1
            continue
        bucket = size_bucket(int(row["heavy_atoms"]))
        if bucket is None:
            skipped["outside_pilot_heavy_atom_range"] += 1
            continue
        pools[(row["source"], bucket)].append(row)
    queues = {bucket: deque(sorted(pool, key=lambda r: (
        sha256(f"{seed}:{r['molecule_id']}".encode()).hexdigest(), r["molecule_id"])))
        for bucket, pool in sorted(pools.items())}
    selected, groups, connectivities = [], set(), set()
    while len(selected) < count and any(queues.values()):
        for queue in queues.values():
            while queue:
                row = queue.popleft()
                ids = identities(row["canonical_isomeric_smiles"])
                mol = Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
                if ids["inchikey"] != row["inchikey"] or mol.GetNumHeavyAtoms() != int(row["heavy_atoms"]):
                    raise ValueError("Source SMILES/identity/size mismatch")
                if (len(Chem.GetMolFrags(mol)) != 1 or any(a.GetAtomicNum() == 0
                        or a.GetNumRadicalElectrons() or a.GetAtomMapNum() for a in mol.GetAtoms())):
                    skipped["outside_current_connected_molecule_contract"] += 1
                    continue
                if ids["group_key"] in groups or ids["connectivity_prefix"] in connectivities:
                    skipped["repeated_pilot_core_or_connectivity"] += 1
                    continue
                groups.add(ids["group_key"]); connectivities.add(ids["connectivity_prefix"])
                selected.append({**row, "group_key": ids["group_key"]})
                break
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError("Not enough distinct source/size/core candidates for requested pilot")
    return selected, {"seed": seed, "selected_molecules": len(selected),
        "policy": "round-robin source x heavy-atom bucket; hash-order; one molecule per core/scaffold",
        "selected_by_source_size": dict(Counter(f"{r['source']}:{size_bucket(int(r['heavy_atoms']))}" for r in selected)),
        "screening_skips": dict(skipped), "public_cache_key_audit": key_audit,
        "public_cache_audit": index.get("audit", {}), "training_authorized": False,
        "external_overlap_status": "partially_checked",
        "coverage_gaps": ["Cached public identities include unresolved records; negative hits are not clearance.",
                          "No fresh-dev or sealed identity membership check was performed.",
                          "Source rights and final mixed release still require review."],
        "validation_use": "forbidden_as_unseen_validation: these are old train identities"}


def verify_selected_assets(root, selected, samples, commitment):
    """Bind selected chemistry to committed tar payloads without extracting tar paths."""
    root = Path(root)
    selected_by_id = {r["molecule_id"]: r for r in selected}
    planned = [r for r in samples if r["molecule_id"] in selected_by_id]
    by_shard = defaultdict(dict)
    for row in planned:
        by_shard[row["output_shard"]][row["sample_key"]] = row
    committed = {r["path"]: r for r in commitment["tars"]}
    seen, verified = set(), []
    for name, rows in sorted(by_shard.items()):
        path = root / name
        if (name not in committed or (root / "shards").is_symlink() or path.is_symlink()
                or not path.is_file() or path.stat().st_size != committed[name]["bytes"]
                or file_hash(path) != committed[name]["sha256"]):
            raise ValueError(f"Committed train shard mismatch: {name}")
        expected = {key + suffix: (row, suffix, field) for key, row in rows.items()
                    for suffix, field in ((".chemmol.json", "chemmol_sha256"), (".sdf", "sdf_sha256"))}
        with tarfile.open(path, mode="r|") as archive:
            for member in archive:
                if member.name not in expected:
                    continue
                if member.name in seen or not member.isfile() or member.size > 4 * 1024 * 1024:
                    raise ValueError("Invalid/duplicate selected train tar member")
                seen.add(member.name)
                row, suffix, field = expected[member.name]
                data = archive.extractfile(member).read()
                if sha256(data).hexdigest() != row[field]:
                    raise ValueError("Selected train payload hash mismatch")
                source = selected_by_id[row["molecule_id"]]
                if suffix == ".chemmol.json":
                    graph = json.loads(data)
                    if (graph["molecule_id"] != source["molecule_id"] or graph["inchikey"] != source["inchikey"]
                            or graph["canonical_isomeric_smiles"] != source["canonical_isomeric_smiles"]):
                        raise ValueError("Committed train graph identity mismatch")
                else:
                    mol = Chem.MolFromMolBlock(data.decode(), sanitize=True, removeHs=False, strictParsing=True)
                    if mol is None or Chem.MolToSmiles(mol) != source["canonical_isomeric_smiles"]:
                        raise ValueError("Committed train SDF identity mismatch")
        if not set(expected) <= seen:
            raise ValueError("Missing selected train tar members")
        verified.append({"path": name, "sha256": committed[name]["sha256"], "selected_images": len(rows)})
        print(f"Verified train chemistry: {name}, {len(rows)} source images", flush=True)
    return planned, verified


def prepare(root, exclusion_index, *, count=200, seed=20260914):
    molecules, samples, commitment, provenance = read_train_source(root)
    index_bytes = Path(exclusion_index).read_bytes()
    selected, selection = select_molecules(molecules, json.loads(index_bytes), count=count, seed=seed)
    provenance["exclusion_index_sha256"] = sha256(index_bytes).hexdigest()
    return prepare_selected(root, selected, samples, commitment, provenance, selection)


def prepare_selected(root, selected, samples, commitment, provenance, selection):
    """Bind a caller's fixed train-only selection to original chemistry before rendering."""
    if (not selected or len({r["molecule_id"] for r in selected}) != len(selected)
            or any(r["split"] != "train" for r in selected)):
        raise ValueError("Explicit distinct train-only selection required")
    provenance = dict(provenance)
    selected_samples, shards = verify_selected_assets(root, selected, samples, commitment)
    entries = []
    for row in selected:
        entry = merged_records([{"record_id": row["molecule_id"], "name": row["molecule_id"],
            "stage": "full_molecule", "concepts": ["full_graph", "size_" + size_bucket(int(row["heavy_atoms"]))],
            "smiles": row["canonical_isomeric_smiles"]}])[0]
        entry["source_provenance"] = {"dataset": "style_train_100k", "source_split": "train",
            "source_molecule": row, "commitment_sha256": commitment["commitment_sha256"],
            "images": [s for s in selected_samples if s["molecule_id"] == row["molecule_id"]],
            "original_pixels_preserved": False, "original_layouts_preserved": False,
            "original_atom_order_preserved": False, "use": "new_render_train_replay_candidate"}
        entries.append(entry)
    provenance.update(schema_version="chemvector-full-molecule-source-0.1", selection=selection,
        verified_train_shards=shards,
        selected_source_images=len(selected_samples), fresh_eval_read=False, sealed_test_read=False,
        original_pixels_preserved=False, original_font_layout_coverage_preserved=False,
        source_tar_verification="full selected-shard SHA256 plus selected ChemMol/SDF chemistry; PNG not decoded",
        training_authorized=False, gpu_used=False, new_render_profiles="existing nine profiles, two fonts, one layout engine")
    return entries, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exclusion-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--molecules", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--task-timeout", type=float, default=60)
    parser.add_argument("--typography", choices=["ink-aware-v1", "stereo-aware-v1"], default="ink-aware-v1")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Use a fresh full-molecule pilot output directory")
    entries, provenance = prepare(args.source_root, args.exclusion_index, count=args.molecules, seed=args.seed)
    result = build(args.output, entries=entries, workers=args.workers, typography=args.typography,
                   task_timeout=args.task_timeout, source_provenance=provenance)
    print(json.dumps({k: result[k] for k in ("unique_molecules", "accepted_images", "rejected_images", "training_authorized")}))
    return 0 if result["accepted_images"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
