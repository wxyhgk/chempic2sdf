"""Fail-closed, CPU-only curriculum identity screening using a cached index.

Never opens benchmark images/structures or sealed-test contents. The legacy
index is incomplete for full admission: report known overlaps and grouped split
PROPOSALS, but do not change quarantine labels or authorize training.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _cached_keys(index, name):
    entries = index.get(name)
    if not isinstance(entries, list):
        raise ValueError(f"Missing or invalid identity index field: {name}")
    valid = [key for key in entries if isinstance(key, str)
             and re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", key)]
    return set(valid), {"entries": len(entries), "valid_unique_keys": len(set(valid)),
                        "invalid_entries": len(entries) - len(valid)}


def identities(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid curriculum SMILES")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if canonical != smiles:
        raise ValueError("Curriculum manifest requires canonical isomeric SMILES")
    key = Chem.MolToInchiKey(molecule)
    if not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", key):
        raise ValueError("Unresolved curriculum molecular identity")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=True) or "ACYCLIC:" + canonical
    core = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(core)
    for atom in core.GetAtoms():
        atom.SetIsotope(0)
    # Same scaffold stays together; empty acyclic scaffolds fall back to a
    # stereo/isotope-independent full graph instead of one universal empty key.
    group_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=core, includeChirality=False)
    group = ("scaffold:" + group_scaffold if group_scaffold
             else "acyclic:" + Chem.MolToSmiles(core, canonical=True, isomericSmiles=True))
    return {"inchikey": key, "connectivity_prefix": key[:14], "legacy_scaffold": scaffold, "group_key": group}


def screen(molecules, samples, index, *, seed=20260914):
    if index.get("schema_version") != "chemvector-joint-overlap-index-0.1":
        raise ValueError("Unsupported identity index")
    keys, benchmark_key_audit = _cached_keys(index, "benchmark_inchikeys")
    combined, combined_key_audit = _cached_keys(index, "inchikeys")
    scaffolds = set(index["benchmark_scaffolds"])
    if not keys:
        raise ValueError("Missing valid benchmark identity keys")
    prefixes = {key[:14] for key in keys}
    key_audit = {"benchmark_inchikeys": benchmark_key_audit, "inchikeys": combined_key_audit}
    invalid_key_gaps = [
        f"Cached {name} contains {audit['invalid_entries']} invalid identity entries; "
        "they cannot clear molecular identity overlap. Counts may refer to the same unresolved records."
        for name, audit in key_audit.items() if audit["invalid_entries"]
    ]
    known, molecular_rows, seen_smiles = {}, [], set()
    for molecule in molecules:
        identity, smiles = molecule["molecule_id"], molecule["canonical_smiles"]
        if identity in known or smiles in seen_smiles:
            raise ValueError("Duplicate curriculum molecule identity")
        seen_smiles.add(smiles)
        ids = identities(smiles)
        exact, connectivity = ids["inchikey"] in keys, ids["connectivity_prefix"] in prefixes
        proposed = "excluded_known_benchmark" if exact or connectivity else (
            "candidate_validation" if int(sha256(f"{seed}:{ids['group_key']}".encode()).hexdigest()[:16], 16) % 5 == 0
            else "candidate_train")
        record = {"molecule_id": identity, "canonical_smiles": smiles, **ids,
                  "known_benchmark_exact_overlap": exact,
                  "known_benchmark_connectivity_overlap": connectivity,
                  "known_benchmark_scaffold_overlap": ids["legacy_scaffold"] in scaffolds,
                  "old_combined_identity_hit": ids["inchikey"] in combined,
                  "proposed_split": proposed, "training_authorized": False,
                  "external_overlap_status": "partially_checked"}
        known[identity] = record
        molecular_rows.append(record)
    sample_rows, seen_keys = [], set()
    for sample in samples:
        key, molecule_id = sample["sample_key"], sample["molecule_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key) or key in seen_keys or molecule_id not in known:
            raise ValueError("Invalid/duplicate sample identity in curriculum manifest")
        seen_keys.add(key)
        record = known[molecule_id]
        sample_rows.append({"sample_key": key, "molecule_id": molecule_id,
                            "group_key": record["group_key"], "proposed_split": record["proposed_split"],
                            "training_authorized": False})
    train_groups = {r["group_key"] for r in molecular_rows if r["proposed_split"] == "candidate_train"}
    validation_groups = {r["group_key"] for r in molecular_rows if r["proposed_split"] == "candidate_validation"}
    if train_groups & validation_groups:
        raise ValueError("Proposed train/validation groups overlap")
    report = {
        "schema_version": "chemvector-curriculum-admission-0.1", "route": "self_developed_chemvector",
        "training_authorized": False, "external_overlap_status": "partially_checked", "seed": seed,
        "checked_molecules": len(molecular_rows), "checked_images": len(sample_rows),
        "known_benchmark_exact_molecules": sum(r["known_benchmark_exact_overlap"] for r in molecular_rows),
        "known_benchmark_connectivity_molecules": sum(r["known_benchmark_connectivity_overlap"] for r in molecular_rows),
        "known_benchmark_scaffold_molecules": sum(r["known_benchmark_scaffold_overlap"] for r in molecular_rows),
        "molecules_by_proposal": dict(Counter(r["proposed_split"] for r in molecular_rows)),
        "images_by_proposal": dict(Counter(r["proposed_split"] for r in sample_rows)),
        "proposed_train_validation_group_overlap": 0,
        "grouping": "Murcko scaffold ignoring stereo/isotope; full core identity for acyclic graphs",
        "hard_exclusion": "Known benchmark full InChIKey OR connectivity-prefix hit; all image variants together",
        "scaffold_overlap_policy": "Reported as a risk, not silently treated as exact identity or a universal external scaffold exclusion",
        "cached_index_audit": index.get("audit", {}),
        "cached_identity_key_audit": key_audit,
        "coverage_gaps": [
            "Legacy benchmark index contains unresolved records; a negative hit is not proof of no overlap.",
            "No fresh-dev/sealed-test identity membership index was opened or checked.",
            "Combined legacy identity list mixes train/evaluation; its membership cannot grant training admission.",
        ] + invalid_key_gaps,
        "notes": ["Split names are proposals, not admitted train/validation data.",
                  "No source image, metadata, label or historical evaluation artifact is rewritten.",
                  "No guarantee of zero overlap with external pretrained data.",
                  "Model-target/visual-quality preflight is separate from identity screening."],
    }
    return report, molecular_rows, sample_rows


def require_training_admission(report):
    if (report.get("schema_version") != "chemvector-curriculum-admission-0.1"
            or report.get("training_authorized") is not True
            or report.get("external_overlap_status") != "checked"
            or report.get("coverage_gaps") != []):
        raise ValueError("Curriculum admission is incomplete; GPU training must not start")


def run(data_root, exclusion_index, output):
    output, root, exclusion_index = Path(output), Path(data_root), Path(exclusion_index)
    if output.exists():
        raise FileExistsError("Use a fresh curriculum admission output directory")
    paths = {"molecules": root / "molecules.json", "samples": root / "samples.jsonl", "index": exclusion_index}
    payload = {name: path.read_bytes() for name, path in paths.items()}
    report, molecules, samples = screen(json.loads(payload["molecules"]),
        [json.loads(row) for row in payload["samples"].splitlines() if row.strip()], json.loads(payload["index"]))
    report["inputs"] = {name: {"path": str(paths[name]), "sha256": sha256(data).hexdigest()} for name, data in payload.items()}
    report["source_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("report.json", report), ("molecules.json", molecules), ("split-proposal.json", samples)):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--exclusion-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.data_root, args.exclusion_index, args.output)
    print(json.dumps({k: report[k] for k in ("checked_molecules", "known_benchmark_exact_molecules", "images_by_proposal", "training_authorized")}, indent=2))


if __name__ == "__main__":
    main()
