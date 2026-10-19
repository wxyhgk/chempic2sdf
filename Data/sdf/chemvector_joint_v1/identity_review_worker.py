"""Data-side identity-only worker. Its sole output is a redacted receipt.

Production execution is through the isolated private_review launcher, not by
importing the legacy builder. Never emit exception messages or private records.
"""

from collections import Counter
import csv
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys

SCHEMA = "chemvector-private-identity-receipt-0.1"
SPLITS = ("fresh_dev", "sealed_test")
RULES = ("full_inchikey", "connectivity_prefix", "stereo_isotope_independent_group_key")
FIELDS = ("inchikey", "connectivity_prefix", "group_key")
VARIANTS = {"original", "rdkit", "coordgen", "stress"}


class ReviewFailure(Exception):
    """Only fixed, non-sensitive error codes may cross the process boundary."""


def require(condition, code):
    if not condition:
        raise ReviewFailure(code)


def set_hash(values):
    digest = sha256()
    for value in sorted(set(values)):
        payload = value.encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def csv_rows(payload, required):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    require(set(required) <= set(reader.fieldnames or []), "private_manifest_invalid")
    rows = list(reader)
    require(all(None not in row and all(row[k] for k in required) for row in rows), "private_manifest_invalid")
    return rows


def same_legacy_scaffold(left, right):
    """Representation normalization only: retain charge, isotope and stereo."""
    from rdkit import Chem
    if left == right:
        return True
    prefix = "ACYCLIC:"
    if left.startswith(prefix) != right.startswith(prefix):
        return False
    values = []
    for value in (left, right):
        molecule = Chem.MolFromSmiles(value.removeprefix(prefix))
        if molecule is None:
            return False
        values.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    return values[0] == values[1]


def committed_group(scaffold):
    """Apply the REQUESTED stereo/isotope-independent rule to stored evidence.

    Return a second, independently committed exclusion group. If it disagrees
    with the SMILES-derived group, retain BOTH rather than choosing a label.
    """
    from rdkit import Chem
    acyclic = scaffold.startswith("ACYCLIC:")
    molecule = Chem.MolFromSmiles(scaffold.removeprefix("ACYCLIC:"))
    if molecule is None:
        return None
    Chem.RemoveStereochemistry(molecule)
    for atom in molecule.GetAtoms():
        atom.SetIsotope(0)
    return ("acyclic:" if acyclic else "scaffold:") + Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def review(request_bytes, candidate_bytes, commitment_bytes, manifests, identity_function,
           identity_sha256, rdkit_version, *, expected_split_counts=None):
    """Pure review core; private input records must never be returned by callers."""
    request, candidates, commitment = map(json.loads, (request_bytes, candidate_bytes, commitment_bytes))
    require(request.get("schema_version") == "chemvector-identity-review-request-0.1"
            and request.get("training_authorized") is False
            and request.get("requested_checks") == list(RULES), "request_invalid")
    require(sha256(candidate_bytes).hexdigest() == request.get("candidate_identities_sha256"), "candidate_binding_failed")
    require(identity_sha256 == request.get("identity_implementation_sha256")
            and rdkit_version == request.get("rdkit_version") == commitment.get("rdkit_version"), "identity_implementation_mismatch")
    unsigned = {k: v for k, v in commitment.items() if k != "commitment_sha256"}
    canonical = (json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
    require(sha256(canonical).hexdigest() == commitment.get("commitment_sha256")
            == request.get("evaluation_commitment_sha256"), "commitment_binding_failed")
    expected = expected_split_counts or {"fresh_dev": 8000, "sealed_test": 4000}
    require(commitment.get("schema_version") == "chemvector-joint-commitment-1.0"
            and commitment.get("dataset") == "fresh_eval_12k"
            and commitment.get("sealed_access") == "locked_until_checkpoint_gate"
            and commitment.get("split_counts") == expected
            and commitment.get("samples") == sum(expected.values())
            and commitment.get("molecules") == sum(expected.values()) // 4, "commitment_invalid")
    require(set(manifests) == set(SPLITS), "private_manifest_invalid")
    require(len(candidates) == request.get("candidate_molecules") and candidates, "candidate_binding_failed")
    seen, image_count = set(), 0
    for row in candidates:
        pair = (row["source"], row["molecule_id"])
        require(row["source"] in {"curriculum", "full_molecule"}
                and re.fullmatch(r"[A-Za-z0-9_-]+", row["molecule_id"])
                and pair not in seen, "candidate_invalid")
        seen.add(pair)
        actual = identity_function(row["canonical_smiles"])
        require(all(row.get(field) == actual[field] for field in FIELDS), "candidate_invalid")
        require(type(row["image_count"]) is int and row["image_count"] > 0
                and row["image_count"] == len(set(row["sample_keys"])), "candidate_invalid")
        image_count += row["image_count"]
    require(image_count == request.get("candidate_images"), "candidate_binding_failed")
    committed_csv = {row["path"]: row for row in commitment["samples_csv"]}
    require(len(commitment["samples_csv"]) == 2
            and set(committed_csv) == {"samples.csv", "sealed_test/samples.csv"}, "commitment_invalid")
    private_molecules, samples_by_split = {}, {}
    all_sample_keys, all_molecule_ids = set(), set()
    for split in SPLITS:
        sample_payload, molecule_payload = manifests[split]
        name = "samples.csv" if split == "fresh_dev" else "sealed_test/samples.csv"
        binding = committed_csv[name]
        require(len(sample_payload) == binding["bytes"]
                and sha256(sample_payload).hexdigest() == binding["sha256"], "private_sample_binding_failed")
        molecules = csv_rows(molecule_payload, ("molecule_id", "inchikey", "scaffold", "canonical_isomeric_smiles", "split"))
        samples = csv_rows(sample_payload, ("sample_key", "molecule_id", "inchikey", "scaffold", "split", "variant"))
        require(len(samples) == expected[split] and len(molecules) * 4 == expected[split], "private_coverage_mismatch")
        by_id = {row["molecule_id"]: row for row in molecules}
        require(len(by_id) == len(molecules) and not all_molecule_ids.intersection(by_id), "private_manifest_invalid")
        all_molecule_ids.update(by_id)
        variants = {mid: set() for mid in by_id}
        for row in samples:
            mid, key, variant = row["molecule_id"], row["sample_key"], row["variant"]
            require(mid in by_id and key not in all_sample_keys and variant in VARIANTS
                    and variant not in variants[mid] and row["split"] == split, "private_manifest_invalid")
            require(all(row[field] == by_id[mid][field] for field in ("inchikey", "scaffold", "split")), "private_manifest_invalid")
            all_sample_keys.add(key)
            variants[mid].add(variant)
        require(all(value == VARIANTS for value in variants.values()), "private_coverage_mismatch")
        private_molecules[split], samples_by_split[split] = molecules, len(samples)
    all_molecules = [row for split in SPLITS for row in private_molecules[split]]
    for field in ("molecule_id", "inchikey", "scaffold"):
        require(set_hash(row[field] for row in all_molecules) == commitment.get(field + "_set_sha256"), "private_set_binding_failed")
    indexes, unresolved = {}, {}
    legacy_representation_differences, legacy_attribute_differences, group_binding_mismatches = 0, 0, 0
    for split in SPLITS:
        indexes[split] = {field: set() for field in FIELDS}
        unresolved[split] = 0
        for row in private_molecules[split]:
            try:
                ids = identity_function(row["canonical_isomeric_smiles"])
            except Exception:
                unresolved[split] += 1
                continue
            require(ids["inchikey"] == row["inchikey"], "private_key_mismatch")
            legacy_same_graph = same_legacy_scaffold(ids["legacy_scaffold"], row["scaffold"])
            legacy_representation_differences += legacy_same_graph and ids["legacy_scaffold"] != row["scaffold"]
            # The request intentionally ignores these attributes for grouping,
            # while full InChIKey equality above remains mandatory and exact.
            legacy_group = committed_group(row["scaffold"])
            if legacy_group != ids["group_key"]:
                group_binding_mismatches += 1
                if legacy_group is None:
                    unresolved[split] += 1
                    continue
            else:
                legacy_attribute_differences += not legacy_same_graph
            for field in FIELDS:
                indexes[split][field].add(ids[field])
            # Conservative membership only, NOT a repair of evaluation labels.
            indexes[split]["group_key"].add(legacy_group)
    excluded, counts = [], Counter({rule: 0 for rule in RULES})
    for row in candidates:
        hits = [rule for rule, field in zip(RULES, FIELDS)
                if any(row[field] in indexes[split][field] for split in SPLITS)]
        if hits:
            # No private ID/key/SMILES, split membership, nearest neighbour or image key.
            excluded.append({"source": row["source"], "molecule_id": row["molecule_id"], "rules": hits})
            counts.update(hits)
    return {
        "schema_version": SCHEMA,
        "status": "incomplete" if sum(unresolved.values()) else ("checked_conservative" if group_binding_mismatches else "checked"),
        "training_authorized": False,
        "request_sha256": sha256(request_bytes).hexdigest(),
        "candidate_identities_sha256": sha256(candidate_bytes).hexdigest(),
        "evaluation_commitment_sha256": commitment["commitment_sha256"],
        "identity_implementation_sha256": identity_sha256, "rdkit_version": rdkit_version,
        "candidate_molecules": len(candidates), "candidate_images": image_count,
        "evaluation_samples_by_split": samples_by_split,
        "evaluation_molecules_by_split": {split: len(private_molecules[split]) for split in SPLITS},
        "unresolved_molecules_by_split": unresolved,
        "sample_manifest_hashes_verified": 2, "identity_set_commitments_verified": 3,
        "legacy_scaffold_equivalent_representation_differences": legacy_representation_differences,
        "legacy_scaffold_differences_resolved_by_requested_group_rule": legacy_attribute_differences,
        "committed_group_binding_mismatches": group_binding_mismatches,
        "group_comparison_policy": "union_of_smiles_derived_and_committed_legacy_groups",
        "candidate_overlap_counts_by_rule": dict(counts), "excluded_candidates": excluded,
        "cross_eval_split_overlap_counts_by_rule": {
            rule: len(indexes["fresh_dev"][field] & indexes["sealed_test"][field])
            for rule, field in zip(RULES, FIELDS)},
        "scope": "manifest_identity_only", "images_or_sdf_read": False,
    }


def main():
    # Suppress Python tracebacks AND native RDKit/InChI diagnostics before data
    # access. The saved pipe is used exactly once, for the allowlisted receipt.
    output_fd = os.dup(1)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    os.close(null_fd)
    result = {"schema_version": SCHEMA, "status": "failed", "failure_code": "worker_failed", "training_authorized": False}
    try:
        sys.dont_write_bytecode = True
        sys.path.extend(["/packages", "/runtime/lib/python3.13/site-packages"])
        from rdkit import rdBase
        require(not any(Path("/dev").glob("nvidia*")), "sandbox_invalid")
        status = Path("/proc/self/status").read_text()
        require(re.search(r"^CapEff:\s+0+$", status, re.MULTILINE), "sandbox_invalid")
        for root in ("/inputs", "/private"):
            require(all(os.statvfs(path).f_flag & os.ST_RDONLY for path in Path(root).iterdir()), "sandbox_invalid")
        module_path = Path("/code/admission.py")
        spec = importlib.util.spec_from_file_location("pinned_identities", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with rdBase.BlockLogs():
            result = review(
                Path("/inputs/request.json").read_bytes(), Path("/inputs/candidates.json").read_bytes(),
                Path("/inputs/commitment.json").read_bytes(),
                {split: (Path(f"/private/{split}_samples.csv").read_bytes(),
                         Path(f"/private/{split}_molecules.csv").read_bytes()) for split in SPLITS},
                module.identities, sha256(module_path.read_bytes()).hexdigest(), rdBase.rdkitVersion)
        result["worker_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    except ReviewFailure as error:
        result = {"schema_version": SCHEMA, "status": "failed", "failure_code": str(error), "training_authorized": False}
    except BaseException:
        pass
    payload = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
    with os.fdopen(output_fd, "wb") as output:
        output.write(payload)


if __name__ == "__main__":
    main()
