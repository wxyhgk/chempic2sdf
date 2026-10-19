"""Read-only public exclusion evidence for a fixed candidate request.

Normal source identities are checked by full key OR connectivity prefix. Failed
identities use a necessary elemental lower bound, masking opaque SUP/alias atoms.
Every candidate that might still overlap is excluded, with ALL image variants.
Native Molfile/InChI results are supplementary positive-only evidence, never a
reason to ignore S-groups, repair source labels, or admit a negative cache hit.
"""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import re

from rdkit import Chem, rdBase
from rdkit.Chem import rdinchi

from .admission import _cached_keys, identities
from .ctab_inventory import UnsupportedInventory, deficiency_witness, literal_inventory, molecular_inventory

SUBSETS = ("uspto_10k", "uspto_10k_l", "uspto_10k_abb")
KEY = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def source_identity(block):
    with rdBase.BlockLogs():
        try:
            molecule = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False, strictParsing=False)
        except Exception:
            return None, "molblock_exception"
        if molecule is None:
            return None, "sanitization_or_parse_failed"
        try:
            key = Chem.MolToInchiKey(molecule)
        except Exception:
            return None, "inchi_key_exception"
        return (key, "valid") if KEY.fullmatch(key) else (None, "inchi_key_empty_or_invalid")


def native_probe(block):
    """No conversion options or source modifications; warnings remain visible."""
    with rdBase.BlockLogs():
        try:
            inchi, code, message, _, _ = rdinchi.MolBlockToInchi(block, options="")
            key = rdinchi.InchiToInchiKey(inchi) if inchi else ""
            return {"return_code": code, "message": message,
                    "key": key if KEY.fullmatch(key) and code in {0, 1} else None,
                    "use": "additional_positive_exclusion_only"}
        except Exception:
            return {"return_code": None, "message": "native_conversion_exception", "key": None,
                    "use": "additional_positive_exclusion_only"}


def candidate_inventory(rows):
    by_ref, inventories = {}, {}
    for row in rows:
        ref = row["source"] + ":" + row["molecule_id"]
        if (ref in by_ref or row["source"] not in {"curriculum", "full_molecule"}
                or not re.fullmatch(r"[A-Za-z0-9_-]+", row["molecule_id"])
                or type(row["image_count"]) is not int or row["image_count"] <= 0
                or row["image_count"] != len(set(row["sample_keys"]))):
            raise ValueError("Invalid/duplicate candidate identity")
        ids = identities(row["canonical_smiles"])
        if any(row[field] != ids[field] for field in ("inchikey", "connectivity_prefix", "group_key")):
            raise ValueError("Candidate molecular identity mismatch")
        by_ref[ref] = row
        inventories[ref] = molecular_inventory(Chem.MolFromSmiles(row["canonical_smiles"]))
    if not by_ref:
        raise ValueError("Empty candidate request")
    return by_ref, inventories


def screen_unresolved(block, candidates, inventories):
    try:
        inventory, unsupported = literal_inventory(block), False
    except UnsupportedInventory:
        inventory, unsupported = None, True
    native = native_probe(block)
    pairs = []
    for ref, row in candidates.items():
        witness = deficiency_witness(inventories[ref], inventory)
        native_hit = bool(native["key"] and row["connectivity_prefix"] == native["key"][:14])
        pairs.append({"candidate_ref": ref, "deficiency_witness": witness,
                      "native_key_positive_hit": native_hit,
                      "excluded_conservatively": witness is None or native_hit})
    return {"inventory": inventory, "unsupported_inventory": unsupported, "native_probe": native,
            "pairs": pairs}


def scan_public(root, candidates, inventories, *, expected_per_subset=10000):
    root = Path(root).resolve()
    index, unresolved, evidence, excluded, bindings = [], [], [], defaultdict(list), {}
    by_prefix = defaultdict(list)
    for ref, candidate in candidates.items():
        by_prefix[candidate["connectivity_prefix"]].append(ref)
    for subset in SUBSETS:
        manifest = root / "manifests" / (subset + ".csv")
        if manifest.is_symlink():
            raise ValueError("Symlinked public manifest")
        payload = manifest.read_bytes()
        bindings[subset] = sha256(payload).hexdigest()
        rows = list(csv.DictReader(io.StringIO(payload.decode())))
        if len(rows) != expected_per_subset or len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("Incomplete/duplicate public manifest")
        for number, row in enumerate(rows, 1):
            sid = row["sample_id"]
            expected = f"prepared/{subset}/source_structures/{sid}.mol"
            path = root / expected
            if (not sid.startswith(subset + "_") or not re.fullmatch(r"[A-Za-z0-9_-]+", sid)
                    or row["source_structure_file"] != expected or path.is_symlink()
                    or not path.resolve().is_relative_to(root / "prepared" / subset / "source_structures")):
                raise ValueError("Unsafe public source-structure path")
            data = path.read_bytes()
            source_hash = sha256(data).hexdigest()
            if source_hash != row["source_structure_sha256"]:
                raise ValueError("Public source hash mismatch")
            block = data.decode()
            key, status = source_identity(block)
            record = {"sample_id": sid, "subset": subset, "source_structure_sha256": source_hash,
                      "normal_identity_status": status, "inchikey": key}
            index.append(record)
            if key:
                for ref in by_prefix[key[:14]]:
                    excluded[ref].append({"sample_id": sid, "reason": "public_key_or_connectivity_hit"})
            else:
                screened = screen_unresolved(block, candidates, inventories)
                for pair in screened["pairs"]:
                    evidence.append({"sample_id": sid, **pair})
                    if pair["excluded_conservatively"]:
                        excluded[pair["candidate_ref"]].append({"sample_id": sid,
                            "reason": "native_key_positive_hit" if pair["native_key_positive_hit"] else "cannot_rule_out_unresolved_identity"})
                unresolved.append({**record, **{k: v for k, v in screened.items() if k != "pairs"},
                                   "candidate_pairs_disproved": sum(p["deficiency_witness"] is not None for p in screened["pairs"]),
                                   "candidate_pairs_retained_as_risk": sum(p["excluded_conservatively"] for p in screened["pairs"])})
            if number % 1000 == 0:
                print(f"Public identity screening: {subset} {number}/{len(rows)}", flush=True)
    return index, unresolved, evidence, excluded, bindings


def run(request_dir, benchmark_root, exclusion_index, output, *, private_receipt=None):
    request_dir, output = Path(request_dir), Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh public-review output directory")
    request_bytes = (request_dir / "request.json").read_bytes()
    candidate_bytes = (request_dir / "candidate-identities.json").read_bytes()
    request, rows = json.loads(request_bytes), json.loads(candidate_bytes)
    if (request.get("schema_version") != "chemvector-identity-review-request-0.1"
            or request.get("training_authorized") is not False
            or request.get("candidate_identities_sha256") != sha256(candidate_bytes).hexdigest()
            or request.get("identity_implementation_sha256") != digest(Path(__file__).with_name("admission.py"))
            or request.get("rdkit_version") != rdBase.rdkitVersion
            or request.get("candidate_molecules") != len(rows)
            or request.get("candidate_images") != sum(r["image_count"] for r in rows)):
        raise ValueError("Fixed candidate request binding failed")
    candidates, inventories = candidate_inventory(rows)
    cache_bytes = Path(exclusion_index).read_bytes()
    if request.get("input_sha256", {}).get(str(Path(exclusion_index).resolve())) != sha256(cache_bytes).hexdigest():
        raise ValueError("Cached public index differs from the fixed request")
    cache = json.loads(cache_bytes)
    if cache.get("schema_version") != "chemvector-joint-overlap-index-0.1":
        raise ValueError("Unsupported cached public index")
    keys, cache_audit = _cached_keys(cache, "benchmark_inchikeys")
    if not keys:
        raise ValueError("No valid cached public keys")
    private_result, private_bytes = None, None
    if private_receipt is not None:
        from .private_review import WORKER, validate_receipt
        private_bytes = Path(private_receipt).read_bytes()
        private_result = validate_receipt(json.loads(private_bytes), request_bytes, candidate_bytes, digest(WORKER))
        if private_result["status"] not in {"checked", "checked_conservative"}:
            raise ValueError("Private identity evidence is incomplete")
    index, unresolved, evidence, excluded, manifests = scan_public(benchmark_root, candidates, inventories)
    prefixes = {key[:14] for key in keys}
    for ref, row in candidates.items():
        if row["connectivity_prefix"] in prefixes:
            excluded[ref].append({"reason": "legacy_public_cache_positive"})
    if private_result:
        for row in private_result["excluded_candidates"]:
            excluded[row["source"] + ":" + row["molecule_id"]].append({"reason": "private_receipt_positive"})
    remaining = [row for ref, row in candidates.items() if ref not in excluded]
    rejected = [{**candidates[ref], "exclusion_reasons": reasons} for ref, reasons in sorted(excluded.items())]
    remaining_bytes = (json.dumps(remaining, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    report = {
        "schema_version": "chemvector-public-candidate-exclusion-0.1", "status": "screened_with_conservative_exclusions",
        "training_authorized": False, "rdkit_version": rdBase.rdkitVersion,
        "request_sha256": sha256(request_bytes).hexdigest(), "candidate_identities_sha256": sha256(candidate_bytes).hexdigest(),
        "public_manifest_sha256": manifests, "cached_index_sha256": sha256(cache_bytes).hexdigest(),
        "implementation_sha256": {name: digest(Path(__file__).with_name(name)) for name in ("public_review.py", "ctab_inventory.py", "admission.py")},
        "public_source_hashes_verified": len(index), "normal_identity_records": len(index) - len(unresolved),
        "unresolved_global_identity_records": len(unresolved),
        "supplementary_native_keys_generated": sum(bool(r["native_probe"]["key"]) for r in unresolved),
        "unsupported_inventory_records": sum(r["unsupported_inventory"] for r in unresolved),
        "unresolved_candidate_pairs_checked": len(evidence),
        "deficiency_witness_pairs": sum(r["deficiency_witness"] is not None for r in evidence),
        "candidate_molecules": len(rows), "candidate_images": sum(r["image_count"] for r in rows),
        "conservatively_excluded_molecules": len(rejected), "conservatively_excluded_images": sum(r["image_count"] for r in rejected),
        "remaining_molecules": len(remaining), "remaining_images": sum(r["image_count"] for r in remaining),
        "remaining_candidates_sha256": sha256(remaining_bytes).hexdigest(),
        "remaining_molecules_by_source": dict(Counter(r["source"] for r in remaining)),
        "remaining_ambiguous_pairs": sum(r["excluded_conservatively"] and r["candidate_ref"] not in excluded for r in evidence),
        "cached_key_audit": cache_audit,
        "policy": "Preserve every public record. Exclude whole candidate molecules with positive keys OR any unresolved possible overlap; a lower-bound match is NOT identity equality.",
        "proof_scope": "Literal V2000 non-H atoms outside SUP/alias groups are necessary in the full molecule; no fragment stripping, label expansion, salt removal or source chemical repair is permitted.",
        "private_receipt_inheritance": None if private_result is None else {
            "receipt_sha256": sha256(private_bytes).hexdigest(), "status": private_result["status"],
            "evaluation_commitment_sha256": private_result["evaluation_commitment_sha256"],
            "method": "Exact candidate subset of the hash-bound request; no new private query or private-data read."},
        "remaining_blockers": ["Training/source-rights/target and mixed release approvals are separate.",
                               "Full-molecule validation integration and the new evaluation group-isolation issue remain unresolved.",
                               "The 13 global identity gaps (if reproduced) are not repaired; the proof only covers this fixed candidate subset.",
                               "Other cached public positives are retained; their source files were not re-audited in this USPTO-only scan."],
        "images_read": False, "sources_modified": False, "cache_rewritten": False,
        "private_sample_data_read": False, "training_performed": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    documents = {"report.json": report, "unresolved-records.json": unresolved,
                 "remaining-candidates.json": remaining, "excluded-candidates.json": rejected,
                 "candidate-element-inventories.json": inventories}
    for name, value in documents.items():
        with (output / name).open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    for name, values in (("public-records.jsonl", index), ("unresolved-pair-evidence.jsonl", evidence)):
        with (output / name).open("x") as stream:
            for value in values:
                stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request-dir", "benchmark-root", "exclusion-index", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--private-receipt", type=Path)
    args = parser.parse_args(argv)
    result = run(args.request_dir, args.benchmark_root, args.exclusion_index, args.output, private_receipt=args.private_receipt)
    print(json.dumps({key: result[key] for key in ("public_source_hashes_verified", "unresolved_global_identity_records",
        "conservatively_excluded_molecules", "remaining_molecules", "remaining_images", "training_authorized")}, indent=2))


if __name__ == "__main__":
    main()
