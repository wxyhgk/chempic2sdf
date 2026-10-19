"""Package candidate-side identities for independent data review, never admission.

Reads only explicit quarantine manifests, the old public exclusion cache and an
aggregate evaluation commitment. No images, SDF, checkpoint or private membership
index is opened. This is a local handoff artifact, not a review service or receipt.
"""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from rdkit import rdBase

from .admission import _cached_keys, identities


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def _read(path, inputs):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected an explicit, non-symlinked review input")
    payload = path.read_bytes()
    inputs[str(path.resolve())] = sha256(payload).hexdigest()
    return payload


def build_request(roots, exclusion_index, evaluation_commitment):
    if set(roots) != {"curriculum", "full_molecule"}:
        raise ValueError("Both curriculum and full_molecule sources are required")
    inputs = {}
    commitment = json.loads(_read(evaluation_commitment, inputs))
    unsigned = {k: v for k, v in commitment.items() if k != "commitment_sha256"}
    legacy_bytes = (json.dumps(unsigned, ensure_ascii=True, sort_keys=True,
                               separators=(",", ":")) + "\n").encode()
    if (commitment.get("schema_version") != "chemvector-joint-commitment-1.0"
            or commitment.get("dataset") != "fresh_eval_12k"
            or commitment.get("sealed_access") != "locked_until_checkpoint_gate"
            or commitment.get("split_counts") != {"fresh_dev": 8000, "sealed_test": 4000}
            or sha256(legacy_bytes).hexdigest() != commitment.get("commitment_sha256")):
        raise ValueError("Expected the locked aggregate fresh-evaluation commitment")
    # Deliberately never follow samples_csv/tars paths listed in the commitment.
    index = json.loads(_read(exclusion_index, inputs))
    if index.get("schema_version") != "chemvector-joint-overlap-index-0.1":
        raise ValueError("Unsupported cached exclusion index")
    keys, key_audit = _cached_keys(index, "benchmark_inchikeys")
    if not keys:
        raise ValueError("Missing valid public exclusion keys")
    prefixes = {key[:14] for key in keys}
    candidates, excluded, absent = [], [], []
    for domain, root in sorted(roots.items()):
        root = Path(root)
        if root.is_symlink():
            raise ValueError("Symlinked candidate source root")
        summary = json.loads(_read(root / "summary.json", inputs))
        if (summary.get("schema_version") != "curriculum-pilot-summary-0.1"
                or summary.get("status") != "quarantine_preview_only"
                or summary.get("training_authorized") is not False):
            raise ValueError("Expected a quarantine curriculum source")
        molecules = json.loads(_read(root / "molecules.json", inputs))
        samples = [json.loads(line) for line in _read(root / "samples.jsonl", inputs).splitlines() if line.strip()]
        # Bind the artifact inventory, without claiming to reverify its payloads.
        _read(root / "sample-sha256.json", inputs)
        if (len(molecules) != summary["unique_molecules"]
                or len(samples) != summary["accepted_images"]
                or len({m["molecule_id"] for m in molecules}) != len(molecules)
                or len({s["sample_key"] for s in samples}) != len(samples)):
            raise ValueError("Source manifest count or identity mismatch")
        by_id = {m["molecule_id"]: m for m in molecules}
        sample_keys = defaultdict(list)
        for sample in samples:
            if (sample["molecule_id"] not in by_id
                    or not re.fullmatch(r"[A-Za-z0-9_-]+", sample["sample_key"])):
                raise ValueError("Invalid sample identity")
            sample_keys[sample["molecule_id"]].append(sample["sample_key"])
        for molecule in molecules:
            if molecule.get("split") != "quarantine" or molecule.get("training_authorized") is not False:
                raise ValueError("Source molecule is not quarantined")
            mid = molecule["molecule_id"]
            ids = identities(molecule["canonical_smiles"])
            row = {"source": domain, "molecule_id": mid,
                   "canonical_smiles": molecule["canonical_smiles"], **ids,
                   "sample_keys": sorted(sample_keys[mid]), "image_count": len(sample_keys[mid])}
            if domain == "full_molecule":
                provenance = molecule.get("source_provenance", {})
                original = provenance.get("source_molecule", {})
                if (provenance.get("source_split") != "train" or original.get("split") != "train"
                        or original.get("inchikey") != ids["inchikey"]):
                    raise ValueError("Full molecules require matching old-train provenance")
                row["source_molecule_id"] = original["molecule_id"]
                row["intended_use"] = "old_train_identity_rerender_not_unseen_validation"
            else:
                row["intended_use"] = "curriculum_candidate_split_not_assigned"
            if ids["inchikey"] in keys or ids["connectivity_prefix"] in prefixes:
                excluded.append(dict(row, reason="known_public_full_key_or_connectivity_hit"))
            elif not sample_keys[mid]:
                absent.append(dict(row, reason="no_accepted_render"))
            else:
                candidates.append(row)
    for rows in (candidates, excluded, absent):
        rows.sort(key=lambda r: (r["source"], r["molecule_id"]))
    groups = defaultdict(list)
    for row in candidates:
        groups[row["group_key"]].append({"source": row["source"], "molecule_id": row["molecule_id"]})
    report = {
        "schema_version": "chemvector-identity-review-request-0.1",
        "training_authorized": False, "external_overlap_status": "pending_independent_review",
        "rdkit_version": rdBase.rdkitVersion,
        "identity_implementation_sha256": sha256(Path(__file__).with_name("admission.py").read_bytes()).hexdigest(),
        "request_implementation_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": inputs,
        "candidate_identities_sha256": sha256(_bytes(candidates)).hexdigest(),
        "evaluation_commitment_sha256": commitment["commitment_sha256"],
        "candidate_molecules": len(candidates),
        "candidate_images": sum(r["image_count"] for r in candidates),
        "candidate_molecules_by_source": dict(Counter(r["source"] for r in candidates)),
        "candidate_images_by_source": {domain: sum(r["image_count"] for r in candidates if r["source"] == domain)
                                       for domain in sorted(roots)},
        "known_public_excluded_molecules": len(excluded),
        "known_public_excluded_images": sum(r["image_count"] for r in excluded),
        "molecules_without_images": len(absent),
        "cached_public_key_audit": key_audit,
        "cross_source_group_collisions": {group: rows for group, rows in sorted(groups.items())
                                          if len({r["source"] for r in rows}) > 1},
        "requested_checks": ["full_inchikey", "connectivity_prefix", "stereo_isotope_independent_group_key"],
        "return_contract": [
            "An authorized independent data custodian must bind results to request.json SHA256 and the evaluation commitment.",
            "Return coverage counts, unresolved counts and only candidate-side exclusion IDs; never hidden IDs/structures/images.",
            "Use identities() from the hash-pinned implementation on the private side; do not reuse stereo-sensitive legacy scaffold keys.",
            "Account for every candidate against both fresh_dev and sealed_test. Do not interpret missing/unknown as a negative hit.",
            "This request does not authorize private-data access and does not define or authenticate an approval receipt.",
        ],
        "remaining_blockers": [
            "Public cache is incomplete; unresolved official source identities need an approved exclusion decision.",
            "No independent fresh-dev/sealed-test exclusion evidence is available.",
            "Full-molecule development validation and an admitted mixed release are not supplied by old-train replay.",
        ],
        "payloads_reverified": False, "training_performed": False,
        "fresh_eval_sample_data_read": False, "sealed_test_sample_data_read": False,
    }
    return report, candidates, excluded, absent


def run(roots, exclusion_index, evaluation_commitment, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh review-request output directory")
    report, candidates, excluded, absent = build_request(roots, exclusion_index, evaluation_commitment)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("request.json", report), ("candidate-identities.json", candidates),
                        ("known-public-exclusions.json", excluded), ("no-render.json", absent)):
        with (output / name).open("xb") as stream:
            stream.write(_bytes(value))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("curriculum-root", "full-molecule-root", "exclusion-index", "evaluation-commitment", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args(argv)
    report = run({"curriculum": args.curriculum_root, "full_molecule": args.full_molecule_root},
                 args.exclusion_index, args.evaluation_commitment, args.output)
    print(json.dumps({k: report[k] for k in ("candidate_molecules", "candidate_images",
        "known_public_excluded_images", "training_authorized")}, indent=2))


if __name__ == "__main__":
    main()
