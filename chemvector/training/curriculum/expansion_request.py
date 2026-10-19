"""Package only unique, compatible expansion candidates for identity review.

No review is performed. Read quarantine evidence, a public cache and the locked
aggregate commitment only; never follow its private manifest or tar paths.
"""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from rdkit import rdBase

from .admission import _cached_keys, identities
from .expansion_inventory import unique_candidates
from .review_request import _bytes, _read


def build_request(expansion_root, inventory_root, exclusion_index, evaluation_commitment):
    expansion_root, inventory_root = map(Path, (expansion_root, inventory_root))
    inputs = {}
    def read(path):
        return json.loads(_read(path, inputs))
    commitment = read(evaluation_commitment)
    unsigned = {k: v for k, v in commitment.items() if k != "commitment_sha256"}
    digest = sha256((json.dumps(unsigned, ensure_ascii=True, sort_keys=True,
                               separators=(",", ":")) + "\n").encode()).hexdigest()
    if (commitment.get("schema_version") != "chemvector-joint-commitment-1.0"
            or commitment.get("dataset") != "fresh_eval_12k"
            or commitment.get("sealed_access") != "locked_until_checkpoint_gate"
            or commitment.get("split_counts") != {"fresh_dev": 8000, "sealed_test": 4000}
            or commitment.get("rdkit_version") != rdBase.rdkitVersion
            or commitment.get("commitment_sha256") != digest):
        raise ValueError("Expected unchanged locked aggregate evaluation commitment")
    cache = read(exclusion_index)
    if cache.get("schema_version") != "chemvector-joint-overlap-index-0.1":
        raise ValueError("Unsupported public exclusion cache")
    keys, key_audit = _cached_keys(cache, "benchmark_inchikeys")
    if not keys:
        raise ValueError("Missing public exclusion keys")
    prefixes = {k[:14] for k in keys}
    inventory = read(inventory_root / "report.json")
    chosen = read(inventory_root / "unique-candidates.json")
    records = read(inventory_root / "all-pixel-records.json")
    plan = read(expansion_root / "plan.json")
    generation = read(expansion_root / "report.json")
    root = expansion_root / "dataset"
    summary = read(root / "summary.json")
    molecules = read(root / "molecules.json")
    samples = [json.loads(line) for line in _read(root / "samples.jsonl", inputs).splitlines()]
    _read(root / "sample-sha256.json", inputs)
    # Verify only the explicit candidate-side artifacts just read. Other inputs
    # (e.g. preflight details) stay hash references, not arbitrary paths to follow.
    for path in (expansion_root / "plan.json", expansion_root / "report.json",
                 *(root / name for name in ("summary.json", "molecules.json", "samples.jsonl", "sample-sha256.json"))):
        if inventory["input_sha256"].get(str(path.resolve())) != inputs[str(path.resolve())]:
            raise ValueError("Inventory is not bound to this expansion")
    expected, duplicates, conflicts = unique_candidates(records)
    by_key, by_id = {r["sample_key"]: r for r in samples}, {r["molecule_id"]: r for r in molecules}
    if (inventory.get("schema_version") != "chemvector-expansion-inventory-0.1"
            or inventory.get("training_authorized") is not False
            or inventory.get("admitted_new_train_images") != 0
            or chosen != expected or not chosen
            or len(chosen) != inventory["compatible_unique_candidate_images"]
            or len(duplicates) != inventory["compatible_duplicate_images_not_counted"]
            or len(conflicts) != inventory["conflicting_pixel_groups"]
            or len(records) != len(by_key) or {r["sample_key"] for r in records} != set(by_key)
            or len(samples) != len(by_key) or len(molecules) != len(by_id)
            or len(samples) != summary["accepted_images"]
            or summary["training_authorized"] is not False
            or generation["plan_sha256"] != inputs[str((expansion_root / "plan.json").resolve())]
            or plan["input_sha256"].get(str(Path(exclusion_index).resolve())) != inputs[str(Path(exclusion_index).resolve())]):
        raise ValueError("Invalid or inconsistent unique expansion candidate inventory")
    variants = defaultdict(list)
    for row in chosen:
        key, mid = row["sample_key"], row["molecule_id"]
        if (not row["target_compatible"] or mid not in by_id or by_key[key]["molecule_id"] != mid
                or row["canonical_smiles"] != by_id[mid]["canonical_smiles"]
                or row["profile_id"] != by_key[key]["profile_id"]):
            raise ValueError("Candidate chemistry/sample binding mismatch")
        variants[mid].append(key)
    candidates, coverage = [], Counter()
    for mid, sample_keys in sorted(variants.items()):
        molecule = by_id[mid]
        source = molecule["source_provenance"]
        original = source["source_molecule"]
        ids = identities(molecule["canonical_smiles"])
        if (molecule["training_authorized"] is not False or molecule["split"] != "quarantine"
                or source["source_split"] != "train" or original["split"] != "train"
                or ids["inchikey"] != original["inchikey"] or ids["connectivity_prefix"] in prefixes):
            raise ValueError("Candidate is not an eligible train-source identity")
        candidates.append(dict(source="full_molecule", molecule_id=mid,
            canonical_smiles=molecule["canonical_smiles"], **ids, sample_keys=sorted(sample_keys),
            image_count=len(sample_keys), source_molecule_id=original["molecule_id"],
            intended_use="old_train_identity_rerender_not_unseen_validation"))
        coverage[original["source"]] += len(sample_keys)
    report = dict(schema_version="chemvector-identity-review-request-0.1", training_authorized=False,
        external_overlap_status="pending_independent_review", rdkit_version=rdBase.rdkitVersion,
        identity_implementation_sha256=sha256(Path(__file__).with_name("admission.py").read_bytes()).hexdigest(),
        request_implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), input_sha256=inputs,
        candidate_identities_sha256=sha256(_bytes(candidates)).hexdigest(),
        evaluation_commitment_sha256=commitment["commitment_sha256"], candidate_molecules=len(candidates),
        candidate_images=sum(r["image_count"] for r in candidates),
        candidate_molecules_by_source={"full_molecule": len(candidates)},
        candidate_images_by_source={"full_molecule": len(chosen)}, origin_images=dict(coverage),
        cached_public_key_audit=key_audit,
        requested_checks=["full_inchikey", "connectivity_prefix", "stereo_isotope_independent_group_key"],
        return_contract=["Return counts and candidate-side exclusion IDs only, never private identities or structures.",
                         "Bind the receipt to these exact request/candidate bytes and the evaluation commitment."],
        payloads_reverified=False, training_performed=False, fresh_eval_sample_data_read=False,
        sealed_test_sample_data_read=False, private_identity_review_executed=False,
        public_review="cached positive exclusion only; not new candidate clearance",
        remaining_blockers=["Explicit authorization for a new isolated identity-only review is required.",
            "Current public clearance, exact source-use review and independent release remain required.",
            "This request does not change existing validation, authorize training, or restart the failed recipe."])
    return report, candidates


def run(expansion_root, inventory_root, exclusion_index, evaluation_commitment, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh expansion request output directory")
    report, candidates = build_request(expansion_root, inventory_root, exclusion_index, evaluation_commitment)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("request.json", report), ("candidate-identities.json", candidates)):
        with (output / name).open("xb") as stream:
            stream.write(_bytes(value))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("expansion-root", "inventory-root", "exclusion-index", "evaluation-commitment", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.expansion_root, args.inventory_root, args.exclusion_index, args.evaluation_commitment, args.output)
    print(json.dumps({k: report[k] for k in ("candidate_molecules", "candidate_images", "training_authorized")}))


if __name__ == "__main__":
    main()
