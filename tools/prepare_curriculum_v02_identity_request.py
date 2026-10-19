"""Freeze candidate-side review bytes after a passed selected-target preflight.

This is a request, NOT authorization and NOT an identity comparison. The only
evaluation-side input is the existing aggregate commitment; no path inside it
is followed. No public/private membership cache, worker, training or split.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from rdkit import rdBase

from chemvector.training.curriculum import admission
from tools import preflight_curriculum_v02_targets as preflight

COMMITMENT = preflight.quality.REPO / "Data/sdf/chemvector_joint_v1/fresh_eval_12k/commitment.json"
read, decode, digest, require = preflight.read, preflight.decode, preflight.digest, preflight.require


def assemble_candidates(molecules, selected):
    by_id = {row["molecule_id"]: row for row in molecules}
    require(len(by_id) == len(molecules), "Duplicate source molecule identity")
    keys, seen = defaultdict(list), set()
    for row in selected:
        mid, key = row["molecule_id"], row["sample_key"]
        require(mid in by_id and key not in seen and preflight.quality.KEY.fullmatch(key)
                and key == f"{mid}__{row['profile_id']}", "Invalid selected candidate/sample identity")
        seen.add(key)
        keys[mid].append(key)
    require(keys, "Empty review request")
    candidates = []
    for mid, sample_keys in sorted(keys.items()):
        molecule = by_id[mid]
        require(molecule.get("split") == "quarantine" and molecule.get("training_authorized") is False
                and molecule.get("external_overlap_status") == "unchecked", "Candidate is not quarantined")
        ids = admission.identities(molecule["canonical_smiles"])
        candidates.append(dict(source="curriculum", molecule_id=mid, canonical_smiles=molecule["canonical_smiles"],
                               **ids, sample_keys=sorted(sample_keys), image_count=len(sample_keys),
                               intended_use="foundation_expansion_quarantine_no_split_or_training_authorization"))
    return candidates


def validate_preflight(report, handoff, journal):
    selected = handoff["retained_candidates"]
    require(report.get("schema_version") == preflight.SCHEMA
            and report.get("status") == "passed_target_compatibility_only" and report.get("complete") is True
            and report.get("failed_images") == 0 and report.get("training_authorized") is False
            and report.get("training_performed") is False and report.get("external_identity_queries") == 0
            and report.get("external_overlap_status") == "unchecked", "A complete passed target preflight is required")
    require(report.get("dataset") == handoff["dataset"]
            and report.get("selected_images") == report.get("checked_images") == report.get("compatible_images") == len(selected)
            and report.get("checked_molecules") == len({row["molecule_id"] for row in selected}),
            "Preflight selected denominator mismatch")
    require(digest(journal) == report.get("journal_sha256"), "Preflight journal binding mismatch")
    rows = [decode(line) for line in journal.splitlines() if line.strip()]
    require(len(rows) == len(selected) and all(row.get("compatible") is True for row in rows)
            and [(row["sample_key"], row["molecule_id"]) for row in rows]
            == [(row["sample_key"], row["molecule_id"]) for row in selected],
            "Journal differs from the fixed selection")


def validate_commitment(commitment):
    unsigned = {key: value for key, value in commitment.items() if key != "commitment_sha256"}
    canonical = (json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
    require(commitment.get("schema_version") == "chemvector-joint-commitment-1.0"
            and commitment.get("dataset") == "fresh_eval_12k"
            and commitment.get("sealed_access") == "locked_until_checkpoint_gate"
            and commitment.get("split_counts") == {"fresh_dev": 8000, "sealed_test": 4000}
            and commitment.get("rdkit_version") == rdBase.rdkitVersion
            and commitment.get("commitment_sha256") == digest(canonical), "Locked aggregate commitment mismatch")


def prepare(preflight_root, output):
    root, output = Path(preflight_root).absolute(), Path(output).absolute()
    require(output.resolve() == output and not output.exists(), "Use a fresh nonsymlink request output")
    inputs = {}

    def bound_read(path):
        path = Path(path).absolute()
        payload = read(path)
        inputs[str(path)] = digest(payload)
        return payload

    report = decode(bound_read(root / "report.json"))
    bindings = report["input_sha256"]
    preflight.verify_bindings(bindings)
    require(bindings.get(str(Path(preflight.__file__).absolute())) == digest(read(Path(preflight.__file__).absolute())),
            "Preflight implementation is not bound")
    handoff_path = Path(report["handoff_path"])
    handoff = decode(bound_read(handoff_path))
    require(inputs[str(handoff_path)] == bindings.get(str(handoff_path)), "Handoff is not bound by preflight")
    require(handoff.get("training_authorized") is False and handoff.get("external_overlap_status") == "unchecked"
            and handoff.get("status") == "proposed_internal_quality_handoff_only", "Unexpected handoff scope")
    journal = bound_read(root / "samples.jsonl")
    validate_preflight(report, handoff, journal)
    data_root = Path(handoff["dataset"])
    require(not output.is_relative_to(data_root) and not output.is_relative_to(root), "Request must be outside source data/preflight")
    molecules_path = data_root / "molecules.json"
    molecules = decode(bound_read(molecules_path))
    require(inputs[str(molecules_path)] == bindings.get(str(molecules_path)), "Molecule registry not bound by preflight")
    with rdBase.BlockLogs():
        candidates = assemble_candidates(molecules, handoff["retained_candidates"])
    commitment = decode(bound_read(COMMITMENT))  # Aggregate only, no listed path is opened.
    validate_commitment(commitment)
    identity_path = Path(admission.__file__).absolute()
    implementation_path = Path(__file__).absolute()
    bound_read(identity_path)
    bound_read(implementation_path)
    for path, expected in bindings.items():
        require(path not in inputs or inputs[path] == expected, "Conflicting input bindings")
        inputs[path] = expected
    images = sum(row["image_count"] for row in candidates)
    request = dict(schema_version="chemvector-identity-review-request-0.1", training_authorized=False,
        authorization_status="not_granted_for_this_batch", external_overlap_status="pending_independent_review",
        rdkit_version=rdBase.rdkitVersion, identity_implementation_sha256=inputs[str(identity_path)],
        request_implementation_sha256=inputs[str(implementation_path)], input_sha256=inputs,
        candidate_identities_sha256=digest(preflight.encoded(candidates)),
        evaluation_commitment_sha256=commitment["commitment_sha256"],
        candidate_molecules=len(candidates), candidate_images=images,
        candidate_molecules_by_source={"curriculum": len(candidates)}, candidate_images_by_source={"curriculum": images},
        requested_checks=["full_inchikey", "connectivity_prefix", "stereo_isotope_independent_group_key"],
        return_contract=["Counts and candidate-side exclusion IDs only; no private IDs, structures, images or logs.",
                         "Exactly one invocation on these fixed bytes after explicit authorization; no adaptive replacements or retries."],
        public_review="not_performed_no_public_membership_index_read", private_identity_review_executed=False,
        training_performed=False, fresh_eval_sample_data_read=False, sealed_test_sample_data_read=False,
        payloads_reverified=False, payload_integrity_evidence="hash-bound completed selected-target preflight; not rerun here",
        selected_target_preflight_sha256=inputs[str(root / "report.json")],
        remaining_blockers=["New explicit one-time identity-review authorization for this exact candidate batch.",
                            "Independent public/private identity review and grouped data release.",
                            "Predicted candidate reachability, loss/parameter selection and a new bounded training protocol."])
    preflight.verify_bindings(inputs)
    output.mkdir(parents=True, exist_ok=False)
    preflight.write_new(output / "candidate-identities.json", candidates)
    preflight.write_new(output / "request.json", request)  # Completion marker written last.
    return request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.preflight, args.output)
    print(json.dumps({key: result[key] for key in ("candidate_molecules", "candidate_images", "authorization_status", "training_authorized")}))


if __name__ == "__main__":
    main()
