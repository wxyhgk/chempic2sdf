"""Propose an internal, conservative quality handoff; never admit training.

Bind a passed independent audit to an unchanged completed dataset, recheck its
PNG byte and decoded-RGB groups, quarantine EVERY profile of EVERY identity in
a cross-identity collision, then propose same-identity PNG-byte deduplication.
No source artifact is deleted, rewritten, relabelled or split.  This does not
resolve stereo semantics, establish chemical stability or grant data admission.
SHA bindings detect changed inputs; they are not cryptographic signer identity.
"""

from collections import defaultdict
from hashlib import sha256
import argparse
import io
import json
from pathlib import Path
import re
import time

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
MANIFESTS = (
    "plan.json", "summary.json", "molecules.json", "profiles.json", "samples.jsonl",
    "rejected.jsonl", "sample-sha256.json", "image-duplicates.json", "progress.jsonl",
    "visible-coverage.jsonl", "planned-coverage.json", "accepted-coverage.json",
)
AUDITOR = REPO / "tools/audit_curriculum_expansion_v02.py"
AUDITOR_DEPENDENCIES = (
    "chemvector/training/curriculum/data.py",
    "Data/sdf/text_source_renderer/renderer.py",
    "Data/sdf/chemvector_curriculum_v0_2/coverage.py",
)
KEY = re.compile(r"[A-Za-z0-9_-]+\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
SCHEMA = "chemvector-curriculum-v0.2-quality-handoff"


class HandoffError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise HandoffError(message)


def digest(data):
    return sha256(data).hexdigest()


def read(path):
    path = Path(path)
    require(path.resolve() == path.absolute() and path.is_file(),
            f"Missing or symlinked input: {path}")
    return path.read_bytes()


def decode(data):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def no_constant(value):
        raise HandoffError(f"Nonfinite JSON value: {value}")
    return json.loads(data, object_pairs_hook=unique_pairs, parse_constant=no_constant)


def lines(data):
    return [decode(line) for line in data.splitlines() if line.strip()]


def conflict_groups(groups):
    return {key: sorted(rows, key=lambda row: row["sample_key"])
            for key, rows in sorted(groups.items())
            if len({row["molecule_id"] for row in rows}) > 1}


def normalized_audit_groups(groups):
    require(isinstance(groups, dict), "Missing audit conflict groups")
    result = {}
    for key, rows in groups.items():
        require(isinstance(key, str) and SHA.fullmatch(key), "Invalid conflict group SHA")
        require(isinstance(rows, list) and len(rows) > 1, "Invalid conflict group rows")
        require(all(isinstance(row, dict) and set(row) ==
                    {"sample_key", "molecule_id", "canonical_smiles"} for row in rows),
                "Invalid conflict identity description")
        require(all(isinstance(value, str) for row in rows for value in row.values()),
                "Invalid conflict row values")
        require(len({row["sample_key"] for row in rows}) == len(rows),
                "Duplicate conflict sample key")
        require(len({row["molecule_id"] for row in rows}) > 1, "Conflict group omits identities")
        result[key] = sorted(rows, key=lambda row: row["sample_key"])
    return result


def check_inputs(dataset, audit_path):
    root, audit_path = Path(dataset).absolute(), Path(audit_path).absolute()
    require(root.is_dir() and root.resolve() == root, "Dataset must be a real nonsymlink directory")
    audit_raw = read(audit_path)
    receipt = decode(audit_raw)
    require(receipt.get("schema_version") == "chemvector-curriculum-v0.2-independent-audit",
            "Wrong audit schema")
    require(receipt.get("status") == "passed_artifact_integrity_only", "Audit did not pass")
    require(receipt.get("dataset") == str(root), "Audit belongs to another dataset/batch")
    require(receipt.get("training_authorized") is False and receipt.get("gpu_used") is False
            and receipt.get("external_identity_queries") == 0 and receipt.get("model_forwards") == 0
            and receipt.get("formal_training_admission") == "not_granted", "Audit scope mismatch")
    require(receipt.get("auditor_source_sha256") == digest(read(AUDITOR)), "Auditor source SHA mismatch")
    dependency_sha = receipt.get("auditor_dependency_sha256")
    require(isinstance(dependency_sha, dict) and set(dependency_sha) == set(AUDITOR_DEPENDENCIES),
            "Unexpected auditor dependency bindings")
    for name, expected in dependency_sha.items():
        require(expected == digest(read(REPO / name)), f"Auditor dependency SHA mismatch: {name}")
    bindings = receipt.get("manifest_sha256")
    require(isinstance(bindings, dict) and set(bindings) == set(MANIFESTS), "Incomplete manifest bindings")
    raw = {}
    for name in MANIFESTS:
        expected = bindings[name]
        require(isinstance(expected, str) and SHA.fullmatch(expected), "Invalid manifest SHA")
        raw[name] = read(root / name)
        require(digest(raw[name]) == expected, f"Manifest SHA mismatch: {name}")
    plan, summary, molecules, profiles, artifact_sha = (
        decode(raw[name]) for name in ("plan.json", "summary.json", "molecules.json", "profiles.json",
                                      "sample-sha256.json"))
    require(plan.get("schema_version") == "chemvector-curriculum-v0.2-plan"
            and summary.get("schema_version") == "chemvector-curriculum-v0.2-result", "Wrong batch schemas")
    require(summary.get("generation_complete") is True and summary.get("status") == "quarantine_preview_only",
            "Dataset generation incomplete or not quarantined")
    require(summary.get("plan_sha256") == bindings["plan.json"], "Summary plan binding mismatch")
    for owner in (plan, summary):
        require(owner.get("training_authorized") is False and owner.get("external_overlap_status") == "unchecked"
                and owner.get("external_identity_queries") == 0 and owner.get("external_corpora_read") is False
                and owner.get("model_forwards") == 0 and owner.get("optimizer_steps") == 0,
                "Batch scope mismatch")
    require(summary.get("gpu_used") is False, "Batch GPU scope mismatch")
    require(isinstance(molecules, list) and molecules and isinstance(profiles, list) and profiles
            and profiles == plan.get("profiles"), "Molecule/profile registry mismatch")
    by_id = {row["molecule_id"]: row for row in molecules}
    profile_ids = [row["profile_id"] for row in profiles]
    require(len(by_id) == len(molecules) and len(set(profile_ids)) == len(profiles), "Duplicate registry IDs")
    require(all(isinstance(key, str) and KEY.fullmatch(key) for key in [*by_id, *profile_ids]), "Unsafe registry ID")
    require(len({row["canonical_smiles"] for row in molecules}) == len(molecules), "Duplicate chemical identity")
    for identity, row in by_id.items():
        require(identity == "curr_" + digest(row["canonical_smiles"].encode())[:16], "Molecule ID mismatch")
        require(row.get("training_authorized") is False and row.get("external_overlap_status") == "unchecked"
                and row.get("split") == "quarantine", "Molecule admission scope mismatch")
    samples, rejected = lines(raw["samples.jsonl"]), lines(raw["rejected.jsonl"])
    pairs = [(row["molecule_id"], row["profile_id"]) for row in [*samples, *rejected]]
    expected_pairs = {(identity, profile) for identity in by_id for profile in profile_ids}
    require(len(pairs) == len(set(pairs)) and set(pairs) == expected_pairs, "Incomplete profile denominator")
    require(len({row["sample_key"] for row in samples}) == len(samples), "Duplicate sample key")
    byte_groups, rgb_groups, png_sha, rgb_sha = defaultdict(list), defaultdict(list), {}, {}
    for row in samples:
        key, identity = row["sample_key"], row["molecule_id"]
        require(key == f"{identity}__{row['profile_id']}" and KEY.fullmatch(key), "Unsafe/mismatched sample key")
        relative = f"samples/{key}.png"
        payload = read(root / relative)
        observed = digest(payload)
        require(artifact_sha.get(relative) == observed, f"PNG SHA mismatch: {key}")
        png_sha[key] = observed
        with Image.open(io.BytesIO(payload)) as image:
            require(image.format == "PNG" and image.mode == "RGB" and image.size == (768, 768),
                    f"Unexpected PNG format: {key}")
            image.load()
            rgb_sha[key] = digest(image.tobytes())
        description = {"sample_key": key, "molecule_id": identity,
                       "canonical_smiles": by_id[identity]["canonical_smiles"]}
        byte_groups[observed].append(description)
        rgb_groups[rgb_sha[key]].append(description)
    computed = {
        "byte_identical_cross_identity_conflicts": conflict_groups(byte_groups),
        "decoded_rgb_cross_identity_conflicts": conflict_groups(rgb_groups),
    }
    for field, actual in computed.items():
        require(normalized_audit_groups(receipt.get(field)) == actual,
                f"Audit conflict groups differ from recomputed PNGs: {field}")
    require(receipt.get("blocking_cross_identity_pixel_conflicts") is bool(any(computed.values())),
            "Audit blocking-conflict flag mismatch")
    require(receipt.get("decoded_rgb_distinct_images") == len(rgb_groups), "Audit RGB denominator mismatch")
    checked = receipt.get("checked", {})
    counters = {"unique_molecules": len(molecules), "planned_images": len(pairs),
                "accepted_images": len(samples), "rejected_images": len(rejected),
                "distinct_png_files": len(byte_groups), "same_png_extra_rows": len(samples) - len(byte_groups),
                "cross_identity_same_png_groups": len(computed["byte_identical_cross_identity_conflicts"]),
                "cross_identity_same_png_rows": sum(map(len, computed["byte_identical_cross_identity_conflicts"].values()))}
    for name, expected in counters.items():
        require(checked.get(name) == summary.get(name) == expected, f"Audit/batch count mismatch: {name}")
    require(plan.get("selected_molecules") == len(molecules) and plan.get("planned_images") == len(pairs),
            "Plan denominator mismatch")
    # Recheck manifests, audit, and code bindings before producing a proposal.
    for name, expected in bindings.items():
        require(digest(read(root / name)) == expected, f"Manifest changed during handoff: {name}")
    for key, expected in png_sha.items():
        require(digest(read(root / "samples" / f"{key}.png")) == expected,
                f"PNG changed during handoff: {key}")
    require(read(audit_path) == audit_raw, "Audit changed during handoff")
    require(digest(read(AUDITOR)) == receipt["auditor_source_sha256"], "Auditor changed during handoff")
    for name, expected in dependency_sha.items():
        require(digest(read(REPO / name)) == expected, f"Auditor dependency changed during handoff: {name}")
    return root, receipt, digest(audit_raw), by_id, profile_ids, samples, rejected, png_sha, rgb_sha, computed


def prepare_proposal(dataset, audit_path):
    root, audit, audit_sha, by_id, profiles, samples, rejected, png_sha, rgb_sha, conflicts = check_inputs(
        dataset, audit_path)
    reasons = defaultdict(list)
    direct_conflict_keys = set()
    for kind, groups in conflicts.items():
        for group_sha, members in groups.items():
            direct_conflict_keys.update(row["sample_key"] for row in members)
            for identity in sorted({row["molecule_id"] for row in members}):
                reasons[identity].append({"kind": kind, "group_sha256": group_sha})
    quarantined, safe_groups, kept, duplicate_rows, duplicate_groups = [], defaultdict(list), [], [], []
    sample_by_id = defaultdict(list)
    for row in sorted(samples, key=lambda item: item["sample_key"]):
        key, identity = row["sample_key"], row["molecule_id"]
        described = {"sample_key": key, "molecule_id": identity, "profile_id": row["profile_id"],
                     "png_sha256": png_sha[key], "decoded_rgb_sha256": rgb_sha[key]}
        sample_by_id[identity].append(key)
        if identity in reasons:
            quarantined.append({**described, "reason": "identity_has_cross_identity_png_or_rgb_conflict",
                                "directly_in_conflict_group": key in direct_conflict_keys})
        else:
            safe_groups[png_sha[key]].append(described)
    for group_sha, members in sorted(safe_groups.items()):
        require(len({row["molecule_id"] for row in members}) == 1, "Unquarantined cross-identity PNG group")
        members.sort(key=lambda row: row["sample_key"])
        kept.append(members[0])
        if len(members) > 1:
            extras = [{**row, "retain_sample_key": members[0]["sample_key"],
                       "reason": "same_identity_byte_identical_png_extra_row"} for row in members[1:]]
            duplicate_rows.extend(extras)
            duplicate_groups.append({"png_sha256": group_sha, "molecule_id": members[0]["molecule_id"],
                                     "retain_sample_key": members[0]["sample_key"],
                                     "exclude_sample_keys": [row["sample_key"] for row in members[1:]]})
    kept.sort(key=lambda row: row["sample_key"])
    duplicate_rows.sort(key=lambda row: row["sample_key"])
    rejected_by_id = defaultdict(list)
    for row in rejected:
        require(row.get("canonical_smiles") == by_id[row["molecule_id"]]["canonical_smiles"] and row.get("reason"),
                "Invalid generation-rejected row")
        rejected_by_id[row["molecule_id"]].append(row["profile_id"])
    identities = [{"molecule_id": identity, "canonical_smiles": by_id[identity]["canonical_smiles"],
                   "reason": "conservative_whole_identity_exclusion_for_cross_identity_pixel_conflict",
                   "trigger_groups": reasons[identity], "all_planned_profile_ids": sorted(profiles),
                   "accepted_sample_keys": sample_by_id[identity],
                   "generation_rejected_profile_ids": sorted(rejected_by_id[identity])}
                  for identity in sorted(reasons)]
    stats = {
        "source_molecules": len(by_id), "profiles_per_identity": len(profiles),
        "planned_source_samples": len(by_id) * len(profiles), "accepted_source_samples": len(samples),
        "generation_rejected_source_samples": len(rejected),
        "direct_cross_identity_conflict_samples": len(direct_conflict_keys),
        "quarantined_identities": len(identities), "quarantined_accepted_samples": len(quarantined),
        "quarantined_generation_rejected_samples": sum(len(rejected_by_id[identity]) for identity in reasons),
        "quarantined_planned_samples": len(reasons) * len(profiles),
        "retained_candidate_samples": len(kept), "retained_distinct_png_bytes": len(kept),
        "retained_distinct_decoded_rgb": len({row["decoded_rgb_sha256"] for row in kept}),
        "retained_distinct_molecules": len({row["molecule_id"] for row in kept}),
        "singleton_png_candidate_samples": sum(len(rows) == 1 for rows in safe_groups.values()),
        "same_identity_duplicate_groups": len(duplicate_groups),
        "same_identity_duplicate_extra_samples": len(duplicate_rows),
        "byte_identical_cross_identity_conflict_groups": len(conflicts["byte_identical_cross_identity_conflicts"]),
        "decoded_rgb_cross_identity_conflict_groups": len(conflicts["decoded_rgb_cross_identity_conflicts"]),
    }
    require(stats["accepted_source_samples"] == len(kept) + len(quarantined) + len(duplicate_rows),
            "Accepted sample accounting mismatch")
    require(stats["quarantined_planned_samples"] == len(quarantined) + stats["quarantined_generation_rejected_samples"],
            "Whole-identity profile accounting mismatch")
    assigned = [row["sample_key"] for rows in (kept, quarantined, duplicate_rows) for row in rows]
    require(len(set(assigned)) == len(samples) and set(assigned) == {row["sample_key"] for row in samples},
            "Candidate/quarantine/duplicate partition mismatch")
    return {
        "schema_version": SCHEMA, "status": "proposed_internal_quality_handoff_only",
        "dataset": str(root), "audit": {"path": str(Path(audit_path).absolute()), "sha256": audit_sha,
                                        "status": audit["status"]},
        "manifest_sha256": audit["manifest_sha256"], "statistics": stats,
        "retained_candidates": kept, "quarantined_identities": identities, "quarantined_samples": quarantined,
        "same_identity_duplicate_exclusion_proposals": duplicate_rows,
        "same_identity_duplicate_groups": duplicate_groups, "generation_rejected_rows": rejected,
        "recomputed_conflict_groups": conflicts,
        "policy": {"cross_identity": "exclude_every_planned_profile_of_every_involved_identity",
                   "same_identity_png_duplicates": "retain_lexicographically_smallest_sample_key",
                   "same_identity_rgb_only_duplicates": "reported_in_counts_not_removed"},
        "training_authorized": False, "external_overlap_status": "unchecked",
        "formal_training_admission": "not_granted", "external_identity_queries": 0,
        "external_corpora_read": False, "gpu_used": False, "model_forwards": 0, "optimizer_steps": 0,
        "source_files_modified": 0, "source_files_deleted": 0, "train_manifest_published": False,
        "split_created": False,
        "limitations": [
            "Conservative internal exclusion proposal only; not a resolution of stereo-label semantics.",
            "Every profile of every conflicting identity is excluded; no cross-identity winner is selected.",
            "The passed audit binds prior full-artifact checks; this handoff rechecks manifests and all PNG bytes/RGB.",
            "No file deletion, training admission, external identity clearance, new split or model inference occurred.",
        ],
    }


def prepare_handoff(dataset, audit_path, output):
    output, root = Path(output).absolute(), Path(dataset).absolute()
    require(output.suffix == ".json" and output.parent.is_dir(), "Output requires an existing parent and .json suffix")
    require(output.resolve() == output, "Symlinked output path")
    require(not output.is_relative_to(root.resolve()), "Handoff output must be outside dataset")
    require(output != Path(audit_path).resolve(), "Handoff output must not replace audit")
    source_sha = digest(read(Path(__file__)))
    started = time.monotonic()
    with output.open("x", encoding="utf-8") as stream:
        result = {"schema_version": SCHEMA, "status": "failed", "training_authorized": False,
                  "external_overlap_status": "unchecked", "handoff_source_sha256": source_sha}
        try:
            result.update(prepare_proposal(dataset, audit_path))
            require(digest(read(Path(__file__))) == source_sha, "Handoff source changed during run")
            return result
        except Exception as exc:
            result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_handoff(args.dataset, args.audit, args.output)
    print(json.dumps({"status": result["status"], "statistics": result["statistics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
