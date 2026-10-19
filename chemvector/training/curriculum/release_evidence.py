"""Consume fixed prior evidence to plan a mixed release; never private queries.

Approval remains a separate, exact-plan review. Historical inputs are read-only.
No scanning of external corpora, renderer reruns or model-dependent selection.
"""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from rdkit import rdBase

from .admission import identities
from .private_review import WORKER, validate_receipt
from .protocol import validate_protocol
from .release import SCHEMA, MIXED_FORMAT, validate_manifest


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def read(path):
    path = Path(path)
    if (not path.is_absolute() or ".." in path.parts or "sealed_test" in path.parts
            or any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file()):
        raise ValueError("Expected explicit non-symlinked, non-sealed release input")
    return path.read_bytes()


def digest(path):
    return sha256(read(path)).hexdigest()


def inputs(spec):
    required = {"request", "request_candidates", "public_report", "remaining", "private_receipt",
                "validation_candidates", "curriculum_split", "source_rights",
                "curriculum_targets", "full_molecule_targets"}
    if set(spec.get("inputs", {})) != required:
        raise ValueError("Incomplete release evidence inputs")
    raw = {}
    for name, binding in spec["inputs"].items():
        payload = read(binding["path"])
        if sha256(payload).hexdigest() != binding["sha256"]:
            raise ValueError(f"Release evidence hash mismatch: {name}")
        raw[name] = payload
    return raw, {name: json.loads(payload) for name, payload in raw.items()}


def verify_reviews(raw, documents):
    request, public = documents["request"], documents["public_report"]
    original, remaining = documents["request_candidates"], documents["remaining"]
    if (request.get("schema_version") != "chemvector-identity-review-request-0.1"
            or request.get("candidate_identities_sha256") != sha256(raw["request_candidates"]).hexdigest()
            or request.get("identity_implementation_sha256") != digest(Path(__file__).with_name("admission.py"))
            or request.get("rdkit_version") != rdBase.rdkitVersion
            or request.get("candidate_molecules") != len(original)
            or request.get("candidate_images") != sum(r["image_count"] for r in original)):
        raise ValueError("Original identity request binding mismatch")
    receipt = validate_receipt(documents["private_receipt"], raw["request"], raw["request_candidates"], digest(WORKER))
    if receipt["status"] not in {"checked", "checked_conservative"}:
        raise ValueError("Incomplete private identity receipt")
    expected = {"schema_version": "chemvector-public-candidate-exclusion-0.1",
        "status": "screened_with_conservative_exclusions", "training_authorized": False,
        "request_sha256": sha256(raw["request"]).hexdigest(),
        "candidate_identities_sha256": sha256(raw["request_candidates"]).hexdigest(),
        "remaining_candidates_sha256": sha256(raw["remaining"]).hexdigest(),
        "remaining_molecules": len(remaining), "remaining_images": sum(r["image_count"] for r in remaining),
        "remaining_ambiguous_pairs": 0, "rdkit_version": rdBase.rdkitVersion}
    if any(public.get(k) != v for k, v in expected.items()):
        raise ValueError("Public candidate review binding mismatch")
    for name, expected_hash in public["implementation_sha256"].items():
        if name not in {"public_review.py", "ctab_inventory.py", "admission.py"} or digest(Path(__file__).with_name(name)) != expected_hash:
            raise ValueError("Public review implementation changed")
    inherited = public["private_receipt_inheritance"]
    if (inherited["receipt_sha256"] != sha256(raw["private_receipt"]).hexdigest()
            or inherited["evaluation_commitment_sha256"] != receipt["evaluation_commitment_sha256"]):
        raise ValueError("Private receipt inheritance mismatch")
    by_ref = {(r["source"], r["molecule_id"]): r for r in original}
    exclusions = {(r["source"], r["molecule_id"]) for r in receipt["excluded_candidates"]}
    seen = set()
    for row in remaining:
        ref = row["source"], row["molecule_id"]
        if ref in seen or ref in exclusions or by_ref.get(ref) != row:
            raise ValueError("Remaining candidates are not an exact cleared subset")
        seen.add(ref)
        ids = identities(row["canonical_smiles"])
        if any(row[k] != ids[k] for k in ("inchikey", "connectivity_prefix", "group_key")):
            raise ValueError("Remaining candidate identity mismatch")
    return receipt


def unify_splits(rows):
    """Union all identity/core/connectivity constraints, preserving holdouts.

    A component containing old-train replay AND a proposed holdout is rejected:
    neither silently sacrifice the holdout nor rename old training as unseen.
    """
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        ids = identities(row["canonical_smiles"])
        for field, key in (("id", row["molecule_id"]), ("core", ids["group_key"]),
                           ("connectivity", ids["connectivity_prefix"])):
            if (field, key) in seen:
                parents[find(i)] = find(seen[(field, key)])
            else:
                seen[field, key] = i
    holdout = {find(i) for i, row in enumerate(rows) if row["split"] == "validation"}
    if any(find(i) in holdout and row["source"] == "full_molecule" for i, row in enumerate(rows)):
        raise ValueError("Old-train replay conflicts with an existing proposed holdout")
    changes = []
    for i, row in enumerate(rows):
        if find(i) in holdout and row["split"] != "validation":
            changes.append({"source": row["source"], "sample_key": row["sample_key"],
                "molecule_id": row["molecule_id"], "from": row["split"], "to": "validation",
                "reason": "preserve holdout across identity/connectivity/core union"})
            row["split"] = "validation"
    return changes


def build_plan(spec):
    if spec.get("schema_version") != "chemvector-release-spec-0.1" or set(spec.get("roots", {})) != {"curriculum", "full_molecule"}:
        raise ValueError("Expected fixed mixed release specification")
    validate_protocol(spec.get("training_protocol"))
    if spec.get("training_protocol") is None:
        raise ValueError("Real promotion requires a fixed training protocol")
    if not spec.get("initial_provenance", {}).get("checkpoints"):
        raise ValueError("Real promotion requires pinned initial checkpoints and inference settings")
    if spec.get("split_policy") != "preserve_existing_validation_union_identity_connectivity_and_core":
        raise ValueError("Explicit global holdout-preserving split policy required")
    raw, docs = inputs(spec)
    receipt = verify_reviews(raw, docs)
    rights = docs["source_rights"]
    if (rights.get("schema_version") != "chemvector-source-use-review-0.1"
            or rights.get("use") != "local_research_only" or rights.get("redistribution_authorized") is not False
            or rights.get("remaining_candidates_sha256") != sha256(raw["remaining"]).hexdigest()
            or set(rights.get("sources", {})) != {"hand_authored", "chembl", "coconut", "dejavu"}
            or any(not r.get("basis") or not r.get("attribution") for r in rights["sources"].values())):
        raise ValueError("Missing bounded source-use review")
    split = {r["sample_key"]: r for r in docs["curriculum_split"]}
    if len(split) != len(docs["curriculum_split"]):
        raise ValueError("Duplicate curriculum split proposal")
    rows, artifacts, original_sources = [], {}, {}
    for domain, root_name in spec["roots"].items():
        root = Path(root_name)
        source = {}
        for name in ("summary.json", "molecules.json", "samples.jsonl", "sample-sha256.json"):
            payload = read(root / name)
            if docs["request"]["input_sha256"].get(str(root / name)) != sha256(payload).hexdigest():
                raise ValueError("Candidate source differs from fixed identity request")
            source[name] = payload
        original_sources[domain] = {name: sha256(value).hexdigest() for name, value in source.items()}
        inventory = json.loads(source["sample-sha256.json"])
        molecules = {r["molecule_id"]: r for r in json.loads(source["molecules.json"])}
        source_rows = {r["sample_key"]: r for r in map(json.loads, source["samples.jsonl"].splitlines())}
        target = docs[domain + "_targets"]
        if (target.get("schema_version") != "chemvector-curriculum-cpu-preflight-0.1"
                or target.get("failed_images") != 0
                or target.get("checked_images") != len(source_rows)
                or target.get("target_compatible_images") != len(source_rows)
                or any(target["input_sha256"].get(name) != original_sources[domain][name]
                       for name in ("summary.json", "samples.jsonl", "sample-sha256.json"))):
            raise ValueError("Model target evidence does not cover the fixed source")
        for candidate in [r for r in docs["remaining"] if r["source"] == domain]:
            molecule = molecules[candidate["molecule_id"]]
            if (molecule.get("training_authorized") is not False or molecule.get("split") != "quarantine"
                    or molecule["canonical_smiles"] != candidate["canonical_smiles"]):
                raise ValueError("Expected unchanged quarantine molecule")
            if domain == "full_molecule":
                origin = molecule["source_provenance"]["source_molecule"]
                if (origin["split"] != "train" or origin["source"] not in {"chembl", "coconut"}
                        or origin["molecule_id"] != candidate["source_molecule_id"]
                        or candidate["intended_use"] != "old_train_identity_rerender_not_unseen_validation"):
                    raise ValueError("Full-molecule replay must come from old train")
            for key in candidate["sample_keys"]:
                if source_rows[key]["molecule_id"] != candidate["molecule_id"]:
                    raise ValueError("Candidate image identity mismatch")
                planned_split = "train"
                if domain == "curriculum":
                    proposed = split[key]
                    if (proposed["molecule_id"] != candidate["molecule_id"] or proposed["group_key"] != candidate["group_key"]
                            or proposed["proposed_split"] not in {"candidate_train", "candidate_validation"}):
                        raise ValueError("Curriculum grouped proposal mismatch")
                    planned_split = proposed["proposed_split"].removeprefix("candidate_")
                row = {k: candidate[k] for k in ("molecule_id", "canonical_smiles", "group_key")}
                row.update(source=domain, domain=domain, split=planned_split, sample_key=key,
                           meta_sha256=inventory["samples/" + key + ".meta.json"])
                rows.append(row)
                prefix = "samples/" + key + "."
                artifacts[domain + ":" + key] = {name: value for name, value in inventory.items() if name.startswith(prefix)}
    adjustments = unify_splits(rows)
    fresh = docs["validation_candidates"]
    if fresh.get("schema_version") != "chemvector-validation-candidates-0.1" or fresh.get("training_authorized") is not False:
        raise ValueError("Expected fixed development-validation candidates")
    if set(fresh["sources"]) & set(spec["roots"]):
        raise ValueError("Conflicting source names")
    manifest = {"schema_version": SCHEMA, "target_format": MIXED_FORMAT,
        "sources": {**spec["roots"], **fresh["sources"]}, "validation_sources": fresh["validation_sources"],
        "samples": rows + fresh["samples"], "seed": spec["seed"], "train_cycle": spec["train_cycle"],
        "training_protocol": spec["training_protocol"], "initial_provenance": spec["initial_provenance"]}
    validate_manifest(manifest)  # Includes globally recomputed molecular groups.
    for binding in fresh["validation_sources"].values():
        access = json.loads(read(binding["access_plan"]))
        if access["commitment"]["commitment_sha256"] != receipt["evaluation_commitment_sha256"]:
            raise ValueError("Development validation and private receipt commitment differ")
    counts = Counter((r["split"], r["domain"]) for r in manifest["samples"])
    return {"schema_version": "chemvector-promotion-plan-0.1", "training_authorized": False,
        "source_spec_sha256": sha256(encoded(spec)).hexdigest(), "manifest_template": manifest,
        "source_artifacts": artifacts, "original_source_manifests": original_sources,
        "split_adjustments": adjustments, "split_policy": spec["split_policy"],
        "counts": {s + ":" + d: n for (s, d), n in sorted(counts.items())},
        "evidence": spec["inputs"], "use": "local_research_only",
        "limitations": ["Development selection only; sealed remains closed and is not claimed group-independent.",
            "The public proof covers this fixed candidate subset, not repaired global public identities.",
            "Re-render pilot only; 88 original planned full-molecule variants failed rendering.",
            "No architecture/learning-rate search, external release, deployment or automatic resume."]}
