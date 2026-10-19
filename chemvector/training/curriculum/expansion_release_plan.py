"""Plan an additive release from a trusted old release and exact new evidence.

Keep every old validation row and source binding. New rows are only compatible,
unique, cleared train candidates. Never request hidden identities or grant admission.
"""

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from .data import load_sample
from .protocol import validate_protocol
from .publish import SUFFIXES
from .release import RecordLoader, validate_manifest, verify_release
from .release_evidence import digest, encoded, read, verify_reviews

INPUTS = {"previous_manifest", "previous_admission", "request", "request_candidates", "private_receipt",
          "public_report", "remaining", "inventory", "inventory_candidates", "targets", "target_samples", "source_rights"}
MINIMUM_IMAGES = 10000


def choose_additions(old, new):
    """Preserve old rows; deduplicate new pixels and reject holdout/conflict risks."""
    by_pixels = defaultdict(list)
    for row in old:
        by_pixels[row["pixel_sha256"]].append(row)
    for rows in by_pixels.values():
        if len({r["split"] for r in rows}) > 1:
            raise ValueError("Existing train/validation pixel overlap requires separate review")
    if len({r["sample_key"] for r in new}) != len(new) or len({r["pixel_sha256"] for r in new}) != len(new):
        raise ValueError("New inventory is not sample/pixel unique")
    heldout_molecules = {r["molecule_id"] for r in new
                        if any(x["split"] == "validation" for x in by_pixels[r["pixel_sha256"]])}
    kept, duplicates, heldout = [], [], []
    for row in new:
        matches = by_pixels[row["pixel_sha256"]]
        if row["molecule_id"] in heldout_molecules:
            heldout.append(row["sample_key"])
        elif matches:
            if any(x["canonical_smiles"] != row["canonical_smiles"] for x in matches):
                raise ValueError("Same pixels have conflicting chemistry across releases")
            duplicates.append(row["sample_key"])
        else:
            kept.append(row)
    return kept, dict(new_duplicate_keys=sorted(duplicates), heldout_pixel_excluded_keys=sorted(heldout),
        heldout_pixel_excluded_molecules=sorted(heldout_molecules),
        old_train_images=len([r for r in old if r["split"] == "train"]),
        old_train_unique_pixels=len({r["pixel_sha256"] for r in old if r["split"] == "train"}))


def evidence(spec):
    if (spec.get("schema_version") != "chemvector-expanded-release-spec-0.1"
            or set(spec.get("inputs", {})) != INPUTS or spec.get("new_source") != "full_molecule_expansion"
            or spec.get("minimum_unique_train_images") != MINIMUM_IMAGES):
        raise ValueError("Incomplete expanded release specification")
    validate_protocol(spec.get("training_protocol"))
    if (spec.get("training_protocol", {}).get("update_recipe") != "freeze-role-v1"
            or type(spec.get("seed")) is not int):
        raise ValueError("Expanded release needs the explicit frozen-role protocol and seed")
    raw = {}
    for name, binding in spec["inputs"].items():
        value = read(binding["path"])
        if sha256(value).hexdigest() != binding["sha256"]:
            raise ValueError(f"Expanded release evidence hash mismatch: {name}")
        raw[name] = value
    docs = {k: json.loads(v) for k, v in raw.items()}
    previous, _, bindings = verify_release(spec["inputs"]["previous_manifest"]["path"], spec["inputs"]["previous_admission"]["path"])
    if spec["new_source"] in previous["sources"]:
        raise ValueError("New source name already exists in previous release")
    verify_reviews(raw, docs)  # Pure verification of the already-issued private receipt.
    rights = docs["source_rights"]
    if (rights.get("schema_version") != "chemvector-source-use-review-0.1"
            or rights.get("use") != "local_research_only" or rights.get("redistribution_authorized") is not False
            or rights.get("remaining_candidates_sha256") != sha256(raw["remaining"]).hexdigest()
            or rights.get("inherited_release_admission_sha256") != bindings["admission_sha256"]
            or set(rights.get("sources", {})) != {"hand_authored", "chembl", "coconut", "dejavu"}
            or any(not r.get("basis") or not r.get("attribution") for r in rights["sources"].values())):
        raise ValueError("Missing exact expanded source-use review")
    return raw, docs, previous, bindings


def build_plan(spec):
    raw, docs, previous, bindings = evidence(spec)
    root = Path(spec["data_root"])
    if not root.is_absolute():
        raise ValueError("Expansion source must be absolute")
    source = {name: read(root / name) for name in ("molecules.json", "samples.jsonl", "sample-sha256.json", "summary.json")}
    hashes = {name: sha256(value).hexdigest() for name, value in source.items()}
    if any(docs["request"]["input_sha256"].get(str(root / name)) != digest_ for name, digest_ in hashes.items()):
        raise ValueError("Expanded source changed since fixed identity request")
    molecules = {r["molecule_id"]: r for r in json.loads(source["molecules.json"])}
    rows = [json.loads(line) for line in source["samples.jsonl"].splitlines()]
    by_key = {r["sample_key"]: r for r in rows}
    corpus = json.loads(source["sample-sha256.json"])
    target = docs["targets"]
    target_rows = docs["target_samples"]
    targets = {r["sample_key"]: r for r in target_rows}
    unique = {r["sample_key"]: r for r in docs["inventory_candidates"]}
    inventory = docs["inventory"]
    if (target.get("schema_version") != "chemvector-curriculum-cpu-preflight-0.1"
            or target.get("training_authorized") is not False or target["checked_images"] != len(rows)
            or len(targets) != len(target_rows) or set(targets) != set(by_key) or len(by_key) != len(rows)
            or target["failed_images"] != sum(not r["compatible"] for r in target_rows)
            or target["target_compatible_images"] != sum(r["compatible"] for r in target_rows)
            or any(target["input_sha256"].get(name) != hashes[name] for name in ("summary.json", "samples.jsonl", "sample-sha256.json"))
            or inventory.get("training_authorized") is not False
            or len(unique) != len(docs["inventory_candidates"]) or len(unique) != inventory["compatible_unique_candidate_images"]):
        raise ValueError("Incomplete or inconsistent explicit target/unique evidence")
    for name in ("targets", "target_samples"):
        binding = spec["inputs"][name]
        if inventory["input_sha256"].get(binding["path"]) != binding["sha256"]:
            raise ValueError("Pixel inventory is not bound to this complete target audit")
    new, artifacts, used = [], {}, set()
    for candidate in docs["remaining"]:
        mid = candidate["molecule_id"]
        molecule = molecules[mid]
        origin = molecule["source_provenance"]["source_molecule"]
        if (candidate["source"] != "full_molecule" or molecule["canonical_smiles"] != candidate["canonical_smiles"]
                or origin["split"] != "train" or origin["molecule_id"] != candidate["source_molecule_id"]
                or origin["source"] not in {"chembl", "coconut"}):
            raise ValueError("Expanded candidate is not original train-only chemistry")
        for key in candidate["sample_keys"]:
            if (key in used or key not in unique or not unique[key]["target_compatible"] or not targets[key]["compatible"]
                    or unique[key]["molecule_id"] != mid or targets[key]["molecule_id"] != mid
                    or by_key[key]["molecule_id"] != mid or unique[key]["canonical_smiles"] != candidate["canonical_smiles"]):
                raise ValueError("Uncleared, duplicated or incompatible expanded sample")
            used.add(key)
            meta_bytes = read(root / "samples" / (key + ".meta.json"))
            meta_hash = sha256(meta_bytes).hexdigest()
            if corpus.get("samples/" + key + ".meta.json") != meta_hash:
                raise ValueError("Expanded metadata changed since target audit")
            sample = load_sample(root, key, purpose="diagnostic")
            meta = sample["meta"]
            if (meta["training_authorized"] is not False or meta["split"] != "quarantine"
                    or sample["graph"]["canonical_smiles"] != candidate["canonical_smiles"]
                    or sha256(sample["image"].tobytes()).hexdigest() != unique[key]["pixel_sha256"]):
                raise ValueError("Expanded pixels, chemistry or quarantine state changed")
            payload_hashes = {**meta["sha256"], ".meta.json": meta_hash}
            if set(payload_hashes) != SUFFIXES or any(corpus.get("samples/" + key + s) != h for s, h in payload_hashes.items()):
                raise ValueError("Expanded artifact inventory mismatch")
            row = {k: candidate[k] for k in ("molecule_id", "canonical_smiles", "group_key")}
            row.update(source=spec["new_source"], domain="full_molecule", split="train", sample_key=key,
                       meta_sha256=meta_hash, pixel_sha256=unique[key]["pixel_sha256"])
            new.append(row)
            artifacts[key] = payload_hashes
        if len(new) // 1000 > (len(new) - len(candidate["sample_keys"])) // 1000:
            print(f"Expanded candidate bundles verified: {len(new)}", flush=True)
    old, loader = [], RecordLoader(previous)
    for number, row in enumerate(previous["samples"], 1):
        sample = loader(previous, row)
        old.append({**row, "pixel_sha256": sha256(sample["image"].tobytes()).hexdigest()})
        if number % 500 == 0:
            print(f"Existing release pixels verified: {number}/{len(previous['samples'])}", flush=True)
    additions, deduplication = choose_additions(old, new)
    unique_train = deduplication["old_train_unique_pixels"] + len(additions)
    if unique_train < MINIMUM_IMAGES:
        raise ValueError("Merged unique train image count below fixed minimum")
    manifest = deepcopy(previous)
    manifest["sources"][spec["new_source"]] = str(root)
    manifest["samples"].extend({k: v for k, v in r.items() if k != "pixel_sha256"} for r in additions)
    manifest.update(seed=spec["seed"], training_protocol=spec["training_protocol"])
    validate_manifest(manifest)
    if ([r for r in manifest["samples"] if r["split"] == "validation"] !=
            [r for r in previous["samples"] if r["split"] == "validation"]):
        raise ValueError("Incremental release changed existing validation")
    counts = Counter(r["split"] + ":" + r["domain"] for r in manifest["samples"])
    return dict(schema_version="chemvector-expanded-promotion-plan-0.1", training_authorized=False,
        spec_sha256=sha256(encoded(spec)).hexdigest(), inherited_release_bindings=bindings,
        manifest_template=manifest, addition_rows=additions, source_artifacts={r["sample_key"]: artifacts[r["sample_key"]] for r in additions},
        source_manifest_sha256=hashes, counts=dict(counts), deduplication=deduplication,
        added_train_images=len(additions), unique_train_images=unique_train,
        validation_changed=False, new_private_query_performed=False, gpu_used=False, training_performed=False,
        limitations=["Old release stays immutable; inherited sources must remain available and hash-valid.",
            "Keep old curriculum and all 944 fixed validation rows; old in-train pixel duplicates remain explicitly counted.",
            "Development validation is not a sealed-test result and retains documented cross-evaluation group limitations.",
            "A 1200-step trial is not a full pass over the expanded training set or proof of curriculum efficacy."])
