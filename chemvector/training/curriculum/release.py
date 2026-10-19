"""Read-only release contracts. This module cannot promote quarantine data."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re

from .admission import identities, require_training_admission
from .protocol import validate_protocol

SCHEMA = "chemvector-continuation-release-0.1"
MIXED_FORMAT = "curriculum-with-original-validation"
CHECKS = {"source_rights", "public_benchmark_identity", "fresh_eval_identity", "group_split", "model_targets"}


def verify_release(manifest_path, admission_path):
    """Bind authorization to exact bytes; no images or GPU are opened here."""
    admission_bytes = Path(admission_path).read_bytes()
    admission = json.loads(admission_bytes)
    require_training_admission(admission)  # Before even opening the data manifest.
    payload = Path(manifest_path).read_bytes()
    digest = sha256(payload).hexdigest()
    if admission.get("approved_manifest_sha256") != digest:
        raise ValueError("Admission is not bound to this exact release manifest")
    checks = admission.get("release_checks", {})
    if any(checks.get(name) != "checked" for name in CHECKS):
        raise ValueError("Release lacks required source/identity/target checks")
    manifest = json.loads(payload)
    validate_manifest(manifest)
    if admission.get("checked_images") != len(manifest["samples"]):
        raise ValueError("Admission/release image count mismatch")
    return manifest, admission, {"manifest_sha256": digest, "admission_sha256": sha256(admission_bytes).hexdigest()}


def validate_manifest(manifest):
    """Structural checks only; this function does not grant training admission."""
    if (manifest.get("schema_version") != SCHEMA
            or manifest.get("target_format") not in {"curriculum-same-svg", MIXED_FORMAT}):
        raise ValueError("Unsupported release schema or target format")
    sources = manifest.get("sources", {})
    if not sources or any(not Path(root).is_absolute() for root in sources.values()):
        raise ValueError("Release source roots must be explicit absolute paths")
    rows = manifest.get("samples", [])
    if not rows:
        raise ValueError("Empty release")
    seen, group_split, identity_split, domain_counts = set(), {}, {}, Counter()
    for row in rows:
        key = row["sample_key"]
        if (row["source"] not in sources or not re.fullmatch(r"[A-Za-z0-9_-]+", key)
                or (row["source"], key) in seen or row["split"] not in {"train", "validation"}
                or row["domain"] not in {"curriculum", "full_molecule"}
                or not re.fullmatch(r"[0-9a-f]{64}", row["meta_sha256"])):
            raise ValueError("Invalid/duplicate release sample")
        seen.add((row["source"], key))
        ids = identities(row["canonical_smiles"])
        if row["group_key"] != ids["group_key"]:
            raise ValueError("Release group does not match molecular core/scaffold")
        for mapping, identity in ((group_split, ids["group_key"]), (identity_split, ids["connectivity_prefix"]),
                                  (identity_split, "id:" + row["molecule_id"])):
            previous = mapping.setdefault(identity, row["split"])
            if previous != row["split"]:
                raise ValueError("Release train/validation molecular groups overlap")
        domain_counts[(row["split"], row["domain"])] += 1
    if any(domain_counts[("train", domain)] == 0 for domain in ("curriculum", "full_molecule")):
        raise ValueError("Training must retain both curriculum and full-molecule sources")
    if not domain_counts[("validation", "full_molecule")]:
        raise ValueError("Full-molecule validation records are required")
    cycle = manifest.get("train_cycle", [])
    if not cycle or set(cycle) != {"curriculum", "full_molecule"} or len(cycle) > 100:
        raise ValueError("Explicit reviewed training mixture cycle is required")
    if type(manifest.get("seed")) is not int:
        raise ValueError("Release requires an explicit integer seed")
    validate_protocol(manifest.get("training_protocol"))
    _validation_sources(manifest)  # Access plans/manifests only; no tar/pixel reads.


def _validation_sources(manifest):
    from .fresh_dev import FreshDevSource
    bindings = manifest.get("validation_sources", {})
    mixed = manifest.get("target_format") == MIXED_FORMAT
    if bool(bindings) != mixed or not isinstance(bindings, dict):
        raise ValueError("Original validation sources require the explicit mixed target format")
    result = {}
    for name, binding in bindings.items():
        if name not in manifest["sources"]:
            raise ValueError("Unknown original validation source")
        rows = [r for r in manifest["samples"] if r["source"] == name]
        if not rows or any(r["split"] != "validation" or r["domain"] != "full_molecule" for r in rows):
            raise ValueError("Original fresh-dev source is validation-only, never training")
        source = FreshDevSource(manifest["sources"][name], binding)
        if {r["sample_key"] for r in rows} != source.rows.keys() or len(rows) != len(source.rows):
            raise ValueError("Release must retain the entire fixed validation selection")
        for row in rows:
            source.check_row(row)
        result[name] = source
    return result


def training_order(manifest):
    """Deterministic infinite stream; step ceiling is owned by the trainer."""
    randomizer = random.Random(manifest["seed"])
    pools = {domain: [r for r in manifest["samples"] if r["split"] == "train" and r["domain"] == domain]
             for domain in set(manifest["train_cycle"])}
    if any(not pool for pool in pools.values()):
        raise ValueError("Empty mixture domain")
    queues = {domain: [] for domain in pools}
    while True:
        for domain in manifest["train_cycle"]:
            if not queues[domain]:
                queues[domain] = list(pools[domain])
                randomizer.shuffle(queues[domain])
            yield queues[domain].pop()


def load_record(manifest, row):
    """One-shot compatible API; a job uses RecordLoader to reuse tar indexes."""
    return RecordLoader(manifest)(manifest, row)


class RecordLoader:
    """Job-local verified sources, without decoded-image or target caches.

    Original tar hash/index is checked once per job; its file signature is
    rechecked on every access and PNG/SDF bytes are always rehashed.
    """
    def __init__(self, manifest):
        self.manifest = manifest
        self.originals = _validation_sources(manifest)
        self.rows = {(r["source"], r["sample_key"]): dict(r) for r in manifest["samples"]}

    def __call__(self, manifest, row):
        if manifest != self.manifest or self.rows.get((row["source"], row["sample_key"])) != row:
            raise ValueError("Record is not in this fixed release")
        if row["source"] in self.originals:
            return self.originals[row["source"]].load(row)
        return _load_same_svg_record(manifest, row)


def _load_same_svg_record(manifest, row):
    from .data import load_sample
    root = Path(manifest["sources"][row["source"]])
    meta_path = root / "samples" / (row["sample_key"] + ".meta.json")
    if meta_path.is_symlink() or sha256(meta_path.read_bytes()).hexdigest() != row["meta_sha256"]:
        raise ValueError("Released sample metadata hash mismatch")
    sample = load_sample(root, row["sample_key"], purpose="training" if row["split"] == "train" else "diagnostic")
    if (sample["meta"]["split"] != row["split"] or sample["meta"]["external_overlap_status"] != "checked"
            or sample["meta"]["molecule_id"] != row["molecule_id"]
            or sample["graph"]["canonical_smiles"] != row["canonical_smiles"]):
        raise ValueError("Released sample identity/split does not match its manifest")
    return sample
