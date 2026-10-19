"""Deterministic train-only sample plan for a bounded chemical-head probe."""

from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import heapq
import json

from .pair_scale_plan import bond_bucket, digest, size_bucket, sample_id


SEED = "20260925-chemical-head-probe-v1"
SAMPLE_COUNT = 1200
CHECKPOINTS = (400, 800, 1200)
DOMAIN_PRIORITY = {"curriculum": 0, "full_molecule": 1}


def _rank(*parts: str) -> str:
    payload = json.dumps([SEED, *parts], ensure_ascii=True, separators=(",", ":"))
    return sha256(payload.encode("ascii")).hexdigest()


def _validate_rows(rows: list[dict]) -> None:
    if not rows or len({sample_id(row) for row in rows}) != len(rows):
        raise ValueError("Distinct published training rows are required")
    for row in rows:
        if row.get("split") != "train" or row.get("domain") not in DOMAIN_PRIORITY:
            raise ValueError("Only released train-split rows may be sampled")
        if not isinstance(row.get("canonical_smiles"), str) or not row["canonical_smiles"]:
            raise ValueError("Canonical structure identity is required")
        if type(row.get("heavy_atoms")) is not int or row["heavy_atoms"] < 0:
            raise ValueError("Heavy-atom count is required")
        if not isinstance(row.get("bond_counts"), dict):
            raise ValueError("Chemical bond counts are required")
        if not isinstance(row.get("profile_id"), str) or not row["profile_id"]:
            raise ValueError("Rendering profile is required")
        for key in ("meta_sha256", "pixel_sha256", "png_sha256", "graph_sha256"):
            value = row.get(key)
            if (not isinstance(value, str) or len(value) != 64
                    or set(value) - set("0123456789abcdef")):
                raise ValueError(f"Valid {key} is required")
        bond_bucket(row["bond_counts"])


def _structure_pools(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["canonical_smiles"]].append(row)
    pools = {}
    for identity, variants in grouped.items():
        priority = max(DOMAIN_PRIORITY[row["domain"]] for row in variants)
        selected = [row for row in variants if DOMAIN_PRIORITY[row["domain"]] == priority]
        signature = (selected[0]["domain"], selected[0]["heavy_atoms"],
                     bond_bucket(selected[0]["bond_counts"]))
        if any((row["domain"], row["heavy_atoms"], bond_bucket(row["bond_counts"])) != signature
               for row in selected):
            raise ValueError("Structure variants disagree on domain, size, or bond stratum")
        pools[identity] = selected
    return pools


def _interleave_strata(pools: dict[str, list[dict]]) -> list[str]:
    strata: dict[str, list[str]] = defaultdict(list)
    for identity, variants in pools.items():
        row = variants[0]
        key = ":".join((row["domain"], size_bucket(row["heavy_atoms"]),
                        bond_bucket(row["bond_counts"])))
        strata[key].append(identity)
    queues = {
        key: sorted(identities, key=lambda identity: _rank(key, identity))
        for key, identities in strata.items()
    }
    heap = [(Fraction(1, 2 * len(queue)), key, 0) for key, queue in queues.items()]
    heapq.heapify(heap)
    order = []
    while heap:
        _, key, index = heapq.heappop(heap)
        order.append(queues[key][index])
        if index + 1 < len(queues[key]):
            heapq.heappush(heap, (Fraction(2 * index + 3, 2 * len(queues[key])), key, index + 1))
    return order


def _choose_style_rows(identities: list[str], pools: dict[str, list[dict]]) -> list[dict]:
    style_totals = Counter(row["profile_id"] for variants in pools.values() for row in variants)
    style_selected: Counter = Counter()
    result = []
    for identity in identities:
        variants = pools[identity]
        by_style: dict[str, list[dict]] = defaultdict(list)
        for row in variants:
            by_style[row["profile_id"]].append(row)
        profile = min(
            by_style,
            key=lambda name: (Fraction(style_selected[name], style_totals[name]),
                              _rank("style", identity, name)),
        )
        chosen = min(by_style[profile], key=lambda row: _rank("image", identity, sample_id(row)))
        result.append(chosen)
        style_selected[profile] += 1
    return result


def make_sample_plan(rows: list[dict], *, count: int = SAMPLE_COUNT) -> dict:
    _validate_rows(rows)
    if type(count) is not int or count < 1:
        raise ValueError("Positive fixed sample count required")
    pools = _structure_pools(rows)
    if len(pools) < count:
        raise ValueError(f"Need {count} distinct structures; only {len(pools)} are available")
    identities = _interleave_strata(pools)[:count]
    selected = _choose_style_rows(identities, pools)
    if len({row["canonical_smiles"] for row in selected}) != count:
        raise ValueError("A molecular structure was selected more than once")
    records = [{key: row[key] for key in (
        "source", "sample_key", "split", "domain", "canonical_smiles", "group_key",
        "meta_sha256", "pixel_sha256", "png_sha256", "graph_sha256", "molecule_id",
        "heavy_atoms", "bond_counts", "profile_id", "stages") if key in row}
        for row in selected]
    summary = {
        "images": len(records),
        "distinct_structures": len({row["canonical_smiles"] for row in records}),
        "domains": dict(sorted(Counter(row["domain"] for row in records).items())),
        "size_buckets": dict(sorted(Counter(size_bucket(row["heavy_atoms"]) for row in records).items())),
        "dominant_bond_buckets": dict(sorted(Counter(bond_bucket(row["bond_counts"]) for row in records).items())),
        "profiles": dict(sorted(Counter(row["profile_id"] for row in records).items())),
        "bond_instances": {
            kind: sum(row["bond_counts"].get(kind, 0) for row in records)
            for kind in ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
        },
    }
    return {
        "schema_version": "chemical-head-probe-training-plan-0.1",
        "seed": SEED,
        "selection": "unique_canonical_structure; proportional_size_bond_domain_interleave; style_deficit_choice",
        "training_only": True,
        "samples": records,
        "summary": summary,
        "samples_sha256": digest(records),
    }


def draft_protocol(*, sample_sha256: str, preparation_sha256: str, initial_sha256: str,
                   manifest_sha256: str) -> dict:
    for value in (sample_sha256, preparation_sha256, initial_sha256, manifest_sha256):
        if (not isinstance(value, str) or len(value) != 64
                or set(value) - set("0123456789abcdef")):
            raise ValueError("All source bindings must be SHA256 digests")
    return {
        "schema_version": "chemical-head-probe-protocol-0.1",
        "status": "draft_not_authorized",
        "training_authorized": False,
        "deployment_authorized": False,
        "training_samples": SAMPLE_COUNT,
        "sample_sha256": sample_sha256,
        "source_preparation_sha256": preparation_sha256,
        "initial_checkpoint_sha256": initial_sha256,
        "release_manifest_sha256": manifest_sha256,
        "trainable_parameters": ["pair.chemical.weight", "pair.chemical.bias"],
        "frozen_parameters": ["pair.ribbon_encoder", "pair.atom_encoder", "pair.score_encoder",
                              "pair.shared", "pair.presence", "all_upstream", "ocr"],
        "objective": "known_matched_positive_pair_chemical_cross_entropy_only",
        "presence_loss": "disabled",
        "unknown_pairs": "excluded; never treated as negatives",
        "empty_supervision": "skip_without_optimizer_update",
        "seed": 20260925,
        "optimizer": {"name": "AdamW", "learning_rate": 2e-5, "weight_decay": 0.0,
                      "gradient_clip": 1.0, "batch_size": 1},
        "mode": {"device": "cuda:0", "dtype": "float32", "autocast": False, "tf32": False},
        "budget": {"wall_seconds": 1800, "maximum_attempts": SAMPLE_COUNT,
                   "maximum_updates": SAMPLE_COUNT, "maximum_invocations": 1,
                   "automatic_retries": 0, "resume_supported": False},
        "checkpoints": list(CHECKPOINTS),
        "evaluation": {"fixed_development_images": 944, "sealed_test_read": False,
                       "new_identity_queries": 0, "media_split": "existing_validation_only"},
        "hard_constraints": ["presence_logits_exact_parity", "selected_pair_set_exact_parity",
                             "all_nonchemical_model_state_exactly_frozen"],
        "selection_policy": {"full_exact": "strict_gain", "full_valid": "not_lower",
                             "full_fallback": "not_higher", "new_fallbacks": 0,
                             "curriculum_exact_drop_max_images": 2},
    }
