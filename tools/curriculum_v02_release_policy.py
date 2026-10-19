"""Pure grouped-release proposal and bounded-fit selection. Grants no admission."""

from collections import Counter, defaultdict
from hashlib import sha256

SEED = "20260919-curriculum-v02-release-v1"
IDENTITY_FIELDS = ("molecule_id", "group_key", "connectivity_prefix")


def grouped_proposal(candidates, old_molecules, diagnostic_ids):
    """Union identity/core/connectivity; old validation ALWAYS wins exclusion.

    Old training and previously diagnosed components are train-only. Untouched
    components use a fixed 20% hash holdout. No model scores or substitutions.
    """
    if (len({r["molecule_id"] for r in candidates}) != len(candidates)
            or not set(diagnostic_ids) <= {r["molecule_id"] for r in candidates}):
        raise ValueError("Invalid candidate/diagnostic identities")
    rows = old_molecules + candidates
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for field in IDENTITY_FIELDS:
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError("Missing grouping identity")
            key = field, row[field]
            if key in seen:
                parents[find(i)] = find(seen[key])
            seen[key] = i
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append((i, row))
    decisions = {}
    for members in components.values():
        old_splits = {r["split"] for i, r in members if i < len(old_molecules)}
        if not old_splits <= {"train", "validation"} or len(old_splits) > 1:
            raise ValueError("Existing train/validation union conflict; do not repair old split")
        key = min(r["group_key"] for _, r in members)
        diagnosed = any(r["molecule_id"] in diagnostic_ids for _, r in members)
        if "validation" in old_splits:
            split, reason = "excluded", "existing_validation_component"
        elif "train" in old_splits:
            split, reason = "train", "existing_training_component"
        elif diagnosed:
            split, reason = "train", "previously_diagnosed_component_not_unseen"
        else:
            split = "validation" if int(sha256(f"{SEED}:{key}".encode()).hexdigest()[:16], 16) % 5 == 0 else "train"
            reason = "fixed_hash_unexposed_component"
        for i, row in members:
            if i >= len(old_molecules):
                decisions[row["molecule_id"]] = dict(molecule_id=row["molecule_id"], proposed_split=split,
                    reason=reason, component_key=key, component_previously_diagnosed=diagnosed,
                    training_authorized=False)
    return [decisions[r["molecule_id"]] for r in candidates]


def partition_samples(candidates, decisions, inventory, old_pixels):
    """Remove exact old-train PNG duplicates; fail on unexplained pixel conflicts.

    Old-validation group exclusions are already whole-component. Byte-identical
    pictures cannot silently cross an additional boundary or carry new labels.
    RGB identity within the new set is inherited from the fixed quality audit.
    """
    decision = {r["molecule_id"]: r for r in decisions}
    output, seen_keys, seen_png, seen_rgb = [], set(), set(), set()
    for candidate in candidates:
        if candidate["image_count"] != len(set(candidate["sample_keys"])):
            raise ValueError("Candidate sample denominator mismatch")
        rule = decision[candidate["molecule_id"]]
        for key in candidate["sample_keys"]:
            item = inventory[key]
            if (key in seen_keys or item["molecule_id"] != candidate["molecule_id"]
                    or item["png_sha256"] in seen_png or item["decoded_rgb_sha256"] in seen_rgb):
                raise ValueError("New sample/pixel inventory is not unique and identity-bound")
            seen_keys.add(key); seen_png.add(item["png_sha256"]); seen_rgb.add(item["decoded_rgb_sha256"])
            hits = old_pixels.get(item["png_sha256"], [])
            if any(r["canonical_smiles"] != candidate["canonical_smiles"] for r in hits):
                raise ValueError("Cross-release identical PNG has conflicting chemistry")
            if any(r["split"] == "validation" for r in hits) and rule["proposed_split"] != "excluded":
                raise ValueError("Old validation pixel overlap escaped group exclusion")
            split, reason = rule["proposed_split"], rule["reason"]
            if split != "excluded" and hits:
                if split != "train" or any(r["split"] != "train" for r in hits):
                    raise ValueError("Old training pixels cannot become new validation")
                split, reason = "duplicate_old_train", "byte_identical_admitted_train_image"
            output.append({**item, **rule, "proposed_split": split, "reason": reason,
                "canonical_smiles": candidate["canonical_smiles"], "group_key": candidate["group_key"],
                "connectivity_prefix": candidate["connectivity_prefix"],
                "old_pixel_matches": [{k: r[k] for k in ("source", "sample_key", "split", "meta_sha256")} for r in hits]})
    if len(output) != sum(r["image_count"] for r in candidates):
        raise ValueError("Partition lost candidate samples")
    return output


def select_trial(diagnostic_samples, partitions, old_replay, validation_count=64):
    """Exact original witnesses or identical admitted aliases; never replacements."""
    by_key = {r["sample_key"]: r for r in partitions}
    fitting, excluded = [], []
    for witness in diagnostic_samples:
        row = by_key[witness["sample_key"]]
        if row["molecule_id"] != witness["molecule_id"] or row["canonical_smiles"] != witness["canonical_smiles"]:
            raise ValueError("Diagnostic identity changed")
        if row["proposed_split"] == "excluded":
            excluded.append({**witness, "reason": row["reason"]})
        elif row["proposed_split"] == "duplicate_old_train":
            alias = min(row["old_pixel_matches"], key=lambda r: (r["source"], r["sample_key"]))
            fitting.append({**witness, "origin": "identical_admitted_training_alias", "admitted_alias": alias})
        elif row["proposed_split"] == "train":
            fitting.append({**witness, "origin": "new_quarantine_needs_publication"})
        else:
            raise ValueError("Already diagnosed witness cannot become unseen validation")
    if any(r["split"] != "train" or r["domain"] != "full_molecule" for r in old_replay):
        raise ValueError("Replay must consist of existing full-molecule training data")
    holdout = defaultdict(list)
    for row in partitions:
        if row["proposed_split"] == "validation":
            if row["component_previously_diagnosed"]:
                raise ValueError("New validation contains an exposed diagnostic component")
            holdout[row["component_key"]].append(row)
    if len(holdout) < validation_count:
        raise ValueError("Too few untouched validation components; no substitutions")
    keys = sorted(holdout, key=lambda k: (sha256(f"{SEED}:trial-dev:{k}".encode()).hexdigest(), k))[:validation_count]
    dev = [min(holdout[k], key=lambda r: (sha256(f"{SEED}:trial-image:{r['sample_key']}".encode()).hexdigest(), r["sample_key"])) for k in keys]
    return dict(fitting_witnesses=fitting, excluded_diagnostic_witnesses=excluded,
        old_full_molecule_replay=old_replay, new_group_validation=dev,
        fitting_origins=dict(Counter(r["origin"] for r in fitting)), training_authorized=False,
        selection="Existing diagnostic order, holdout conflicts excluded without replacement; exact old-pixel aliases only",
        validation_selection="64 hash-ordered untouched components, one hash-ordered image each; no model outcome selection")
