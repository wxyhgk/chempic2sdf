"""Plan a larger identity-disjoint visual group-selector pool, metadata only."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
MINE = ROOT / "outputs/audits/chemvector-targeted-selector-group-pool-mine-20261006-r1/report.json"
PUBLIC = ROOT / "outputs/audits/structure-expansion-public-identity-20260923/remaining-candidates.json"
OLD = ROOT / "outputs/audits/chemvector-structure-expansion-targeted-split-plan-20261006-r5/report.json"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-visual-pool-plan-20261006-r1"
PROFILES = ("oa11", "oa13")
TRAIN_MOLECULES = 160
HOLDOUT_MOLECULES = 80


def stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    mine = json.loads(MINE.read_text(encoding="utf-8"))
    molecules = {r["molecule_id"]: r for r in json.loads((DATA / "molecules.json").read_text(encoding="utf-8"))}
    samples = {r["sample_key"]: r for r in json.loads((DATA / "samples.json").read_text(encoding="utf-8"))}
    cleared = {key for row in json.loads(PUBLIC.read_text(encoding="utf-8")) for key in row["sample_keys"]}
    old_ids = {row["molecule_id"] for row in json.loads(OLD.read_text(encoding="utf-8"))["rows"]}
    by_molecule: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in mine["records"]:
        key = record["sample_key"]
        if key not in cleared or record["molecule_id"] in old_ids:
            continue
        by_molecule[record["molecule_id"]][record["profile_id"]] = record
    candidates = []
    for molecule_id, profiles in by_molecule.items():
        if set(profiles) != set(PROFILES) or molecule_id not in molecules:
            continue
        molecule = molecules[molecule_id]
        candidates.append({
            "molecule_id": molecule_id, "canonical_smiles": molecule["canonical_smiles"],
            "connection_identity_without_isotope_stereo": molecule["connection_identity_without_isotope_stereo"],
            "profiles": profiles,
            "strict_by_profile": {profile: int(profiles[profile]["strict_selector_drop"]) for profile in PROFILES},
            "strict_total": sum(int(profiles[profile]["strict_selector_drop"]) for profile in PROFILES),
        })
    if len(candidates) < TRAIN_MOLECULES + HOLDOUT_MOLECULES:
        raise ValueError("Insufficient two-profile candidates")

    def choose(pool: list[dict], count: int, used_groups: set[str]) -> list[dict]:
        chosen: list[dict] = []
        local_groups = set(used_groups)
        profile_counts = Counter()
        # First guarantee a substantial positive image quota in each style.
        quotas = {"oa11": max(20, count // 4), "oa13": max(20, count // 4)}
        while len(chosen) < count and any(profile_counts[p] < quotas[p] for p in PROFILES):
            options = [row for row in pool if row["connection_identity_without_isotope_stereo"] not in local_groups
                       and any(profile_counts[p] < quotas[p] and row["strict_by_profile"][p] > 0 for p in PROFILES)]
            if not options:
                break
            row = max(options, key=lambda r: (sum(r["strict_by_profile"][p] for p in PROFILES if profile_counts[p] < quotas[p]),
                                               r["strict_total"], stable(r["molecule_id"])))
            chosen.append(row); local_groups.add(row["connection_identity_without_isotope_stereo"])
            for profile in PROFILES:
                profile_counts[profile] += int(row["strict_by_profile"][profile] > 0)
        while len(chosen) < count:
            options = [row for row in pool if row["connection_identity_without_isotope_stereo"] not in local_groups]
            if not options:
                raise ValueError("Unable to fill identity-disjoint split")
            row = max(options, key=lambda r: (r["strict_total"], stable(r["molecule_id"])))
            chosen.append(row); local_groups.add(row["connection_identity_without_isotope_stereo"])
        return chosen

    holdout = choose(sorted(candidates, key=lambda r: stable(r["molecule_id"])), HOLDOUT_MOLECULES, set())
    holdout_groups = {r["connection_identity_without_isotope_stereo"] for r in holdout}
    train = choose([r for r in candidates if r["connection_identity_without_isotope_stereo"] not in holdout_groups], TRAIN_MOLECULES, holdout_groups)
    train_groups = {r["connection_identity_without_isotope_stereo"] for r in train}
    if holdout_groups & train_groups or {r["molecule_id"] for r in holdout} & {r["molecule_id"] for r in train}:
        raise ValueError("Split identity overlap")
    planned = []
    for split, selected in (("train", train), ("holdout", holdout)):
        for molecule in selected:
            for profile in PROFILES:
                key = molecule["profiles"][profile]["sample_key"]
                planned.append({"sample_key": key, "molecule_id": molecule["molecule_id"],
                                "canonical_smiles": molecule["canonical_smiles"],
                                "connection_identity_without_isotope_stereo": molecule["connection_identity_without_isotope_stereo"],
                                "profile_id": profile, "split": split,
                                "png_sha256": samples[key]["png_sha256"],
                                "strict_selector_drop_mined": molecule["profiles"][profile]["strict_selector_drop"]})
    planned.sort(key=lambda r: r["sample_key"])
    report = {
        "schema_version": "chemvector-targeted-selector-visual-pool-plan-0.1",
        "complete": True, "status": "plan_only_quarantine", "read_only": True,
        "training_authorized": False, "training_performed": False, "model_forwards": 0,
        "sealed_test_read": False,
        "source": {"dataset": str(DATA.relative_to(ROOT)), "mine_sha256": hashlib.sha256(MINE.read_bytes()).hexdigest(),
                    "public_remaining_sha256": hashlib.sha256(PUBLIC.read_bytes()).hexdigest(),
                    "old_plan_sha256": hashlib.sha256(OLD.read_bytes()).hexdigest(),
                    "candidate_molecules": len(candidates)},
        "counts": {"train_molecules": len(train), "holdout_molecules": len(holdout),
                   "train_images": len(train) * 2, "holdout_images": len(holdout) * 2,
                   "train_mined_strict_images": sum(r["strict_total"] for r in train),
                   "holdout_mined_strict_images": sum(r["strict_total"] for r in holdout),
                   "train_mined_strict_by_profile": {p: sum(r["strict_by_profile"][p] for r in train) for p in PROFILES},
                   "holdout_mined_strict_by_profile": {p: sum(r["strict_by_profile"][p] for r in holdout) for p in PROFILES}},
        "identity_split": {"molecule_overlap": 0, "connection_overlap": 0, "sealed_access": False},
        "rows": planned,
        "gate": {"minimum_train_strict_50": sum(r["strict_total"] for r in train) >= 50,
                 "minimum_holdout_strict_14": sum(r["strict_total"] for r in holdout) >= 14,
                 "oa11_train_positive_images": sum(r["strict_by_profile"]["oa11"] > 0 for r in train),
                 "oa11_holdout_positive_images": sum(r["strict_by_profile"]["oa11"] > 0 for r in holdout),
                 "training_admissible": False,
                 "next": "fresh frozen-initial group contract on this plan before any release"},
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (OUTPUT / "manifest.json").write_text(json.dumps({"rows": planned}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "counts": report["counts"], "gate": report["gate"]}, indent=2))
    return report


if __name__ == "__main__":
    run()
