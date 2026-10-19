"""Plan an identity/scaffold-disjoint release from the audited quarantine batch.

This is metadata-only: no images, SDF payloads, sealed identities, model, or
optimizer are opened.  The output is not training authorization; it is the
fixed input to a later byte/admission audit and frozen replay.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "outputs/audits/structure-expansion-public-identity-20260923/remaining-candidates.json"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
OUTPUT = ROOT / "outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1"
# The public-reviewed batch has 1,437 complete eight-profile molecules after
# 88 oa13 quality rejects and 9 conservative identity exclusions.  Keep a
# deterministic reserve rather than pretending the missing profiles exist.
TRAIN_MOLECULES = 1100
HOLDOUT_MOLECULES = 160
PROFILES = ("oa00", "oa02", "oa11", "oa13", "oa20", "oa22", "oa31", "oa33")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rank(split: str, key: str) -> str:
    return hashlib.sha256(f"chemvector-10k-release-20261016:{split}:{key}".encode()).hexdigest()


def group_key(row: dict) -> tuple[str, str, str]:
    # The public candidate record already binds the three identity notions
    # used by the prior review. Empty legacy scaffolds are kept separate by
    # molecule rather than accidentally grouping all acyclic structures.
    scaffold = row.get("legacy_scaffold") or f"acyclic:{row['molecule_id']}"
    return str(row["molecule_id"]), str(row["connectivity_prefix"]), scaffold


def choose(rows: list[dict], count: int, split: str, used_conn: set[str], used_scaffold: set[str]) -> list[dict]:
    eligible = [r for r in rows if r["image_count"] == len(PROFILES)
                and r["connectivity_prefix"] not in used_conn
                and (r.get("legacy_scaffold") or f"acyclic:{r['molecule_id']}") not in used_scaffold]
    eligible.sort(key=lambda r: rank(split, r["molecule_id"]))
    chosen = eligible[:count]
    if len(chosen) != count:
        raise ValueError(f"unable to choose {count} full-profile molecules for {split}; got {len(chosen)}")
    return chosen


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    public = json.loads(PUBLIC.read_text(encoding="utf-8"))
    if not public or any(row["image_count"] not in {7, 8} for row in public):
        raise ValueError("unexpected public candidate inventory")
    # Work only from the public-side remaining candidate inventory.  The source
    # samples are metadata-checked, not payload-read, and are used to bind the
    # exact profile membership in the resulting manifest.
    samples = json.loads((SOURCE / "samples.json").read_text(encoding="utf-8"))
    sample_by_id: dict[str, list[dict]] = defaultdict(list)
    for row in samples:
        sample_by_id[row["molecule_id"]].append(row)
    full = [row for row in public if row["image_count"] == len(PROFILES)
            and {s["profile_id"] for s in sample_by_id[row["molecule_id"]]} == set(PROFILES)]
    if len(full) < TRAIN_MOLECULES + HOLDOUT_MOLECULES:
        raise ValueError(f"not enough full-profile molecules: {len(full)}")
    # Greedy deterministic selection ensures both connection and scaffold
    # disjointness. The public identity review already removed evaluation
    # overlaps; this plan does not claim a broader historical ledger.
    train = choose(full, TRAIN_MOLECULES, "train", set(), set())
    train_conn = {r["connectivity_prefix"] for r in train}
    train_scaffold = {r.get("legacy_scaffold") or f"acyclic:{r['molecule_id']}" for r in train}
    remaining = [r for r in full if r["molecule_id"] not in {x["molecule_id"] for x in train}]
    holdout = choose(remaining, HOLDOUT_MOLECULES, "holdout", train_conn, train_scaffold)
    train_ids = {r["molecule_id"] for r in train}; holdout_ids = {r["molecule_id"] for r in holdout}
    if train_ids & holdout_ids:
        raise ValueError("molecule identity overlap")
    rows_out = []
    for split, selected in (("train", train), ("holdout", holdout)):
        for molecule in sorted(selected, key=lambda r: r["molecule_id"]):
            by_profile = {s["profile_id"]: s for s in sample_by_id[molecule["molecule_id"]]}
            for profile in PROFILES:
                sample = by_profile[profile]
                rows_out.append({
                    "sample_key": sample["sample_key"], "molecule_id": molecule["molecule_id"],
                    "canonical_smiles": molecule["canonical_smiles"], "connection_identity": molecule["connectivity_prefix"],
                    "legacy_scaffold": molecule.get("legacy_scaffold") or f"acyclic:{molecule['molecule_id']}",
                    "profile_id": profile, "split": split,
                })
    cell_counts = Counter()
    for row in rows_out:
        cell_counts[f"{row['split']}/{row['profile_id']}"] += 1
    report = {
        "schema_version": "chemvector-structure-expansion-10k-release-plan-0.1",
        "complete": True, "read_only": True, "payloads_read": False,
        "training_performed": False, "training_authorized": False,
        "gpu_used": False, "sealed_test_read": False, "deployment_authorized": False,
        "source": {"public_inventory": str(PUBLIC.relative_to(ROOT)), "public_sha256": digest(PUBLIC),
                    "source_samples_sha256": digest(SOURCE / "samples.json"),
                    "source_summary_sha256": digest(SOURCE / "summary.json")},
        "selection": {"train_molecules": len(train_ids), "holdout_molecules": len(holdout_ids),
                       "train_images": sum(r["split"] == "train" for r in rows_out),
                       "holdout_images": sum(r["split"] == "holdout" for r in rows_out),
                       "total_images": len(rows_out), "profiles": list(PROFILES),
                       "full_profile_only": True},
        "identity_split": {"molecule_overlap": len(train_ids & holdout_ids),
                           "connection_overlap": len(train_conn & {r["connectivity_prefix"] for r in holdout}),
                           "scaffold_overlap": len(train_scaffold & {r.get("legacy_scaffold") or f"acyclic:{r['molecule_id']}" for r in holdout}),
                           "evaluation_overlap_checked_by_public_review": True,
                           "complete_historical_prior_use_ledger": False},
        "profile_counts": dict(cell_counts),
        "gate": {"metadata_plan_pass": True, "training_admissible": False,
                 "next": ["exact-byte release/admission audit", "frozen initial replay",
                          "raw reachability >= 0.98 per core style", "strict strata report",
                          "only then consider one bounded training shadow"]},
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "manifest.json").write_text(json.dumps({"schema_version": report["schema_version"], "rows": rows_out}, indent=2, ensure_ascii=False) + "\n")
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "selection": report["selection"], "identity_split": report["identity_split"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
