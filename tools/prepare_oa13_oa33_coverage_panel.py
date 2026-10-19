"""Prepare the oa13/oa33 coverage-panel metadata plan.

The command is deliberately metadata-only: it reads the quarantined 10k
manifest and a fixed profile-plan JSON, selects an identity-disjoint
256/64 molecule split, and writes manifests.  It never opens raster/SDF
payloads, runs the model, renders an image, or authorizes training.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/manifest.json"
PROFILE_PLAN = ROOT / "outputs/audits/chemvector-oa13-oa33-coverage-metadata-plan-20261017-r1/plan.json"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
OUTPUT = ROOT / "outputs/audits/chemvector-oa13-oa33-coverage-panel-plan-20261017-r3"
TRAIN_MOLECULES = 256
HOLDOUT_MOLECULES = 64
SOURCE_PROFILES = ("oa00", "oa02", "oa11", "oa13", "oa20", "oa22", "oa31", "oa33")
SEED = "chemvector-oa13-oa33-coverage-panel-20261017-r1:"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_key(value: str) -> str:
    return hashlib.sha256((SEED + value).encode("utf-8")).hexdigest()


def _load_profiles(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 8:
        raise ValueError("coverage profile plan must contain exactly eight profiles")
    ids = tuple(str(row.get("profile_id")) for row in profiles)
    if len(set(ids)) != 8 or any(not value.startswith("cov") for value in ids):
        raise ValueError("coverage profile IDs are not fixed/unique")
    for row in profiles:
        if row.get("changed_factor") not in {"none", "font_size", "rotation_degrees", "font_family"}:
            raise ValueError("coverage panel contains an unexpected changed factor")
        if row.get("parent_profile_id") not in {"oa13", "oa33"}:
            raise ValueError("coverage panel parent must be oa13 or oa33")
    return profiles


def _load_candidates(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source manifest is empty")
    by_molecule: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        required = ("sample_key", "molecule_id", "canonical_smiles", "connection_identity", "legacy_scaffold", "profile_id")
        if any(name not in row for name in required):
            raise ValueError("source row missing identity metadata")
        if row["profile_id"] not in SOURCE_PROFILES:
            raise ValueError(f"unexpected source profile {row['profile_id']!r}")
        by_molecule[str(row["molecule_id"])].append(row)
    candidates: dict[str, dict] = {}
    for molecule_id, molecule_rows in by_molecule.items():
        profiles = {str(row["profile_id"]): row for row in molecule_rows}
        if set(profiles) != set(SOURCE_PROFILES):
            continue
        identities = {
            str(row["canonical_smiles"]) for row in molecule_rows
        }
        connections = {str(row["connection_identity"]) for row in molecule_rows}
        scaffolds = {str(row["legacy_scaffold"]) for row in molecule_rows}
        splits = {str(row.get("split")) for row in molecule_rows}
        if len(identities) != 1 or len(connections) != 1 or len(scaffolds) != 1 or len(splits) != 1:
            raise ValueError(f"identity metadata mismatch for {molecule_id}")
        candidates[molecule_id] = {
            "molecule_id": molecule_id,
            "canonical_smiles": next(iter(identities)),
            "connection_identity": next(iter(connections)),
            "legacy_scaffold": next(iter(scaffolds)),
            "source_split": next(iter(splits)),
            "source_rows": profiles,
        }
    if len(candidates) < TRAIN_MOLECULES + HOLDOUT_MOLECULES:
        raise ValueError("source manifest has fewer than 320 complete identities")
    return candidates


def _choose(candidates: Iterable[dict], count: int, *, source_split: str,
            used_connections: set[str], used_scaffolds: set[str]) -> list[dict]:
    chosen: list[dict] = []
    for row in sorted(candidates, key=lambda item: stable_key(item["molecule_id"])):
        if row["source_split"] != source_split:
            continue
        if row["connection_identity"] in used_connections or row["legacy_scaffold"] in used_scaffolds:
            continue
        chosen.append(row)
        used_connections.add(row["connection_identity"])
        used_scaffolds.add(row["legacy_scaffold"])
        if len(chosen) == count:
            return chosen
    raise ValueError(f"unable to select {count} scaffold/connection-disjoint identities")


def build_plan(*, source_manifest: Path = SOURCE_MANIFEST, profile_plan: Path = PROFILE_PLAN) -> dict:
    profiles = _load_profiles(profile_plan)
    candidates = _load_candidates(source_manifest)
    # Holdout is reserved first.  The same connection identity and scaffold
    # cannot occur in the train split, even if molecule IDs differ.
    holdout_connections: set[str] = set()
    holdout_scaffolds: set[str] = set()
    holdout = _choose(candidates.values(), HOLDOUT_MOLECULES, source_split="holdout",
                      used_connections=holdout_connections,
                      used_scaffolds=holdout_scaffolds)
    holdout_ids = {row["molecule_id"] for row in holdout}
    train_connections = set(holdout_connections)
    train_scaffolds = set(holdout_scaffolds)
    train = _choose((row for row in candidates.values() if row["molecule_id"] not in holdout_ids),
                    TRAIN_MOLECULES, source_split="train", used_connections=train_connections,
                    used_scaffolds=train_scaffolds)
    train_ids = {row["molecule_id"] for row in train}
    if train_ids & holdout_ids:
        raise AssertionError("molecule overlap")

    planned: list[dict] = []
    profile_by_parent = defaultdict(list)
    for profile in profiles:
        profile_by_parent[str(profile["parent_profile_id"])].append(profile)
    if sorted(len(rows) for rows in profile_by_parent.values()) != [4, 4]:
        raise ValueError("expected four coverage profiles per parent")
    for split, selected in (("train", train), ("holdout", holdout)):
        for molecule in sorted(selected, key=lambda row: row["molecule_id"]):
            for profile in profiles:
                source_parent = str(profile["parent_profile_id"])
                source_row = molecule["source_rows"][source_parent]
                planned.append({
                    "sample_key": f"{molecule['molecule_id']}__{profile['profile_id']}",
                    "source_sample_key": source_row["sample_key"],
                    "molecule_id": molecule["molecule_id"],
                    "canonical_smiles": molecule["canonical_smiles"],
                    "connection_identity": molecule["connection_identity"],
                    "legacy_scaffold": molecule["legacy_scaffold"],
                    "profile_id": profile["profile_id"],
                    "parent_profile_id": source_parent,
                    "changed_factor": profile["changed_factor"],
                    "split": split,
                    "render_status": "not_rendered",
                })
    planned.sort(key=lambda row: row["sample_key"])
    return {
        "schema_version": "chemvector-oa13-oa33-coverage-panel-plan-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "model_forwards": 0,
        "optimizer_steps": 0,
        "production_checkpoint_write": False,
        "selector_retraining_authorized": False,
        "deployment_authorized": False,
        "sealed_test_read": False,
        "initial_checkpoint_sha256": INITIAL_SHA,
        "source_manifest": str(source_manifest.relative_to(ROOT)),
        "source_manifest_sha256": digest(source_manifest),
        "profile_plan": str(profile_plan.relative_to(ROOT)),
        "profile_plan_sha256": digest(profile_plan),
        "identity_split": {
            "train_molecules": len(train),
            "holdout_molecules": len(holdout),
            "train_images": len(train) * len(profiles),
            "holdout_images": len(holdout) * len(profiles),
            "images": len(planned),
            "molecule_overlap": len(train_ids & holdout_ids),
            "connection_identity_overlap": len({row["connection_identity"] for row in train} &
                                                {row["connection_identity"] for row in holdout}),
            "scaffold_overlap": len({row["legacy_scaffold"] for row in train} &
                                     {row["legacy_scaffold"] for row in holdout}),
            "source_split_preserved": {
                "train": len({row["molecule_id"] for row in train if row["source_split"] == "train"}) == len(train),
                "holdout": len({row["molecule_id"] for row in holdout if row["source_split"] == "holdout"}) == len(holdout),
            },
        },
        "profiles": profiles,
        "gate": {
            "metadata_only": True,
            "exactly_eight_profiles": len(profiles) == 8,
            "identity_disjoint": not bool(train_ids & holdout_ids),
            "connection_disjoint": not bool({row["connection_identity"] for row in train} &
                                             {row["connection_identity"] for row in holdout}),
            "scaffold_disjoint": not bool({row["legacy_scaffold"] for row in train} &
                                           {row["legacy_scaffold"] for row in holdout}),
            "training_admissible": False,
            "next": "render/quality-audit then one frozen initial replay; no selector training",
        },
        "rows": planned,
    }


def run(*, output: Path = OUTPUT, source_manifest: Path = SOURCE_MANIFEST,
        profile_plan: Path = PROFILE_PLAN) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if digest(INITIAL) != INITIAL_SHA:
        raise ValueError("initial checkpoint SHA256 changed")
    report = build_plan(source_manifest=source_manifest, profile_plan=profile_plan)
    output.mkdir(parents=True)
    manifest = {"schema_version": report["schema_version"], "rows": report.pop("rows")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    report["manifest_sha256"] = digest(output / "manifest.json")
    (output / "profiles.json").write_text(json.dumps(report["profiles"], indent=2, ensure_ascii=False) + "\n")
    report["profiles_sha256"] = digest(output / "profiles.json")
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "identity_split": report["identity_split"],
                      "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--profile-plan", type=Path, default=PROFILE_PLAN)
    args = parser.parse_args()
    run(output=args.output, source_manifest=args.source_manifest, profile_plan=args.profile_plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
