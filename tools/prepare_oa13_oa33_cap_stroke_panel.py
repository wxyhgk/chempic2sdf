"""Prepare the second, metadata-only cap/stroke isolation panel.

The panel is a fail-branch diagnostic after renderer-quality or coverage
failure.  It uses only identity metadata from the quarantined 10k manifest;
it does not render, read image/SDF payloads, run the model, or authorize any
training.  By default cov13 is prepared first.  ``--include-cov33`` adds the
same three-cell diagnostic for the second failing parent.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/manifest.json"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
OUTPUT = ROOT / "outputs/audits/chemvector-oa13-oa33-cap-stroke-panel-plan-20261017-r1"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
TRAIN = 128
HOLDOUT = 32
SEED = "chemvector-oa13-oa33-cap-stroke-panel-20261017-r1:"

PARENTS = {
    "cov13": {
        "parent_profile_id": "oa13", "font_family": "DejaVu Serif", "font_size": 32,
        "bond_length_cap": 28.0, "bond_line_width": 3.2, "rotation_degrees": 180.0,
        "source_split": "train",
    },
    "cov33": {
        "parent_profile_id": "oa33", "font_family": "Nimbus Roman", "font_size": 32,
        "bond_length_cap": 52.0, "bond_line_width": 1.6, "rotation_degrees": 0.0,
        "source_split": "train",
    },
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_key(value: str) -> str:
    return hashlib.sha256((SEED + value).encode("utf-8")).hexdigest()


def load_candidates(source: Path) -> dict[str, dict]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source manifest is empty")
    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        required = ("sample_key", "molecule_id", "canonical_smiles", "connection_identity", "legacy_scaffold", "profile_id", "split")
        if any(field not in row for field in required):
            raise ValueError("source row lacks identity metadata")
        by_id[str(row["molecule_id"])].append(row)
    candidates: dict[str, dict] = {}
    for molecule_id, molecule_rows in by_id.items():
        profile_rows = {str(row["profile_id"]): row for row in molecule_rows}
        if "oa13" not in profile_rows or "oa33" not in profile_rows:
            continue
        identities = {str(row["canonical_smiles"]) for row in molecule_rows}
        connections = {str(row["connection_identity"]) for row in molecule_rows}
        scaffolds = {str(row["legacy_scaffold"]) for row in molecule_rows}
        splits = {str(row["split"]) for row in molecule_rows}
        if any(len(values) != 1 for values in (identities, connections, scaffolds, splits)):
            raise ValueError(f"identity metadata mismatch for {molecule_id}")
        candidates[molecule_id] = {
            "molecule_id": molecule_id,
            "canonical_smiles": next(iter(identities)),
            "connection_identity": next(iter(connections)),
            "legacy_scaffold": next(iter(scaffolds)),
            "source_split": next(iter(splits)),
            "source_rows": {key: profile_rows[key] for key in ("oa13", "oa33")},
        }
    return candidates


def choose(candidates: list[dict], source_split: str, count: int,
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
    raise ValueError(f"unable to choose {count} {source_split} identities")


def profile_rows(parent: str, include_cov33: bool) -> list[dict]:
    p = PARENTS[parent]
    prefix = parent
    return [
        {"profile_id": f"{prefix}_base", "parent_profile_id": p["parent_profile_id"], "changed_factor": "none", **{k: p[k] for k in ("font_family", "font_size", "bond_length_cap", "bond_line_width", "rotation_degrees")}},
        {"profile_id": f"{prefix}_cap52" if parent == "cov13" else f"{prefix}_cap28", "parent_profile_id": p["parent_profile_id"], "changed_factor": "bond_length_cap", **{k: (52.0 if parent == "cov13" and k == "bond_length_cap" else 28.0 if parent == "cov33" and k == "bond_length_cap" else p[k]) for k in ("font_family", "font_size", "bond_length_cap", "bond_line_width", "rotation_degrees")}},
        {"profile_id": f"{prefix}_stroke16" if parent == "cov13" else f"{prefix}_stroke32", "parent_profile_id": p["parent_profile_id"], "changed_factor": "bond_line_width", **{k: (1.6 if parent == "cov13" and k == "bond_line_width" else 3.2 if parent == "cov33" and k == "bond_line_width" else p[k]) for k in ("font_family", "font_size", "bond_length_cap", "bond_line_width", "rotation_degrees")}},
    ]


def build_plan(*, source: Path = SOURCE, include_cov33: bool = False) -> dict:
    candidates = load_candidates(source)
    parents = ["cov13", "cov33"] if include_cov33 else ["cov13"]
    profiles = [profile for parent in parents for profile in profile_rows(parent, include_cov33)]
    # Select once and reuse the same identity groups across parent panels. This
    # makes the cap/stroke comparison paired and keeps the denominator fixed.
    holdout_connections: set[str] = set()
    holdout_scaffolds: set[str] = set()
    holdout = choose(list(candidates.values()), "holdout", HOLDOUT,
                     holdout_connections, holdout_scaffolds)
    train_connections = set(holdout_connections)
    train_scaffolds = set(holdout_scaffolds)
    train = choose(list(candidates.values()), "train", TRAIN,
                   train_connections, train_scaffolds)
    holdout_ids = {row["molecule_id"] for row in holdout}
    train_ids = {row["molecule_id"] for row in train}
    if train_ids & holdout_ids:
        raise AssertionError("molecule identity overlap")
    planned: list[dict] = []
    for split, selected in (("train", train), ("holdout", holdout)):
        for molecule in sorted(selected, key=lambda row: row["molecule_id"]):
            for profile in profiles:
                parent = str(profile["parent_profile_id"])
                source_row = molecule["source_rows"][parent]
                planned.append({
                    "sample_key": f"{molecule['molecule_id']}__{profile['profile_id']}",
                    "source_sample_key": source_row["sample_key"],
                    "molecule_id": molecule["molecule_id"],
                    "canonical_smiles": molecule["canonical_smiles"],
                    "connection_identity": molecule["connection_identity"],
                    "legacy_scaffold": molecule["legacy_scaffold"],
                    "source_split": molecule["source_split"],
                    "profile_id": profile["profile_id"],
                    "parent_profile_id": parent,
                    "changed_factor": profile["changed_factor"],
                    "split": split,
                    "render_status": "not_rendered",
                })
    planned.sort(key=lambda row: row["sample_key"])
    train_groups = {row["connection_identity"] for row in train}
    holdout_groups = {row["connection_identity"] for row in holdout}
    train_scaf = {row["legacy_scaffold"] for row in train}
    holdout_scaf = {row["legacy_scaffold"] for row in holdout}
    return {
        "schema_version": "chemvector-oa13-oa33-cap-stroke-panel-plan-0.1",
        "status": "metadata_only_fail_branch",
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
        "source_manifest": str(source.relative_to(ROOT)),
        "source_manifest_sha256": digest(source),
        "parents": parents,
        "profiles": profiles,
        "identity_split": {
            "train_molecules": len(train), "holdout_molecules": len(holdout),
            "profiles_per_parent": 3, "parents": len(parents),
            "train_images": len(train) * len(profiles),
            "holdout_images": len(holdout) * len(profiles), "images": len(planned),
            "molecule_overlap": len(train_ids & holdout_ids),
            "connection_identity_overlap": len(train_groups & holdout_groups),
            "scaffold_overlap": len(train_scaf & holdout_scaf),
            "source_split_preserved": all(row["source_split"] == "train" for row in train) and all(row["source_split"] == "holdout" for row in holdout),
        },
        "gate": {
            "metadata_only": True,
            "single_factor_controls": all(profile["changed_factor"] in {"none", "bond_length_cap", "bond_line_width"} for profile in profiles),
            "identity_disjoint": not bool(train_ids & holdout_ids),
            "connection_disjoint": not bool(train_groups & holdout_groups),
            "scaffold_disjoint": not bool(train_scaf & holdout_scaf),
            "training_admissible": False,
            "next": "render-quality audit and one frozen initial replay; do not train selector",
        },
        "rows": planned,
    }


def run(*, output: Path = OUTPUT, source: Path = SOURCE, include_cov33: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if digest(INITIAL) != INITIAL_SHA:
        raise ValueError("initial checkpoint SHA256 changed")
    report = build_plan(source=source, include_cov33=include_cov33)
    output.mkdir(parents=True)
    rows = report.pop("rows")
    (output / "manifest.json").write_text(json.dumps({"schema_version": report["schema_version"], "rows": rows}, indent=2, ensure_ascii=False) + "\n")
    (output / "profiles.json").write_text(json.dumps(report["profiles"], indent=2, ensure_ascii=False) + "\n")
    report["manifest_sha256"] = digest(output / "manifest.json")
    report["profiles_sha256"] = digest(output / "profiles.json")
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "identity_split": report["identity_split"], "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--include-cov33", action="store_true")
    args = parser.parse_args()
    run(output=args.output, source=args.source, include_cov33=args.include_cov33)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
