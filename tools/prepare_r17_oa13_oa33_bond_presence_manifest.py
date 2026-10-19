#!/usr/bin/env python3
"""Build a metadata-only, identity-clean oa13/oa33 bond-presence cohort.

This is a data/protocol audit, not a training launcher.  It selects complete
profile pairs from the public structure-expansion release after excluding all
known r17 cache/artifact cohorts, and reports post-hoc strata for bonds
incident to visible O/N labels.  The manifest itself contains no coordinates,
graphs, labels, or target IDs; those are used only to produce aggregate audit
counts.  It is therefore safe to hand to a frozen replay before any bounded
CPU shadow is considered.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/manifest.json"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/samples"
KNOWN = ROOT / "outputs/audits/chemvector-r17-selector-known-clean-shadow-cohort-v1/manifest.json"
SLICE = ROOT / "outputs/audits/chemvector-r17-selector-slice80-disjoint-shadow-v1/manifest.json"
STRESS = ROOT / "outputs/audits/chemvector-r17-oa13-oa33-known-clean-stress-manifest-v2/manifest.json"
CACHE = ROOT / "outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m128-t0p72-c1-panel128/feature-cache.pt"
OUT = ROOT / "outputs/audits/chemvector-r17-oa13-oa33-bond-presence-manifest-v1"
PROFILES = ("oa13", "oa33")
TRAIN_MOLECULES = 32
HOLDOUT_MOLECULES = 16
SEED = "r17-oa13-oa33-bond-presence-v1:"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return list(value.get("rows", []))


def canonical_key(row: dict, field: str) -> str:
    return str(row.get(field, "")).strip()


def excluded_ids() -> tuple[set[str], dict[str, set[str]]]:
    """Return molecule IDs and identity fields seen by prior public artifacts."""
    paths = (KNOWN, SLICE, STRESS)
    seen: dict[str, set[str]] = {field: set() for field in ("molecule_id", "canonical_smiles", "connection_identity", "legacy_scaffold")}
    for path in paths:
        for row in rows(path):
            for field in seen:
                value = canonical_key(row, field)
                if value:
                    seen[field].add(value)
    # Cache rows only expose sample keys.  Resolve their molecule IDs through
    # the public release metadata without opening graph/layer targets.
    import torch
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    release = rows(RELEASE)
    by_sample = {str(row["sample_key"]): row for row in release}
    for records in cache.get("records", {}).values():
        for record in records:
            row = by_sample.get(str(record.get("sample_key")))
            if row is None:
                continue
            for field in seen:
                value = canonical_key(row, field)
                if value:
                    seen[field].add(value)
    return seen["molecule_id"], seen


def classify_molecule(sample_key: str) -> dict:
    """Post-hoc strata only; this output never enters model inputs."""
    layer = json.loads((SOURCE / f"{sample_key}.layers.json").read_text(encoding="utf-8"))
    graph = json.loads((SOURCE / f"{sample_key}.graph.json").read_text(encoding="utf-8"))
    atoms = {int(atom["atom_id"]): atom for atom in layer.get("atoms", [])}
    degree: Counter[int] = Counter()
    bonds_by_type: Counter[str] = Counter()
    incident_by_type: Counter[str] = Counter()
    visible_on: set[int] = set()
    visible_o: set[int] = set()
    visible_n: set[int] = set()
    for atom_id, atom in atoms.items():
        if bool(atom.get("visible_text")) and str(atom.get("element")) in {"O", "N"}:
            visible_on.add(atom_id)
            (visible_o if str(atom.get("element")) == "O" else visible_n).add(atom_id)
    incident = 0
    for bond in graph.get("bonds", []):
        a, b = int(bond["begin_atom"]), int(bond["end_atom"])
        degree[a] += 1
        degree[b] += 1
        typ = str(bond.get("chemical_type", "SINGLE")).upper()
        bonds_by_type[typ] += 1
        if a in visible_on or b in visible_on:
            incident += 1
            incident_by_type[typ] += 1
    return {
        "visible_o": len(visible_o),
        "visible_n": len(visible_n),
        "visible_on": len(visible_on),
        "visible_o_degree_1": sum(degree[i] == 1 for i in visible_o),
        "visible_o_degree_2plus": sum(degree[i] >= 2 for i in visible_o),
        "visible_n_degree_1": sum(degree[i] == 1 for i in visible_n),
        "visible_n_degree_2plus": sum(degree[i] >= 2 for i in visible_n),
        "all_bonds": sum(bonds_by_type.values()),
        "incident_visible_on_bonds": incident,
        "incident_visible_on_bonds_by_type": dict(sorted(incident_by_type.items())),
        "all_bonds_by_type": dict(sorted(bonds_by_type.items())),
    }


def summarize(selected: Iterable[dict]) -> dict:
    total = Counter()
    by_profile: dict[str, Counter] = defaultdict(Counter)
    bond_types: dict[str, Counter] = defaultdict(Counter)
    for row in selected:
        stats = classify_molecule(str(row["sample_key"]))
        profile = str(row["profile_id"])
        total.update({key: value for key, value in stats.items() if isinstance(value, int)})
        by_profile[profile].update({key: value for key, value in stats.items() if isinstance(value, int)})
        bond_types[profile].update(stats["incident_visible_on_bonds_by_type"])
    return {
        "rows": len(list(selected)) if not isinstance(selected, list) else len(selected),
        "images": len(list(selected)) if not isinstance(selected, list) else len(selected),
        "by_profile": {k: dict(v) for k, v in sorted(by_profile.items())},
        "totals": dict(total),
        "incident_visible_on_bonds_by_type": {k: dict(v) for k, v in sorted(bond_types.items())},
    }


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    release = rows(RELEASE)
    molecule_excluded, identities = excluded_ids()
    by_molecule: dict[str, list[dict]] = defaultdict(list)
    for row in release:
        if row.get("profile_id") in PROFILES and row.get("split") in {"train", "holdout"}:
            by_molecule[str(row["molecule_id"])].append(row)
    eligible = []
    for molecule_id, group in by_molecule.items():
        if molecule_id in molecule_excluded:
            continue
        if {str(r["profile_id"]) for r in group} != set(PROFILES):
            continue
        if any(str(r.get(field, "")) in identities[field] for r in group for field in ("canonical_smiles", "connection_identity", "legacy_scaffold")):
            continue
        eligible.append(sorted(group, key=lambda r: str(r["sample_key"])))
    # Stable ordering makes the cohort reproducible and avoids target-driven
    # selection.  Counts are checked after selection, not optimized from labels.
    eligible.sort(key=lambda group: hashlib.sha256((SEED + str(group[0]["molecule_id"])).encode()).hexdigest())
    train_pool = [g for g in eligible if g[0]["split"] == "train"]
    hold_pool = [g for g in eligible if g[0]["split"] == "holdout"]
    train_groups = train_pool[:TRAIN_MOLECULES]
    hold_groups = hold_pool[:HOLDOUT_MOLECULES]
    if len(train_groups) != TRAIN_MOLECULES or len(hold_groups) != HOLDOUT_MOLECULES:
        raise RuntimeError(f"insufficient eligible identities: train={len(train_groups)}, holdout={len(hold_groups)}")
    train_rows = [r for group in train_groups for r in group]
    hold_rows = [r for group in hold_groups for r in group]
    train_ids = {str(r["molecule_id"]) for r in train_rows}
    hold_ids = {str(r["molecule_id"]) for r in hold_rows}
    if train_ids & hold_ids:
        raise RuntimeError("train/holdout molecule overlap")
    identity_overlap = {}
    for field in ("molecule_id", "canonical_smiles", "connection_identity", "legacy_scaffold"):
        train_values = {canonical_key(r, field) for r in train_rows if canonical_key(r, field)}
        hold_values = {canonical_key(r, field) for r in hold_rows if canonical_key(r, field)}
        overlap = train_values & hold_values
        identity_overlap[field] = len(overlap)
        if overlap:
            raise RuntimeError(f"train/holdout {field} overlap: {sorted(overlap)[:3]}")
    all_sample_keys = [str(r["sample_key"]) for r in train_rows + hold_rows]
    if len(all_sample_keys) != len(set(all_sample_keys)):
        raise RuntimeError("duplicate sample_key in selected cohort")
    manifest_rows = []
    for split, selected in (("train", train_rows), ("holdout", hold_rows)):
        for row in selected:
            manifest_rows.append({
                key: row[key] for key in ("sample_key", "molecule_id", "canonical_smiles", "connection_identity", "legacy_scaffold", "profile_id")
            } | {"split": split, "metadata_only": True})
    manifest_rows.sort(key=lambda r: str(r["sample_key"]))
    # Ensure strata are genuinely present in both profiles before writing a
    # release candidate; this prevents a profile-balanced but chemically empty
    # cohort from being mistaken for bond-presence coverage.
    train_summary = summarize(train_rows)
    hold_summary = summarize(hold_rows)
    for summary, name in ((train_summary, "train"), (hold_summary, "holdout")):
        for profile in PROFILES:
            if summary["by_profile"].get(profile, {}).get("incident_visible_on_bonds", 0) <= 0:
                raise RuntimeError(f"{name}/{profile} has no visible O/N incident bond")
    manifest = {
        "schema_version": "chemvector-r17-oa13-oa33-bond-presence-manifest-0.1",
        "rows": manifest_rows,
        "protocol": {
            "metadata_only": True,
            "target_fields_excluded": ["layers", "graph", "coordinates", "target_atom_id", "nearest_distance_px"],
            "posthoc_audit_fields": ["visible_o", "visible_n", "incident_visible_on_bonds", "incident_visible_on_bonds_by_type"],
            "profiles": list(PROFILES),
            "identity_disjoint_from": [str(KNOWN.relative_to(ROOT)), str(SLICE.relative_to(ROOT)), str(STRESS.relative_to(ROOT)), str(CACHE.relative_to(ROOT))],
        },
    }
    report = {
        "schema_version": manifest["schema_version"], "complete": True,
        "read_only": True, "metadata_only": True, "training_performed": False,
        "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False,
        "release_manifest_sha256": digest(RELEASE), "cache_sha256": digest(CACHE),
        "excluded_manifests": {str(p.relative_to(ROOT)): digest(p) for p in (KNOWN, SLICE, STRESS)},
        "train_molecules": len(train_ids), "holdout_molecules": len(hold_ids),
        "identity_overlap": identity_overlap,
        "train_summary": train_summary, "holdout_summary": hold_summary,
        "gate": {"metadata_only": True, "identity_disjoint": True,
                 "oa13_and_oa33_present": True, "visible_on_bond_strata_present": True,
                 "training_admissible": False,
                 "next": "run frozen target-free graph replay; then a bounded CPU shadow with per-profile precision gates"},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["manifest_sha256"] = digest(OUT / "manifest.json")
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "report": report}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
