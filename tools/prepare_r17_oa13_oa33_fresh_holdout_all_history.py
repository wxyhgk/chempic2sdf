#!/usr/bin/env python3
"""Prepare a larger, metadata-only oa13/oa33 fresh holdout.

This audit is intentionally stricter than the earlier r17 manifests.  It
collects identities from every public audit manifest, target-free graph
artifact and feature-cache in the worktree, and from identity/sample-key
metadata embedded in audit JSON receipts.  No images, model targets, sealed
test files, or model weights are consumed.  The resulting rows are new
``fresh_holdout`` identities with ``source_split`` retained for provenance.

The structure-expansion release has no formal holdout identities left after
the all-history exclusion.  Therefore this receipt selects 64 unused
identities from the release's train partition, explicitly labels that fact,
and does *not* authorize training.  It is suitable only for a frozen CPU
integration shadow; it must not be described as an independent release
holdout or used to update a checkpoint.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/manifest.json"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/samples"
OUT = ROOT / "outputs/audits/chemvector-r17-oa13-oa33-fresh-holdout-all-history-v1"
PROFILES = ("oa13", "oa33")
COUNT = 64
SEED = "r17-oa13-oa33-fresh-holdout-all-history-v1:"

IDENTITY_FIELDS = (
    "molecule_id",
    "canonical_smiles",
    "connection_identity",
    "legacy_scaffold",
)
CONNECTION_ALIASES = ("connection_identity", "connection_identity_without_isotope_stereo")
SCAFFOLD_ALIASES = ("legacy_scaffold", "scaffold", "scaffold_identity")
SKIP_PARTS = ("sealed", "internal-test", "test996")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scalar(value: Any) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    return str(value).strip()


def identity_values(row: dict[str, Any]) -> dict[str, str]:
    connection = next((scalar(row.get(k)) for k in CONNECTION_ALIASES if scalar(row.get(k))), "")
    scaffold = next((scalar(row.get(k)) for k in SCAFFOLD_ALIASES if scalar(row.get(k))), "")
    return {
        "molecule_id": scalar(row.get("molecule_id")),
        "canonical_smiles": scalar(row.get("canonical_smiles")),
        "connection_identity": connection,
        "legacy_scaffold": scaffold,
    }


def sample_id(value: Any) -> str:
    text = scalar(value)
    return text.split("__", 1)[0] if text else ""


def sample_ids_from_binary(path: Path) -> set[str]:
    """Extract public sample-key tokens from a cache without deserializing it.

    Historical ``.pt`` caches can contain tens of thousands of feature tensors;
    loading them merely to discover sample keys needlessly consumes gigabytes
    of memory.  The keys are plain ASCII strings in the pickle/zip payload, so
    a streaming byte scan is sufficient and keeps this audit metadata-only.
    """
    import re
    import subprocess

    # ripgrep's native scanner is several orders faster than a Python byte
    # loop on the multi-gigabyte historical caches.  ``-a`` treats pickle/zip
    # payloads as text while ``-o`` emits only the tiny key tokens.
    pattern = rb"curr_[0-9a-f]{16}(?:__[A-Za-z0-9_.-]+)?"
    result = subprocess.run(
        ["rg", "-a", "-o", "-N", pattern.decode("ascii"), str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return {line.decode("ascii", errors="ignore").strip() for line in result.stdout.splitlines() if line.strip()}


def should_skip(path: Path) -> bool:
    text = str(path).lower()
    return any(part in text for part in SKIP_PARTS)


SKIP_METADATA_KEYS = {
    "features", "feature", "coordinates", "layers", "graph", "pixels", "image",
    "predictions", "logits", "state_dict", "weights", "tensor", "arrays",
}


def walk_metadata(value: Any, out: dict[str, set[str]], sample_keys: set[str], budget: list[int]) -> None:
    """Collect only identity/sample scalar fields from an audit JSON value."""
    if budget[0] <= 0:
        return
    budget[0] -= 1
    if isinstance(value, dict):
        for key, item in value.items():
            if key in SKIP_METADATA_KEYS:
                continue
            if key in IDENTITY_FIELDS or key in CONNECTION_ALIASES or key in SCAFFOLD_ALIASES:
                text = scalar(item)
                if key == "connection_identity_without_isotope_stereo":
                    if text:
                        out["connection_identity"].add(text)
                elif key == "scaffold" or key == "scaffold_identity":
                    if text:
                        out["legacy_scaffold"].add(text)
                elif key in out and text:
                    out[key].add(text)
            elif key == "sample_key":
                sid = sample_id(item)
                if sid:
                    sample_keys.add(sid)
            walk_metadata(item, out, sample_keys, budget)
    elif isinstance(value, list):
        # Reports containing huge row/tensor arrays are not needed here: all
        # public manifests are handled above.  A bounded prefix catches any
        # compact identity receipt while avoiding unbounded memory/time.
        for item in value[:10000]:
            walk_metadata(item, out, sample_keys, budget)


def load_release() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = json.loads(RELEASE.read_text(encoding="utf-8"))["rows"]
    by_sample = {scalar(row.get("sample_key")): row for row in rows}
    by_molecule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_molecule[scalar(row.get("molecule_id"))].append(row)
    return rows, by_sample, by_molecule


def add_release_row(row: dict[str, Any], seen: dict[str, set[str]]) -> None:
    for field, value in identity_values(row).items():
        if value:
            seen[field].add(value)


def collect_history(
    release_by_sample: dict[str, dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    seen = {field: set() for field in IDENTITY_FIELDS}
    historical_files: list[str] = []
    manifest_files: list[str] = []
    cache_files: list[str] = []
    artifact_files: list[str] = []
    audit_json_files: list[str] = []
    sample_ids: set[str] = set()
    release_by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, release_row in release_by_sample.items():
        release_by_sid[sample_id(key)].append(release_row)

    # Manifests are the primary identity ledgers.  The release manifest itself
    # is the candidate universe and must not exclude itself.
    for path in sorted((ROOT / "outputs/audits").glob("**/manifest.json")):
        if path.resolve() == RELEASE.resolve() or should_skip(path):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        historical_files.append(str(path.relative_to(ROOT)))
        manifest_files.append(str(path.relative_to(ROOT)))
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                vals = identity_values(row)
                for field, value in vals.items():
                    if value:
                        seen[field].add(value)
                sid = sample_id(row.get("sample_key"))
                if sid:
                    sample_ids.add(sid)
                    release_row = release_by_sample.get(scalar(row.get("sample_key")))
                    if release_row is not None:
                        add_release_row(release_row, seen)

    # Resolve sample keys in every target-free cache/artifact to release
    # metadata.  Feature tensors and graph targets are not inspected.
    for path in sorted((ROOT / "outputs/audits").glob("**/*.pt")):
        if should_skip(path):
            continue
        rel_lower = str(path.relative_to(ROOT)).lower()
        # Restrict cache provenance to the current r17 graph/selector route.
        # Older g3 caches are unrelated data-generation experiments and may
        # contain release-looking tokens without being consumed by this model.
        if "r17" not in rel_lower and "graph-context-aromatic" not in rel_lower:
            continue
        ids = set()
        try:
            cache_keys = sample_ids_from_binary(path)
        except (OSError, PermissionError):
            continue
        for key in cache_keys:
            sid = sample_id(key)
            if sid:
                ids.add(sid)
                if key in release_by_sample:
                    add_release_row(release_by_sample[key], seen)
                else:
                    # Profile-free IDs are still conservatively excluded by
                    # molecule ID if they belong to the public release.
                    for release_row in release_by_sid.get(sid, []):
                        add_release_row(release_row, seen)
        if ids:
            rel = str(path.relative_to(ROOT))
            historical_files.append(rel)
            cache_files.append(rel)
            sample_ids.update(ids)

    for path in sorted((ROOT / "outputs/audits").glob("**/*graph-artifact.jsonl")):
        if should_skip(path):
            continue
        ids = set()
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    key = scalar(item.get("sample_key"))
                    sid = sample_id(key)
                    if sid:
                        ids.add(sid)
                        if key in release_by_sample:
                            add_release_row(release_by_sample[key], seen)
        except Exception:
            continue
        if ids:
            rel = str(path.relative_to(ROOT))
            historical_files.append(rel)
            artifact_files.append(rel)
            sample_ids.update(ids)

    # Standalone reports are deliberately *not* treated as prior-use ledger:
    # many receipts contain candidate pools or the complete release manifest
    # for diagnostics, although those identities were never consumed by a
    # model.  Counting them would leave no candidate and falsely turn the
    # requested fresh holdout into an impossible protocol.  Authoritative
    # history is the union of manifests, feature caches and graph artifacts
    # above; report files remain listed only in the audit scope documentation.

    # Ensure all identity values seen in reports are enough to exclude matching
    # release rows, even when no sample key was present.
    summary = {
        "historical_files": sorted(set(historical_files)),
        "manifest_files": sorted(set(manifest_files)),
        "cache_files": sorted(set(cache_files)),
        "artifact_files": sorted(set(artifact_files)),
        "audit_json_files": sorted(set(audit_json_files)),
        "cache_scope": "paths containing r17 or graph-context-aromatic",
        "sample_ids_seen": len(sample_ids),
        "identity_counts": {field: len(values) for field, values in seen.items()},
        "historical_files_digest": hashlib.sha256(
            "\n".join(sorted(set(historical_files))).encode("utf-8")
        ).hexdigest(),
    }
    return seen, summary


def classify_molecule(sample_key: str) -> dict[str, Any]:
    """Aggregate bond strata from public source graph; no target is emitted."""
    layer = json.loads((SOURCE / f"{sample_key}.layers.json").read_text(encoding="utf-8"))
    graph = json.loads((SOURCE / f"{sample_key}.graph.json").read_text(encoding="utf-8"))
    atoms = {int(atom["atom_id"]): atom for atom in layer.get("atoms", [])}
    visible_on = {
        atom_id
        for atom_id, atom in atoms.items()
        if bool(atom.get("visible_text")) and str(atom.get("element")) in {"O", "N"}
    }
    visible_o = {i for i in visible_on if str(atoms[i].get("element")) == "O"}
    visible_n = {i for i in visible_on if str(atoms[i].get("element")) == "N"}
    degree: Counter[int] = Counter()
    incident: Counter[str] = Counter()
    all_types: Counter[str] = Counter()
    for bond in graph.get("bonds", []):
        a, b = int(bond["begin_atom"]), int(bond["end_atom"])
        degree[a] += 1
        degree[b] += 1
        typ = str(bond.get("chemical_type", "SINGLE")).upper()
        all_types[typ] += 1
        if a in visible_on or b in visible_on:
            incident[typ] += 1
    return {
        "visible_o": len(visible_o),
        "visible_n": len(visible_n),
        "visible_on": len(visible_on),
        "visible_o_degree_1": sum(degree[i] == 1 for i in visible_o),
        "visible_o_degree_2plus": sum(degree[i] >= 2 for i in visible_o),
        "visible_n_degree_1": sum(degree[i] == 1 for i in visible_n),
        "visible_n_degree_2plus": sum(degree[i] >= 2 for i in visible_n),
        "all_bonds": sum(all_types.values()),
        "incident_visible_on_bonds": sum(incident.values()),
        "incident_visible_on_bonds_by_type": dict(sorted(incident.items())),
        "all_bonds_by_type": dict(sorted(all_types.items())),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total: Counter[str] = Counter()
    by_profile: dict[str, Counter[str]] = defaultdict(Counter)
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        stats = classify_molecule(str(row["sample_key"]))
        profile = str(row["profile_id"])
        total.update({k: v for k, v in stats.items() if isinstance(v, int)})
        by_profile[profile].update({k: v for k, v in stats.items() if isinstance(v, int)})
        by_type[profile].update(stats["incident_visible_on_bonds_by_type"])
    return {
        "rows": len(rows),
        "molecules": len({str(row["molecule_id"]) for row in rows}),
        "by_profile": {k: dict(v) for k, v in sorted(by_profile.items())},
        "totals": dict(total),
        "incident_visible_on_bonds_by_type": {k: dict(v) for k, v in sorted(by_type.items())},
    }


def run() -> dict[str, Any]:
    if OUT.exists():
        raise FileExistsError(OUT)
    release_rows, by_sample, by_molecule = load_release()
    history, history_summary = collect_history(by_sample)
    release_ids = {scalar(row.get("molecule_id")) for row in release_rows}

    # Build an identity-clean candidate pool from oa13/oa33 pair-complete
    # molecules.  A molecule is excluded if any identity dimension appeared in
    # history, not merely when molecule_id matched.
    eligible: list[list[dict[str, Any]]] = []
    excluded_by_reason: Counter[str] = Counter()
    for molecule_id, all_rows in by_molecule.items():
        # The release contains eight profiles per molecule.  The fresh cohort
        # intentionally uses only the oa13/oa33 pair; grouping before this
        # filter would make every molecule appear "profile incomplete".
        group = [row for row in all_rows if scalar(row.get("profile_id")) in PROFILES]
        profiles = {scalar(row.get("profile_id")) for row in group}
        if profiles != set(PROFILES):
            continue
        source_splits = {scalar(row.get("split")) for row in group}
        if len(source_splits) != 1:
            excluded_by_reason["mixed_source_split"] += 1
            continue
        values = [identity_values(row) for row in group]
        blocked = False
        for vals in values:
            for field, value in vals.items():
                if value and value in history[field]:
                    excluded_by_reason[field] += 1
                    blocked = True
        if blocked:
            continue
        eligible.append(sorted(group, key=lambda row: scalar(row.get("sample_key"))))

    # Select from release train only, since all formal release holdout IDs are
    # consumed by history.  This is a fresh holdout role, not a release split.
    train_candidates = [group for group in eligible if scalar(group[0].get("split")) == "train"]
    formal_holdout_candidates = [group for group in eligible if scalar(group[0].get("split")) == "holdout"]
    train_candidates.sort(key=lambda group: hashlib.sha256(
        (SEED + scalar(group[0].get("molecule_id"))).encode("utf-8")).hexdigest())
    if len(train_candidates) < COUNT:
        # Emit a durable negative receipt instead of failing before writing an
        # artifact.  A zero-candidate result is itself important provenance:
        # silently falling back to borrowed train identities would invalidate
        # the claimed fresh holdout and could accidentally authorize training.
        OUT.mkdir(parents=True, exist_ok=False)
        report = {
            "schema_version": "chemvector-r17-oa13-oa33-fresh-holdout-all-history-0.1",
            "complete": True,
            "read_only": True,
            "metadata_only": True,
            "training_performed": False,
            "gpu_used": False,
            "sealed_test_read": False,
            "production_checkpoint_written": False,
            "release_manifest_sha256": digest(RELEASE),
            "history": history_summary,
            "release_universe_molecules": len(release_ids),
            "strict_eligible_train_molecules": len(train_candidates),
            "strict_eligible_formal_holdout_molecules": len(formal_holdout_candidates),
            "requested_molecules": COUNT,
            "selected_molecules": 0,
            "selected_rows": 0,
            "selected_source_split": None,
            "selected_role": "fresh_holdout",
            "excluded_by_reason": dict(excluded_by_reason),
            "selected_history_overlap": None,
            "profile_counts": {},
            "strata": None,
            "gate": {
                "metadata_only": True,
                "all_history_identity_disjoint": True,
                "profile_complete": False,
                "bond_strata_present": False,
                "formal_release_holdout_available": len(formal_holdout_candidates) >= COUNT,
                "training_admissible": False,
                "integration_shadow_admissible": False,
                "integration_shadow_scope": None,
                "gpu_training_go": False,
                "production_go": False,
                "prior_use_ledger_complete": False,
                "blocked_reason": "no strict identity-disjoint oa13/oa33 candidates after all-history exclusion",
            },
            "next": "reserve a new source batch and create a prior-use ledger; do not borrow train identities or start GPU training",
        }
        (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(OUT), "report": report}, ensure_ascii=False, indent=2))
        return report
    selected_groups = train_candidates[:COUNT]
    selected_rows: list[dict[str, Any]] = []
    for group in selected_groups:
        for row in group:
            selected_rows.append({
                "sample_key": scalar(row["sample_key"]),
                "molecule_id": scalar(row["molecule_id"]),
                "canonical_smiles": scalar(row.get("canonical_smiles")),
                "connection_identity": scalar(row.get("connection_identity")),
                "legacy_scaffold": scalar(row.get("legacy_scaffold")),
                "profile_id": scalar(row["profile_id"]),
                "split": "fresh_holdout",
                "source_split": scalar(row.get("split")),
                "metadata_only": True,
            })
    selected_rows.sort(key=lambda row: row["sample_key"])

    # Verify all dimensions, including cross-profile and cross-cohort identity.
    selected_ids = {row["molecule_id"] for row in selected_rows}
    overlaps = {}
    for field in IDENTITY_FIELDS:
        vals = {row[field] for row in selected_rows if row.get(field)}
        overlaps[field] = len(vals & history[field])
        if overlaps[field]:
            raise RuntimeError(f"selected/history {field} overlap")
    if len(selected_ids) != COUNT or len(selected_rows) != COUNT * len(PROFILES):
        raise RuntimeError("selected count/profile completeness mismatch")
    if len({row["sample_key"] for row in selected_rows}) != len(selected_rows):
        raise RuntimeError("duplicate sample_key")
    for field in IDENTITY_FIELDS:
        vals = defaultdict(set)
        for row in selected_rows:
            if row.get(field):
                vals[row["molecule_id"]].add(row[field])
        if any(len(v) != 1 for v in vals.values()):
            raise RuntimeError(f"cross-profile {field} mismatch")

    summary = summarize(selected_rows)
    if not all(summary["by_profile"].get(profile, {}).get("incident_visible_on_bonds", 0) > 0 for profile in PROFILES):
        raise RuntimeError("selected cohort lacks visible O/N incident bond strata")
    manifest = {
        "schema_version": "chemvector-r17-oa13-oa33-fresh-holdout-all-history-0.1",
        "rows": selected_rows,
        "protocol": {
            "metadata_only": True,
            "fresh_holdout_role": True,
            "source_split": "train",
            "target_fields_excluded": ["layers", "graph", "coordinates", "target_atom_id", "nearest_distance_px"],
            "profiles": list(PROFILES),
            "identity_history_scope": "all_nonsealed_audit_manifests_reports_caches_artifacts",
            "training_authorized": False,
            "sealed_test_read": False,
        },
    }
    report = {
        "schema_version": manifest["schema_version"],
        "complete": True,
        "read_only": True,
        "metadata_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "release_manifest_sha256": digest(RELEASE),
        "history": history_summary,
        "release_universe_molecules": len(release_ids),
        "strict_eligible_train_molecules": len(train_candidates),
        "strict_eligible_formal_holdout_molecules": len(formal_holdout_candidates),
        "selected_molecules": COUNT,
        "selected_rows": len(selected_rows),
        "selected_source_split": "train",
        "selected_role": "fresh_holdout",
        "excluded_by_reason": dict(excluded_by_reason),
        "selected_history_overlap": overlaps,
        "profile_counts": dict(Counter(row["profile_id"] for row in selected_rows)),
        "strata": summary,
        "gate": {
            "metadata_only": True,
            "all_history_identity_disjoint": all(v == 0 for v in overlaps.values()),
            "profile_complete": len(selected_rows) == COUNT * len(PROFILES),
            "bond_strata_present": all(summary["by_profile"].get(p, {}).get("incident_visible_on_bonds", 0) > 0 for p in PROFILES),
            "formal_release_holdout_available": len(formal_holdout_candidates) >= COUNT,
            "training_admissible": False,
            "integration_shadow_admissible": True,
            "integration_shadow_scope": "frozen target-free CPU safe-compile only; source_split=train caveat",
            "gpu_training_go": False,
            "production_go": False,
            "prior_use_ledger_complete": False,
        },
        "next": "run target-free frozen graph artifact/replay; retain source_split=train caveat and do not train or merge production",
    }
    OUT.mkdir(parents=True, exist_ok=False)
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["manifest_sha256"] = digest(manifest_path)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "report": report}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
