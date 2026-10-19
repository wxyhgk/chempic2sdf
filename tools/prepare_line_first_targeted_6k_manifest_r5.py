"""Prepare an identity-disjoint 6k-image line-first targeted manifest.

This is metadata only: it does not render pixels, run a model, inspect sealed
identities, or authorize training.  The manifest is deliberately stratified
around the measured reachability failures (text seams, multiple bonds, short
and oblique bonds, and weak cov13/cov33 styles).
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data.sdf.chemvector_curriculum_v0_2.catalog import entries
from tools.prepare_line_first_valence_targeted_manifest_r5 import (
    connection_identity,
    graph_strata,
    smiles_graph,
)

SOURCE = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-targeted-6k-manifest-r5"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"

STYLES = (
    {"profile_id": "cov13_b32_r30", "font_size": 32, "bond_length_cap": 32.0, "bond_line_width": 3.2, "rotation_degrees": 30.0},
    {"profile_id": "cov13_b40_r45", "font_size": 32, "bond_length_cap": 40.0, "bond_line_width": 3.2, "rotation_degrees": 45.0},
    {"profile_id": "cov13_b48_r60", "font_size": 32, "bond_length_cap": 48.0, "bond_line_width": 3.2, "rotation_degrees": 60.0},
    {"profile_id": "cov13_stroke16_r30", "font_size": 32, "bond_length_cap": 40.0, "bond_line_width": 1.6, "rotation_degrees": 30.0},
    {"profile_id": "cov13_cap52_r45", "font_size": 32, "bond_length_cap": 52.0, "bond_line_width": 3.2, "rotation_degrees": 45.0},
    {"profile_id": "cov33_cap32_r60", "font_size": 32, "bond_length_cap": 32.0, "bond_line_width": 3.2, "rotation_degrees": 60.0},
)
QUOTAS = {
    "nonring_aromatic_bridge": 100,
    "multiple_bond": 250,
    "NO_multiple_bond": 200,
    "heteroatom_valence": 200,
    "junction_degree3plus": 150,
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = [json.loads(line) for line in (SOURCE / "samples.jsonl").read_text().splitlines() if line.strip()]
    source_connections = {str(row.get("connection_identity")) for row in rows if row.get("connection_identity")}
    source_molecules = {str(row.get("molecule_id")) for row in rows if row.get("molecule_id")}
    pool = {}
    for entry in entries():
        ci = connection_identity(entry["canonical_smiles"])
        if ci in source_connections or ci in pool:
            continue
        tags, detail = graph_strata(smiles_graph(entry["canonical_smiles"]))
        pool[ci] = {
            "molecule_id": entry["molecule_id"],
            "canonical_smiles": entry["canonical_smiles"],
            "connection_identity": ci,
            "tags": sorted(set(tags)),
            "detail": detail,
        }
    ordered = sorted(pool.values(), key=lambda item: stable_key(item["connection_identity"]))
    selected: list[dict] = []
    used: set[str] = set()
    for tag, quota in QUOTAS.items():
        count = 0
        for item in ordered:
            if item["connection_identity"] in used or tag not in item["tags"]:
                continue
            selected.append({**item, "selection_stratum": tag})
            used.add(item["connection_identity"])
            count += 1
            if count >= quota:
                break
    for item in ordered:
        if len(selected) >= 1000:
            break
        if item["connection_identity"] not in used:
            selected.append({**item, "selection_stratum": "mixed_residual"})
            used.add(item["connection_identity"])
    if len(selected) != 1000:
        raise RuntimeError(f"expected 1000 selected molecules, got {len(selected)}")
    selected.sort(key=lambda item: stable_key(item["connection_identity"]))
    for index, item in enumerate(selected):
        item["split"] = "train" if index < 800 else ("dev" if index < 900 else "fresh_holdout")
        item["target_profiles"] = [style["profile_id"] for style in STYLES]
    split_counts = Counter(item["split"] for item in selected)
    stratum_counts = Counter(item["selection_stratum"] for item in selected)
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-targeted-6k-manifest-r5-0.1",
        "complete": True,
        "metadata_only": True,
        "status": "quarantine_plan_only",
        "images_rendered": False,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "initial_sha256": INITIAL_SHA,
        "source": str(SOURCE),
        "source_summary_sha256": digest(SOURCE / "summary.json"),
        "source_samples_sha256": digest(SOURCE / "samples.jsonl"),
        "source_molecule_count": len(source_molecules),
        "source_connection_count": len(source_connections),
        "candidate_pool": len(pool),
        "selected_molecules": len(selected),
        "planned_images": len(selected) * len(STYLES),
        "split_counts": dict(sorted(split_counts.items())),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "styles": list(STYLES),
        "identity_disjoint_from_fixed_source": not bool(used & source_connections),
        "items": selected,
        "gate": {
            "identity_disjoint": not bool(used & source_connections),
            "fixed_split": split_counts == Counter({"train": 800, "dev": 100, "fresh_holdout": 100}),
            "style_count": len(STYLES) == 6,
            "planned_images_6000": len(selected) * len(STYLES) == 6000,
            "metadata_only": True,
            "sealed_test_read": False,
            "training_authorized": False,
            "next": "render quarantine panel, run exact-byte/style/roundtrip and endpoint ceiling audits before any CPU fit",
        },
    }
    OUT.mkdir(parents=True)
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (OUT / "items.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in selected))
    print(json.dumps({key: report[key] for key in ("candidate_pool", "selected_molecules", "planned_images", "split_counts", "stratum_counts", "gate")}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
