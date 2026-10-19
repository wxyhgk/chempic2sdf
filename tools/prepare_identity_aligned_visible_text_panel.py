"""Prepare an identity-disjoint O/N visible-text panel and render a tiny smoke.

This is quarantine metadata/render work only.  It does not read a training or
evaluation corpus, load a model, or authorize training.  Full rendering is a
separate explicit step after reviewing this plan and smoke report.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from Data.sdf.chemvector_curriculum_v0_1.rendering import Profile, render
from Data.sdf.chemvector_curriculum_v0_2.catalog import entries


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r1"
TRAIN_MOLECULES = 120
HOLDOUT_MOLECULES = 80
SEED = "chemvector-identity-aligned-visible-text-20261017-r1:"
PROFILES = (
    Profile("cov13_base", font_size=32, bond_length_cap=28.0, bond_line_width=3.2, rotation_degrees=180.0, font_family="DejaVu Serif"),
    Profile("cov13_cap52", font_size=32, bond_length_cap=52.0, bond_line_width=3.2, rotation_degrees=180.0, font_family="DejaVu Serif"),
    Profile("cov13_stroke16", font_size=32, bond_length_cap=28.0, bond_line_width=1.6, rotation_degrees=180.0, font_family="DejaVu Serif"),
    Profile("cov33_base", font_size=32, bond_length_cap=52.0, bond_line_width=1.6, rotation_degrees=0.0, font_family="DejaVu Sans"),
    Profile("cov33_cap28", font_size=32, bond_length_cap=28.0, bond_line_width=1.6, rotation_degrees=0.0, font_family="DejaVu Sans"),
    Profile("cov33_stroke32", font_size=32, bond_length_cap=52.0, bond_line_width=3.2, rotation_degrees=0.0, font_family="DejaVu Sans"),
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable(value: str) -> str:
    return hashlib.sha256((SEED + value).encode()).hexdigest()


def source_bindings() -> dict[str, str]:
    paths = [Path("Data/sdf/chemvector_curriculum_v0_2/catalog.py"),
             Path("Data/sdf/chemvector_curriculum_v0_2/atoms_bonds.py"),
             Path("Data/sdf/chemvector_curriculum_v0_2/functional_contexts.py"),
             Path("Data/sdf/chemvector_curriculum_v0_1/rendering.py"),
             Path("Data/sdf/chemvector_curriculum_v0_1/typography.py"),
             Path("Data/sdf/text_source_renderer/v2/renderer.py")]
    return {str(path): digest(path) for path in paths}


def candidates() -> list[dict]:
    result = []
    for entry in entries():
        mol = Chem.MolFromSmiles(entry["canonical_smiles"])
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            continue
        visible_elements = {atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() in {"O", "N"}}
        # Every selected identity carries both labels, so each style profile
        # contributes to both required O and N strata.
        if not {"O", "N"}.issubset(visible_elements):
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        result.append({
            "molecule_id": entry["molecule_id"], "canonical_smiles": entry["canonical_smiles"],
            "connection_identity": entry.get("connection_identity"), "scaffold": scaffold,
            "visible_elements": sorted(visible_elements), "source_record_ids": entry["source_record_ids"],
            "entry": entry,
        })
    return result


def choose_split(pool: list[dict], target: int, used_scaffolds: set[str]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        if row["scaffold"] not in used_scaffolds:
            groups[row["scaffold"]].append(row)
    selected: list[dict] = []
    for scaffold in sorted(groups, key=stable):
        members = sorted(groups[scaffold], key=lambda row: stable(row["molecule_id"]))
        if len(selected) >= target:
            break
        selected.extend(members[: target - len(selected)])
        used_scaffolds.add(scaffold)
    if len(selected) < target:
        raise RuntimeError(f"insufficient scaffold-disjoint molecules: wanted={target} got={len(selected)}")
    return selected


def build_plan() -> dict:
    pool = candidates()
    used: set[str] = set()
    holdout = choose_split(pool, HOLDOUT_MOLECULES, used)
    train = choose_split([r for r in pool if r["molecule_id"] not in {x["molecule_id"] for x in holdout}], TRAIN_MOLECULES, used)
    if {r["scaffold"] for r in train} & {r["scaffold"] for r in holdout}:
        raise AssertionError("scaffold overlap")
    rows = []
    for split, selected in (("train", train), ("holdout", holdout)):
        for row in sorted(selected, key=lambda x: x["molecule_id"]):
            for profile in PROFILES:
                rows.append({"sample_key": f"{row['molecule_id']}__{profile.profile_id}",
                             "molecule_id": row["molecule_id"], "canonical_smiles": row["canonical_smiles"],
                             "connection_identity": row["connection_identity"], "scaffold": row["scaffold"],
                             "visible_elements": row["visible_elements"], "split": split,
                             "profile_id": profile.profile_id, "render_status": "not_rendered"})
    return {
        "schema_version": "chemvector-identity-aligned-visible-text-panel-plan-0.1",
        "status": "quarantine_metadata_and_smoke_only", "complete": True,
        "training_authorized": False, "gpu_used": False, "model_forwards": 0,
        "optimizer_steps": 0, "sealed_test_read": False, "external_corpora_read": False,
        "selection": {"catalog": "chemvector-curriculum-v0.2", "candidate_count": len(pool),
                       "train_molecules": len(train), "holdout_molecules": len(holdout),
                       "scaffold_overlap": 0, "profiles": [asdict(p) for p in PROFILES],
                       "planned_images": len(rows)},
        "source_bindings": source_bindings(), "rows": rows,
        "identity_contract": {"elements": ["O", "N"], "target_anchor_separation_px": 48,
                               "candidate_target_radius_px": 16, "ambiguous_rows_masked": True,
                               "hungarian_assignment_forbidden": True},
    }


def smoke(plan: dict) -> dict:
    smoke_dir = OUT / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=False)
    by_id = {r["molecule_id"]: r for r in candidates()}
    selected = []
    for split in ("train", "holdout"):
        selected.append(next(r for r in plan["rows"] if r["split"] == split)["molecule_id"])
    selected_ids = list(dict.fromkeys(selected))
    results = []
    for molecule_id in selected_ids:
        entry = by_id[molecule_id]["entry"]
        for profile in PROFILES:
            row = next(r for r in plan["rows"] if r["molecule_id"] == molecule_id and r["profile_id"] == profile.profile_id)
            try:
                files, meta = render(entry, profile, typography="stereo-aware-v1")
                for suffix in (".png", ".layers.json", ".meta.json"):
                    (smoke_dir / f"{row['sample_key']}{suffix}").write_bytes(files[suffix])
                layers = json.loads(files[".layers.json"])
                visible = sum(bool(atom.get("visible_text")) and atom.get("element") in {"O", "N"} for atom in layers["atoms"])
                results.append({"sample_key": row["sample_key"], "split": row["split"], "profile_id": profile.profile_id,
                                "status": "accepted", "visible_on": visible,
                                "png_sha256": hashlib.sha256(files[".png"]).hexdigest()})
            except ValueError as exc:
                results.append({"sample_key": row["sample_key"], "split": row["split"], "profile_id": profile.profile_id,
                                "status": "quality_rejected", "reason": str(exc)})
    accepted = [r for r in results if r["status"] == "accepted"]
    return {"schema_version": "chemvector-identity-aligned-visible-text-smoke-0.1", "complete": True,
            "training_authorized": False, "gpu_used": False, "sealed_test_read": False,
            "planned_images": len(results), "accepted_images": len(accepted), "rejected_images": len(results) - len(accepted),
            "accepted_with_visible_on": sum(r.get("visible_on", 0) > 0 for r in accepted), "results": results}


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = build_plan()
    OUT.mkdir(parents=True)
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    result = smoke(plan)
    (OUT / "smoke.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    report = {"schema_version": "chemvector-identity-aligned-visible-text-preparation-0.1", "complete": True,
              "training_authorized": False, "gpu_used": False, "sealed_test_read": False,
              "plan_sha256": digest(OUT / "plan.json"), "smoke_sha256": digest(OUT / "smoke.json"),
              "selection": plan["selection"], "smoke": {k: result[k] for k in ("planned_images", "accepted_images", "rejected_images", "accepted_with_visible_on")},
              "next": "review smoke and identity ledger; no training or full render until explicit data release"}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
