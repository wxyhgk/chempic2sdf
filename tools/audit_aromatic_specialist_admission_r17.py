#!/usr/bin/env python3
"""Audit r17 identity admission and source render/graph/SDF integrity.

This is the required pre-training boundary for the r17 quarantine slice.  It
reads selected source artifacts and existing identity-review outputs, but never
opens sealed evaluation samples, runs a model, or authorizes training.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "outputs/audits/chemvector-aromatic-specialist-plan-r17-v2/rows.jsonl"
SOURCE = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
PRIVATE_REMAINING = ROOT / "outputs/audits/structure-expansion-identity-partition-20260923/remaining-candidates.json"
PUBLIC_REMAINING = ROOT / "outputs/audits/structure-expansion-public-identity-20260923/remaining-candidates.json"
OUT = ROOT / "outputs/audits/chemvector-aromatic-specialist-admission-r17"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    rows = [json.loads(line) for line in PLAN.read_text().splitlines() if line.strip()]
    molecules = {row["molecule_id"]: row for row in json.loads((SOURCE / "molecules.json").read_text())}
    source_hashes = json.loads((SOURCE / "sample-sha256.json").read_text())
    private = {row["molecule_id"] for row in json.loads(PRIVATE_REMAINING.read_text())}
    public = {row["molecule_id"] for row in json.loads(PUBLIC_REMAINING.read_text())}
    selected_ids = {row["molecule_id"] for row in rows}
    missing_private = sorted(selected_ids - private)
    missing_public = sorted(selected_ids - public)
    missing_source = sorted(selected_ids - set(molecules))
    rows_checked = 0
    bad_files = []
    bad_hashes = []
    bad_labels = []
    bad_sdf = []
    for row in rows:
        mid, key = row["molecule_id"], row["sample_key"]
        molecule = molecules.get(mid)
        if molecule is None or molecule.get("split") != "quarantine" or molecule.get("training_authorized") is not False:
            bad_labels.append({"sample_key": key, "reason": "molecule quarantine contract"})
        for suffix in (".png", ".layers.json", ".graph.json", ".sdf"):
            path = SOURCE / "samples" / f"{key}{suffix}"
            if not path.exists():
                bad_files.append(str(path)); continue
            expected = source_hashes.get(f"samples/{key}{suffix}")
            if expected is not None and sha(path) != expected:
                bad_hashes.append(str(path))
        layers_path = SOURCE / "samples" / f"{key}.layers.json"
        graph_path = SOURCE / "samples" / f"{key}.graph.json"
        sdf_path = SOURCE / "samples" / f"{key}.sdf"
        if layers_path.exists() and graph_path.exists():
            layers = json.loads(layers_path.read_text())
            graph = json.loads(graph_path.read_text())
            if layers.get("molecule_id") != mid or layers.get("profile_id") != row["profile_id"]:
                bad_labels.append({"sample_key": key, "reason": "layers identity"})
            if layers.get("training_authorized") is not False:
                bad_labels.append({"sample_key": key, "reason": "layers training flag"})
            if len(layers.get("atoms", [])) == 0 or len(graph.get("bonds", [])) == 0:
                bad_labels.append({"sample_key": key, "reason": "empty graph/layers"})
        if sdf_path.exists():
            mol = Chem.MolFromMolFile(str(sdf_path), sanitize=True, removeHs=False)
            if mol is None:
                bad_sdf.append({"sample_key": key, "reason": "SDF parse"})
            else:
                actual = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                expected = Chem.MolToSmiles(Chem.MolFromSmiles(row["canonical_smiles"]), canonical=True, isomericSmiles=True)
                if actual != expected:
                    bad_sdf.append({"sample_key": key, "reason": "canonical identity", "actual": actual, "expected": expected})
        rows_checked += 1
    split_ids = {split: {row["molecule_id"] for row in rows if row["split"] == split} for split in ("train", "dev", "fresh_holdout")}
    overlaps = {f"{a}:{b}": len(split_ids[a] & split_ids[b]) for a, b in (("train", "dev"), ("train", "fresh_holdout"), ("dev", "fresh_holdout"))}
    report = {
        "schema_version": "chemvector-aromatic-specialist-admission-r17-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "plan": str(PLAN.relative_to(ROOT)), "source": str(SOURCE.relative_to(ROOT)),
        "rows_checked": rows_checked, "image_counts": dict(Counter(row["split"] for row in rows)),
        "molecule_counts": {split: len(ids) for split, ids in split_ids.items()},
        "identity_review": {"private_remaining": len(private), "public_remaining": len(public), "selected_missing_private": missing_private, "selected_missing_public": missing_public, "selected_missing_source": missing_source},
        "split_overlaps": overlaps, "bad_files": bad_files, "bad_hashes": bad_hashes, "bad_labels": bad_labels, "bad_sdf": bad_sdf,
        "gate": {
            "private_identity_admitted": not missing_private,
            "public_identity_admitted": not missing_public,
            "source_membership": not missing_source,
            "split_identity_disjoint": all(value == 0 for value in overlaps.values()),
            "render_hash_contract": not bad_files and not bad_hashes,
            "graph_sdf_contract": not bad_labels and not bad_sdf,
            "sealed_test_read": False, "training_admissible": False, "production_go": False,
            "next": "separate training authorization required; run coverage report before CPU sidecar",
        },
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "rows_checked": rows_checked, "bad_files": len(bad_files), "bad_hashes": len(bad_hashes), "bad_sdf": len(bad_sdf), "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
