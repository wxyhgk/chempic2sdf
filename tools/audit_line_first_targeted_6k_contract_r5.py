"""Read-only contract audit for the 6k targeted quarantine panel."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-targeted-6k-manifest-r5/manifest.json"
DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-6k-r5-parallel"
SOURCE = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-targeted-6k-contract-r5-v2"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
REQUIRED = (".png", ".svg", ".layers.json", ".text-records.json", ".text.png", ".bond.png", ".auxiliary.png", ".semantic.png", ".graph.json", ".sdf", ".meta.json")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    summary = json.loads((DATA / "summary.json").read_text())
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    source_rows = [json.loads(line) for line in (SOURCE / "samples.jsonl").read_text().splitlines() if line.strip()]
    source_ids = {row["molecule_id"] for row in source_rows}
    row_ids = {row["sample_key"] for row in rows}
    required_ok = 0
    hash_ok = 0
    graph_ok = 0
    sdf_ok = 0
    failures = []
    for row in rows:
        base = DATA / "samples" / row["sample_key"]
        missing = [suffix for suffix in REQUIRED if not base.with_suffix(suffix).exists()]
        if missing:
            failures.append({"sample_key": row["sample_key"], "kind": "missing", "fields": missing})
            continue
        required_ok += 1
        actual_hashes = {suffix: sha(base.with_suffix(suffix)) for suffix in REQUIRED}
        expected = {suffix: row.get("sha256", {}).get(suffix) for suffix in REQUIRED}
        if actual_hashes == expected:
            hash_ok += 1
        else:
            failures.append({"sample_key": row["sample_key"], "kind": "hash"})
        try:
            graph = json.loads(base.with_suffix(".graph.json").read_text())
            if graph.get("molecule_id") == row["molecule_id"] and graph.get("canonical_smiles") == row["canonical_smiles"]:
                graph_ok += 1
            else:
                failures.append({"sample_key": row["sample_key"], "kind": "graph_identity"})
            mol = Chem.MolFromMolFile(str(base.with_suffix(".sdf")), sanitize=True, removeHs=False, strictParsing=True)
            if mol is not None and Chem.MolToSmiles(mol) == Chem.MolToSmiles(Chem.MolFromSmiles(row["canonical_smiles"])):
                sdf_ok += 1
            else:
                failures.append({"sample_key": row["sample_key"], "kind": "sdf_identity"})
        except Exception as exc:
            failures.append({"sample_key": row["sample_key"], "kind": "parse", "error": str(exc)})
    split_counts = Counter(row["split"] for row in rows)
    profile_counts = Counter(row["profile_id"] for row in rows)
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-targeted-6k-contract-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False,
        "initial_sha256": INITIAL_SHA, "manifest_sha256": sha(MANIFEST),
        "dataset_summary": summary, "rows": len(rows), "unique_sample_keys": len(row_ids),
        "split_counts": dict(sorted(split_counts.items())), "profile_counts": dict(sorted(profile_counts.items())),
        "required_files_ok": required_ok, "hash_rows_ok": hash_ok,
        "graph_identity_ok": graph_ok, "sdf_identity_ok": sdf_ok,
        "source_molecule_overlap": len({row["molecule_id"] for row in rows} & source_ids),
        "failures": failures[:100], "failure_count": len(failures),
        "gate": {
            "manifest_planned_6000": manifest.get("planned_images") == 6000,
            "render_complete": summary.get("complete") is True and summary.get("accepted_images", 0) == len(rows),
            "identity_disjoint": not bool({row["molecule_id"] for row in rows} & source_ids),
            "required_files_complete": required_ok == len(rows),
            "hashes_complete": hash_ok == len(rows),
            "graph_identity_complete": graph_ok == len(rows),
            "sdf_identity_complete": sdf_ok == len(rows),
            "sealed_test_read": False,
            "training_authorized": False,
            "endpoint_ceiling_admissible": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "rows": len(rows), "failures": len(failures), "gate": report["gate"]}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
