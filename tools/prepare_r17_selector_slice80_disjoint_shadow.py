#!/usr/bin/env python3
"""Prepare a small identity-clean shadow cohort excluding the public slice80."""
from __future__ import annotations
from collections import defaultdict
import hashlib, json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs/audits/chemvector-structure-expansion-10k-release-plan-20261016-r1/manifest.json"
MOLECULES = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923/molecules.json"
PLAN = ROOT / "outputs/audits/chemvector-aromatic-specialist-plan-r17-v2/rows.jsonl"
SLICE = ROOT / "outputs/audits/chemvector-r17-selector-graph-replay-10k-slice80-aromatic/report.json"
OUT = ROOT / "outputs/audits/chemvector-r17-selector-slice80-disjoint-shadow-v1"
TRAIN_N, HOLDOUT_N = 32, 16

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ident(row):
    return (str(row.get("canonical_smiles") or ""), str(row.get("connection_identity") or row.get("connection_identity_without_isotope_stereo") or ""), str(row.get("legacy_scaffold") or row.get("scaffold") or ""))
def rank(prefix, mid): return hashlib.sha256((prefix + str(mid)).encode()).hexdigest()

def run():
    if OUT.exists(): raise FileExistsError(OUT)
    release = json.loads(RELEASE.read_text())["rows"]
    molecules = {str(r["molecule_id"]): r for r in json.loads(MOLECULES.read_text())}
    known_ids, known_fields = set(), [set(), set(), set()]
    for line in PLAN.read_text().splitlines():
        if line.strip():
            r = json.loads(line); known_ids.add(str(r["molecule_id"]))
            for i, value in enumerate(ident(r)):
                if value: known_fields[i].add(value)
    for path in ROOT.glob("outputs/audits/chemvector-graph-context-aromatic-cpu-r17-*/feature-cache.pt"):
        cache = torch.load(path, map_location="cpu", weights_only=False)
        for records in cache.get("records", {}).values():
            for r in records:
                if r.get("sample_key"): known_ids.add(str(r["sample_key"]).rsplit("__", 1)[0])
    release_by_key = {r["sample_key"]: r for r in release}
    slice_ids = {str(release_by_key[r["sample_key"]]["molecule_id"]) for r in json.loads(SLICE.read_text())["rows"] if r["sample_key"] in release_by_key}
    known_ids |= slice_ids
    by = defaultdict(list)
    for r in release: by[str(r["molecule_id"])].append(r)
    clean = []
    for mid, rows in by.items():
        if mid in known_ids or len(rows) != 8: continue
        meta = molecules.get(mid, {}); values = (str(meta.get("canonical_smiles") or rows[0].get("canonical_smiles") or ""), str(meta.get("connection_identity_without_isotope_stereo") or rows[0].get("connection_identity") or ""), str(meta.get("scaffold") or rows[0].get("legacy_scaffold") or ""))
        if any(v and v in known_fields[i] for i, v in enumerate(values)): continue
        clean.append((mid, rows, values))
    train = [x for x in sorted(clean, key=lambda x: rank("r17-slice80-disjoint-train:", x[0])) if x[1][0]["split"] == "train"][:TRAIN_N]
    hold = [x for x in sorted(clean, key=lambda x: rank("r17-slice80-disjoint-holdout:", x[0])) if x[1][0]["split"] == "holdout"][:HOLDOUT_N]
    if len(train) != TRAIN_N or len(hold) != HOLDOUT_N: raise RuntimeError(f"insufficient clean cohort: {len(train)}/{len(hold)}")
    def vals(items, i): return {x[2][i] for x in items if x[2][i]}
    overlap = {name: len(vals(train, i) & vals(hold, i)) for i, name in enumerate(("canonical_smiles", "connection_identity", "scaffold"))}
    if any(overlap.values()): raise RuntimeError(f"split overlap: {overlap}")
    rows = []
    for split, items in (("train", train), ("holdout", hold)):
        for _, group, _ in items:
            rows.extend({**r, "split": split, "slice80_disjoint_shadow": True} for r in sorted(group, key=lambda x: x["sample_key"]))
    report = {"schema_version": "chemvector-r17-selector-slice80-disjoint-shadow-0.1", "complete": True, "read_only": True, "training_performed": False, "training_authorized": False, "sealed_test_read": False, "production_checkpoint_written": False, "excluded_slice80_molecules": len(slice_ids), "known_excluded_molecules": len(known_ids), "clean_candidates": len(clean), "selection": {"train_molecules": TRAIN_N, "holdout_molecules": HOLDOUT_N, "images": len(rows)}, "identity_overlap": overlap, "inputs": {"release_sha256": digest(RELEASE), "slice80_sha256": digest(SLICE), "plan_sha256": digest(PLAN)}, "gate": {"slice80_excluded": True, "known_plan_and_cache_excluded": True, "train_holdout_identity_disjoint": not any(overlap.values()), "historical_ledger_complete": False, "cpu_shadow_only": True, "gpu_training_go": False}}
    OUT.mkdir(parents=False); (OUT / "manifest.json").write_text(json.dumps({"schema_version": report["schema_version"], "rows": rows}, indent=2, ensure_ascii=False) + "\n"); (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n"); print(json.dumps({"output": str(OUT), "selection": report["selection"], "overlap": overlap, "gate": report["gate"]}, indent=2, ensure_ascii=False))

if __name__ == "__main__": run()
