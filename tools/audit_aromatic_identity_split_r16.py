#!/usr/bin/env python3
"""Read-only audit for an identity-stratified aromatic split proposal.

The current r6 rows are deliberately *not* used as an aromatic evaluation
split: their molecule-level ordering put every fresh molecule in the
hetero-multiple control stratum.  This audit reports the starvation and the
available deterministic catalog pool so a replacement manifest can be built
before any rendering or CPU fitting.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Data.sdf.chemvector_curriculum_v0_2.catalog import entries
from tools.prepare_line_first_valence_targeted_manifest_r5 import connection_identity


def category(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    aromatic_atoms = [a for a in mol.GetAtoms() if a.GetIsAromatic()]
    if not aromatic_atoms:
        if any(b.GetBondType() == Chem.BondType.TRIPLE for b in mol.GetBonds()):
            return "TRIPLE_CONTROL"
        if any(b.GetBondType() == Chem.BondType.DOUBLE for b in mol.GetBonds()):
            return "DOUBLE_CONTROL"
        return "NONAROMATIC"
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()
             if r and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    # Fused means rings share at least one *bond* (two atom indices).  Merely
    # having two disconnected aromatic rings (e.g. biphenyl) is not fused and
    # must not silently inflate the fused stratum.
    fused = any(len(left & right) >= 2 for i, left in enumerate(rings)
                for right in rings[i + 1:])
    polyaromatic_nonfused = len(rings) >= 2 and not fused
    hetero = any(a.GetSymbol() != "C" for a in aromatic_atoms)
    outside_multiple = any(
        b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
        and not (b.GetBeginAtom().GetIsAromatic() and b.GetEndAtom().GetIsAromatic())
        for b in mol.GetBonds()
    )
    if fused:
        return "FUSED_AROMATIC"
    if polyaromatic_nonfused:
        return "POLYAROMATIC_NONFUSED"
    if hetero:
        return "HETEROAROMATIC_NSP"
    if outside_multiple:
        return "AROMATIC_MIXED_MULTIPLE"
    return "AROMATIC_CARBON"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--source-rows", type=Path, required=True)
    parser.add_argument("--panel6k", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(x) for x in args.rows.read_text().splitlines() if x.strip()]
    source = [json.loads(x) for x in args.source_rows.read_text().splitlines() if x.strip()]
    panel = json.loads(args.panel6k.read_text())
    excluded = {r.get("connection_identity") for r in source if r.get("connection_identity")}
    excluded |= {r.get("connection_identity") for r in panel.get("items", []) if r.get("connection_identity")}
    excluded |= {r.get("connection_identity") for r in rows if r.get("connection_identity")}

    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    molecule_categories: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key = row.get("molecule_id")
        cat = category(str(row.get("canonical_smiles", "")))
        by_split[str(row.get("split"))][cat] += 1
        molecule_categories[str(key)].add(cat)
    overlap: dict[str, int] = {}
    split_molecules = {s: {r.get("molecule_id") for r in rows if r.get("split") == s}
                       for s in ("train", "dev", "fresh_holdout")}
    for left, right in (("train", "dev"), ("train", "fresh_holdout"), ("dev", "fresh_holdout")):
        overlap[f"{left}__{right}"] = len(split_molecules[left] & split_molecules[right])

    pool = Counter()
    pool_examples: dict[str, list[str]] = defaultdict(list)
    pool_ids: set[str] = set()
    for entry in entries():
        ci = connection_identity(entry["canonical_smiles"])
        if ci in excluded or ci in pool_ids:
            continue
        pool_ids.add(ci)
        cat = category(entry["canonical_smiles"])
        pool[cat] += 1
        if len(pool_examples[cat]) < 5:
            pool_examples[cat].append(entry["canonical_smiles"])

    report = {
        "schema_version": "chemvector-aromatic-identity-split-r16-audit-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "rows_sha256": digest(args.rows),
        "source_rows_sha256": digest(args.source_rows),
        "panel6k_sha256": digest(args.panel6k),
        "current_split_rows": {split: dict(sorted(counts.items())) for split, counts in sorted(by_split.items())},
        "current_split_molecule_counts": {split: len(ids) for split, ids in split_molecules.items()},
        "molecule_overlap_counts": overlap,
        "excluded_connection_identity_count": len(excluded),
        "available_catalog_pool_after_exclusion": dict(sorted(pool.items())),
        "pool_examples": dict(sorted(pool_examples.items())),
        "finding": {
            "fresh_aromatic_rows": sum(v for k, v in by_split.get("fresh_holdout", {}).items() if "AROMATIC" in k),
            "fresh_aromatic_positive_is_valid": False,
            "fused_pool_is_sufficient_without_catalog_extension": pool["FUSED_AROMATIC"] >= 12,
        },
        "recommended_split": {
            "identity_unit": ["molecule_id", "connection_identity", "canonical_smiles"],
            "aromatic_strata": ["AROMATIC_CARBON", "HETEROAROMATIC_NSP", "FUSED_AROMATIC", "AROMATIC_MIXED_MULTIPLE"],
            "controls": ["DOUBLE_CONTROL", "TRIPLE_CONTROL"],
            "per_stratum_molecules": {"train": 20, "dev": 8, "fresh_holdout": 8},
            "profiles_per_molecule": 4,
            "split_assignment": "stable SHA-256 ordering within each stratum, before profile expansion",
            "admission": "render contract/hash/graph-SDF identity first; CPU atom head <=300 steps, then pair scorer <=500 steps; no GPU before both fresh gates",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
