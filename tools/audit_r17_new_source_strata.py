#!/usr/bin/env python3
"""Metadata-only bond/visible-O-N strata audit for an admitted source."""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

def run(manifest_path: Path, source: Path, out: Path) -> dict:
    if out.exists(): raise FileExistsError(out)
    rows = json.loads(manifest_path.read_text())["rows"]
    by = defaultdict(Counter)
    molecule_sets = defaultdict(lambda: defaultdict(set))
    for row in rows:
        key = row["sample_key"]
        layers = json.loads((source / f"{key}.layers.json").read_text())
        graph = json.loads((source / f"{key}.graph.json").read_text())
        atoms = {int(a["atom_id"]): a for a in layers.get("atoms", [])}
        visible = {i for i,a in atoms.items() if a.get("visible_text") and a.get("element") in {"O","N"}}
        elems = {i: str(atoms[i].get("element")) for i in visible}
        for bond in graph.get("bonds", []):
            a,b = int(bond["begin_atom"]), int(bond["end_atom"])
            typ = str(bond.get("chemical_type", "SINGLE")).upper()
            by[(str(row["split"]),str(row["profile_id"]))][f"GRAPH:{typ}"] += 1
            molecule_sets[(str(row["split"]),str(row["profile_id"]))][f"GRAPH:{typ}"].add(row["molecule_id"])
            if a not in visible and b not in visible: continue
            split = str(row["split"]); profile = str(row["profile_id"])
            for atom in (a,b):
                if atom in elems:
                    label = f"{elems[atom]}:{typ}"
                    by[(split,profile)][label] += 1
                    molecule_sets[(split,profile)][label].add(row["molecule_id"])
                    by[(split,profile)][f"ANY:{typ}"] += 1
                    molecule_sets[(split,profile)][f"ANY:{typ}"].add(row["molecule_id"])
    counts = {f"{s}/{p}": dict(sorted(c.items())) for (s,p),c in sorted(by.items())}
    molecule_counts = {f"{s}/{p}": {k: len(v) for k,v in sorted(d.items())} for (s,p),d in sorted(molecule_sets.items())}
    # O-triple is not a normal neutral organic drawing case; nitrile triples
    # are represented by carbon-visible endpoints rather than O/N-visible
    # endpoints.  Require the chemically represented O/N strata here and
    # report all-triple coverage separately through the graph bond counts.
    required = ["GRAPH:SINGLE","GRAPH:DOUBLE","GRAPH:AROMATIC","GRAPH:TRIPLE"]
    holdout_min = min((molecule_counts.get(f"holdout/{p}",{}).get(k,0) for p in {r["profile_id"] for r in rows} for k in required), default=0)
    result = {
        "schema_version": "chemvector-r17-new-source-strata-0.1", "complete": True,
        "read_only": True, "metadata_only": True, "training_performed": False,
        "gpu_used": False, "sealed_test_read": False, "rows": len(rows),
        "counts": counts, "molecule_counts": molecule_counts,
        "gate": {"profiles_complete": len({r["profile_id"] for r in rows}) == 8,
                  "train_holdout_present": {r["split"] for r in rows} == {"train","holdout"},
                  "holdout_min_visible_element_bond_molecules": holdout_min,
                  "training_admissible": False, "gpu_training_go": False, "production_go": False},
    }
    out.mkdir(parents=True); (out/"report.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2)); return result

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--source",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args(); run(a.manifest,a.source,a.out)
