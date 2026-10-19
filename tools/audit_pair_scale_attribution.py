"""CPU-only attribution of a saved pair-scale validation comparison.

This module intentionally reads only the two saved validation JSON files and
the run report.  It never loads a checkpoint, initializes torch, or reads an
image.  The output describes observable structure changes; it does not claim
which hidden pair logit or optimizer update caused a change.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/experiments/chemvector-pair-scale-20260924"
DEFAULT_OUTPUT = ROOT / "outputs/audits/chemvector-pair-scale-attribution-20260925"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mol_features(molblock: str | None) -> dict[str, Any]:
    """Extract stable graph counts without sanitizing or changing a molecule."""
    if not molblock:
        return {
            "parsed": False,
            "atoms": None,
            "bonds": None,
            "aromatic_atoms": None,
            "aromatic_bonds": None,
            "components": None,
            "rings": None,
        }
    # Saved structures are often Kekulé MolBlocks.  Standard parsing is needed
    # to recover aromatic flags; the unsanitized fallback preserves counts for
    # structures whose final output is chemically invalid.
    mol = Chem.MolFromMolBlock(molblock, sanitize=True, removeHs=False)
    if mol is None:
        mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
    if mol is None:
        return {
            "parsed": False,
            "atoms": None,
            "bonds": None,
            "aromatic_atoms": None,
            "aromatic_bonds": None,
            "components": None,
            "rings": None,
        }
    return {
        "parsed": True,
        "atoms": mol.GetNumAtoms(),
        "bonds": mol.GetNumBonds(),
        "aromatic_atoms": sum(atom.GetIsAromatic() for atom in mol.GetAtoms()),
        "aromatic_bonds": sum(bond.GetIsAromatic() for bond in mol.GetBonds()),
        "components": len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)),
        "rings": mol.GetRingInfo().NumRings(),
    }


def _error_category(error: str | None) -> str | None:
    if not error:
        return None
    text = error.lower()
    if "explicit valence" in text:
        return "explicit_valence"
    if "kekulize" in text:
        return "kekulize"
    if "sanitize" in text:
        return "sanitize"
    return "other"


def enrich(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": record["key"],
        "domain": record["domain"],
        "valid": bool(record["valid"]),
        "exact": bool(record["exact"]),
        "fallback": bool(record["fallback"]),
        "predicted_smiles": record.get("predicted_smiles"),
        "error": record.get("error"),
        "error_category": _error_category(record.get("error")),
        "graph": _mol_features(record.get("molblock")),
    }


def compare(baseline: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    """Return observable deltas for one fixed validation key."""
    if baseline["key"] != final["key"] or baseline["domain"] != final["domain"]:
        raise ValueError("baseline/final key or domain mismatch")
    before, after = enrich(baseline), enrich(final)
    old_graph, new_graph = before["graph"], after["graph"]
    categories: list[str] = []
    if before["exact"] and not after["exact"]:
        categories.append("broken_exact")
    if not before["exact"] and after["exact"]:
        categories.append("rescued_exact")
    if before["valid"] and not after["valid"]:
        categories.append("became_invalid")
    if not before["fallback"] and after["fallback"]:
        categories.append("new_fallback")
    if before["fallback"] and not after["fallback"]:
        categories.append("fallback_removed")
    if (old_graph["aromatic_atoms"] is not None and new_graph["aromatic_atoms"] is not None
            and new_graph["aromatic_atoms"] < old_graph["aromatic_atoms"]):
        categories.append("aromatic_atoms_decreased")
    if (old_graph["aromatic_bonds"] is not None and new_graph["aromatic_bonds"] is not None
            and new_graph["aromatic_bonds"] < old_graph["aromatic_bonds"]):
        categories.append("aromatic_bonds_decreased")
    if (old_graph["components"] is not None and new_graph["components"] is not None
            and new_graph["components"] != old_graph["components"]):
        categories.append("component_count_changed")
    if before["error_category"] != after["error_category"] and after["error_category"]:
        categories.append("error_category_changed")
    deltas = {
        field: (old_graph[field], new_graph[field], new_graph[field] - old_graph[field])
        for field in ("atoms", "bonds", "aromatic_atoms", "aromatic_bonds", "components", "rings")
        if old_graph[field] is not None and new_graph[field] is not None
    }
    return {
        "key": before["key"],
        "domain": before["domain"],
        "categories": categories,
        "baseline": before,
        "final": after,
        "graph_deltas": deltas,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    groups = [("all", rows)]
    groups.extend((domain, [row for row in rows if row["domain"] == domain])
                  for domain in sorted({row["domain"] for row in rows}))
    for name, subset in groups:
        counts = Counter(category for row in subset for category in row["categories"])
        result[name] = {
            "images": len(subset),
            "baseline": {
                "exact": sum(row["baseline"]["exact"] for row in subset),
                "valid": sum(row["baseline"]["valid"] for row in subset),
                "fallback": sum(row["baseline"]["fallback"] for row in subset),
            },
            "final": {
                "exact": sum(row["final"]["exact"] for row in subset),
                "valid": sum(row["final"]["valid"] for row in subset),
                "fallback": sum(row["final"]["fallback"] for row in subset),
            },
            "category_counts": dict(sorted(counts.items())),
        }
    return result


def build_report(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "report.json"
    baseline_path = run_dir / "validation-000000.json"
    final_path = run_dir / "validation-015443.json"
    source_report = read_json(report_path)
    baseline = read_json(baseline_path)["records"]
    final = read_json(final_path)["records"]
    old = {row["key"]: row for row in baseline}
    new = {row["key"]: row for row in final}
    if set(old) != set(new) or len(old) != len(baseline) or len(new) != len(final):
        raise ValueError("validation keys must be unique and identical")
    rows = [compare(old[key], new[key]) for key in sorted(old)]
    changed = [row for row in rows if row["categories"]]
    report = {
        "schema_version": "chemvector-pair-scale-attribution-0.1",
        "scope": {
            "run": str(run_dir.resolve()),
            "records": len(rows),
            "domains": sorted({row["domain"] for row in rows}),
            "read_images": False,
            "read_checkpoints": False,
            "model_forwards": 0,
            "optimizer_steps": 0,
            "gpu_used": False,
            "sealed_test_read": False,
        },
        "source_run": {
            "training_steps": source_report.get("optimizer_steps"),
            "training_performed": source_report.get("training_performed"),
            "deployment_authorized": source_report.get("deployment_authorized"),
            "selected_for_review": source_report.get("selected_for_review"),
        },
        "summary": _summary(rows),
        "lists": {
            "broken_exact": [row["key"] for row in rows if "broken_exact" in row["categories"]],
            "rescued_exact": [row["key"] for row in rows if "rescued_exact" in row["categories"]],
            "aromatic_degradation": [row["key"] for row in rows if
                                     {"aromatic_atoms_decreased", "aromatic_bonds_decreased"} & set(row["categories"])],
            "new_fallback": [row["key"] for row in rows if "new_fallback" in row["categories"]],
            "became_invalid": [row["key"] for row in rows if "became_invalid" in row["categories"]],
            "component_count_changed": [row["key"] for row in rows if "component_count_changed" in row["categories"]],
        },
        "changed_cases": changed,
        "limitations": [
            "The saved artifacts contain no pair logits, candidate rows, feature tensors, or compiler trace.",
            "Categories are output-level observations and do not identify a causal hidden logit or optimizer update.",
            "MolBlocks are parsed with sanitize=False solely to count observable aromaticity and components.",
        ],
        "input_sha256": {
            "report.json": digest(report_path),
            "validation-000000.json": digest(baseline_path),
            "validation-015443.json": digest(final_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_report(args.run_dir, args.output_dir)
    print(json.dumps({"records": report["scope"]["records"], "changed_cases": len(report["changed_cases"]),
                      "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
