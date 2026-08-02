#!/usr/bin/env python3
"""Decompose an SDF benchmark into skeleton, atom, and bond-type accuracy."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Callable

import networkx as nx
from rdkit import Chem, RDLogger

from chemvector.paths import PROJECT_ROOT

def molecular_graph(mol: Chem.Mol) -> nx.Graph:
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(
            atom.GetIdx(),
            atomic_number=atom.GetAtomicNum(),
            isotope=atom.GetIsotope(),
            formal_charge=atom.GetFormalCharge(),
            radicals=atom.GetNumRadicalElectrons(),
            aromatic=atom.GetIsAromatic(),
        )
    for bond in mol.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=str(bond.GetBondType()),
            aromatic=bond.GetIsAromatic(),
        )
    return graph


def _atom_type_match(left: dict, right: dict) -> bool:
    keys = ("atomic_number", "isotope", "formal_charge", "radicals")
    return all(left[key] == right[key] for key in keys)


def _full_atom_match(left: dict, right: dict) -> bool:
    return _atom_type_match(left, right) and left["aromatic"] == right["aromatic"]


def _bond_type_match(left: dict, right: dict) -> bool:
    return (
        left["bond_type"] == right["bond_type"]
        and left["aromatic"] == right["aromatic"]
    )


def _atom_signature(value: dict) -> str:
    symbol = Chem.GetPeriodicTable().GetElementSymbol(value["atomic_number"])
    if value["isotope"]:
        symbol = f"{value['isotope']}{symbol}"
    if value["formal_charge"]:
        symbol = f"{symbol}{value['formal_charge']:+d}"
    if value["radicals"]:
        symbol = f"{symbol}[radical={value['radicals']}]"
    return symbol


def _bond_signature(value: dict) -> str:
    return "AROMATIC" if value["aromatic"] else value["bond_type"]


def _isomorphic(
    expected: nx.Graph,
    predicted: nx.Graph,
    *,
    node_match: Callable[[dict, dict], bool] | None = None,
    edge_match: Callable[[dict, dict], bool] | None = None,
) -> bool:
    if (
        expected.number_of_nodes() != predicted.number_of_nodes()
        or expected.number_of_edges() != predicted.number_of_edges()
        or nx.number_connected_components(expected)
        != nx.number_connected_components(predicted)
    ):
        return False
    return nx.is_isomorphic(
        expected,
        predicted,
        node_match=node_match,
        edge_match=edge_match,
    )


def best_attribute_alignment(
    expected: nx.Graph,
    predicted: nx.Graph,
    *,
    maximum_isomorphisms: int = 4096,
) -> dict:
    matcher = nx.algorithms.isomorphism.GraphMatcher(expected, predicted)
    best_atom_correct = 0
    best_bond_correct = 0
    best_atom_confusions: Counter[str] = Counter()
    best_bond_confusions: Counter[str] = Counter()
    mappings_examined = 0
    truncated = False
    for mapping in matcher.isomorphisms_iter():
        mappings_examined += 1
        atom_pairs = [
            (expected.nodes[left], predicted.nodes[right])
            for left, right in mapping.items()
        ]
        bond_pairs = [
            (
                expected.edges[left, right],
                predicted.edges[mapping[left], mapping[right]],
            )
            for left, right in expected.edges
        ]
        atom_correct = sum(_atom_type_match(*pair) for pair in atom_pairs)
        bond_correct = sum(_bond_type_match(*pair) for pair in bond_pairs)
        if atom_correct > best_atom_correct or mappings_examined == 1:
            best_atom_correct = atom_correct
            best_atom_confusions = Counter(
                f"{_atom_signature(left)}->{_atom_signature(right)}"
                for left, right in atom_pairs
                if not _atom_type_match(left, right)
            )
        if bond_correct > best_bond_correct or mappings_examined == 1:
            best_bond_correct = bond_correct
            best_bond_confusions = Counter(
                f"{_bond_signature(left)}->{_bond_signature(right)}"
                for left, right in bond_pairs
                if not _bond_type_match(left, right)
            )
        if (
            best_atom_correct == expected.number_of_nodes()
            and best_bond_correct == expected.number_of_edges()
        ):
            break
        if mappings_examined >= maximum_isomorphisms:
            truncated = True
            break
    return {
        "atom_type_correct_max": best_atom_correct,
        "atom_type_total": expected.number_of_nodes(),
        "bond_type_correct_max": best_bond_correct,
        "bond_type_total": expected.number_of_edges(),
        "topology_isomorphisms_examined": mappings_examined,
        "topology_alignment_truncated": truncated,
        "atom_type_confusions": dict(sorted(best_atom_confusions.items())),
        "bond_type_confusions": dict(sorted(best_bond_confusions.items())),
    }


def compare_molecular_graphs(expected: Chem.Mol, predicted: Chem.Mol) -> dict:
    expected_graph = molecular_graph(expected)
    predicted_graph = molecular_graph(predicted)
    skeleton_exact = _isomorphic(expected_graph, predicted_graph)
    atom_typed = skeleton_exact and _isomorphic(
        expected_graph,
        predicted_graph,
        node_match=_atom_type_match,
    )
    bond_typed = skeleton_exact and _isomorphic(
        expected_graph,
        predicted_graph,
        edge_match=_bond_type_match,
    )
    labeled = skeleton_exact and _isomorphic(
        expected_graph,
        predicted_graph,
        node_match=_full_atom_match,
        edge_match=_bond_type_match,
    )
    result = {
        "atom_count_match": expected_graph.number_of_nodes()
        == predicted_graph.number_of_nodes(),
        "bond_count_match": expected_graph.number_of_edges()
        == predicted_graph.number_of_edges(),
        "skeleton_exact": skeleton_exact,
        "atom_typed_skeleton_exact": atom_typed,
        "bond_typed_skeleton_exact": bond_typed,
        "labeled_graph_exact": labeled,
    }
    if skeleton_exact:
        result.update(best_attribute_alignment(expected_graph, predicted_graph))
    else:
        result.update(
            {
                "atom_type_correct_max": 0,
                "atom_type_total": 0,
                "bond_type_correct_max": 0,
                "bond_type_total": 0,
                "topology_isomorphisms_examined": 0,
                "topology_alignment_truncated": False,
                "atom_type_confusions": {},
                "bond_type_confusions": {},
            }
        )
    return result


def read_mol(path: Path) -> Chem.Mol:
    mol = Chem.MolFromMolFile(
        str(path),
        sanitize=True,
        removeHs=False,
        strictParsing=False,
    )
    if mol is None:
        mol = Chem.MolFromMolFile(
            str(path),
            sanitize=False,
            removeHs=False,
            strictParsing=False,
        )
    if mol is None:
        raise ValueError(f"Cannot parse SDF: {path}")
    return mol


def summarize(records: list[dict]) -> dict:
    samples = len(records)
    parseable = [record for record in records if record["prediction_parseable"]]
    count_exact = [
        record
        for record in parseable
        if record["atom_count_match"] and record["bond_count_match"]
    ]
    skeleton_exact = [record for record in parseable if record["skeleton_exact"]]

    def count(key: str, rows: list[dict] = parseable) -> int:
        return sum(bool(record.get(key)) for record in rows)

    def rate(numerator: int, denominator: int) -> float:
        return numerator / max(denominator, 1)

    skeleton_count = len(skeleton_exact)
    atom_typed_count = count("atom_typed_skeleton_exact")
    bond_typed_count = count("bond_typed_skeleton_exact")
    labeled_count = count("labeled_graph_exact")
    atom_total = sum(int(record["atom_type_total"]) for record in skeleton_exact)
    atom_correct = sum(
        int(record["atom_type_correct_max"]) for record in skeleton_exact
    )
    bond_total = sum(int(record["bond_type_total"]) for record in skeleton_exact)
    bond_correct = sum(
        int(record["bond_type_correct_max"]) for record in skeleton_exact
    )
    atom_confusions = Counter()
    bond_confusions = Counter()
    for record in skeleton_exact:
        atom_confusions.update(record["atom_type_confusions"])
        bond_confusions.update(record["bond_type_confusions"])
    return {
        "samples": samples,
        "prediction_parseable_count": len(parseable),
        "prediction_parseable_rate": rate(len(parseable), samples),
        "count_pair_exact_count": len(count_exact),
        "count_pair_exact_rate": rate(len(count_exact), samples),
        "skeleton_exact_count": skeleton_count,
        "skeleton_exact_rate": rate(skeleton_count, samples),
        "skeleton_exact_given_count_pair_exact_rate": rate(
            skeleton_count, len(count_exact)
        ),
        "atom_typed_skeleton_exact_count": atom_typed_count,
        "atom_typed_skeleton_exact_rate": rate(atom_typed_count, samples),
        "atom_type_exact_given_skeleton_rate": rate(
            atom_typed_count, skeleton_count
        ),
        "bond_typed_skeleton_exact_count": bond_typed_count,
        "bond_typed_skeleton_exact_rate": rate(bond_typed_count, samples),
        "bond_type_exact_given_skeleton_rate": rate(
            bond_typed_count, skeleton_count
        ),
        "labeled_graph_exact_count": labeled_count,
        "labeled_graph_exact_rate": rate(labeled_count, samples),
        "labeled_graph_exact_given_skeleton_rate": rate(
            labeled_count, skeleton_count
        ),
        "atom_type_micro_correct": atom_correct,
        "atom_type_micro_total": atom_total,
        "atom_type_micro_accuracy_given_skeleton": rate(atom_correct, atom_total),
        "bond_type_micro_correct": bond_correct,
        "bond_type_micro_total": bond_total,
        "bond_type_micro_accuracy_given_skeleton": rate(bond_correct, bond_total),
        "topology_alignment_truncated_count": sum(
            bool(record["topology_alignment_truncated"])
            for record in skeleton_exact
        ),
        "atom_type_confusions": dict(atom_confusions.most_common(20)),
        "bond_type_confusions": dict(bond_confusions.most_common(20)),
        "skeleton_exact_atom_type_error_count": sum(
            record["skeleton_exact"]
            and not record["atom_typed_skeleton_exact"]
            for record in parseable
        ),
        "skeleton_exact_bond_type_error_count": sum(
            record["skeleton_exact"]
            and not record["bond_typed_skeleton_exact"]
            for record in parseable
        ),
        "skeleton_exact_labeled_error_count": sum(
            record["skeleton_exact"] and not record["labeled_graph_exact"]
            for record in parseable
        ),
    }


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def evaluate_benchmark_directory(
    benchmark_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> dict:
    RDLogger.DisableLog("rdApp.*")
    benchmark_dir = resolve_project_path(benchmark_dir)
    config = json.loads(
        (benchmark_dir / "benchmark_config.json").read_text(encoding="utf-8")
    )
    manifest_path = resolve_project_path(
        manifest_path or config["benchmark_manifest"]
    )
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = {
            row["molecule_id"]: row for row in csv.DictReader(handle)
        }
    source_records = [
        json.loads(line)
        for line in (benchmark_dir / "records.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    records = []
    for source_record in source_records:
        molecule_id = source_record["molecule_id"]
        row = manifest.get(molecule_id)
        if row is None:
            raise ValueError(f"Missing manifest row: {molecule_id}")
        expected_path = manifest_path.parent / row["sdf_file"]
        prediction_file = source_record.get("prediction_sdf_file")
        record = {
            "molecule_id": molecule_id,
            "prediction_parseable": False,
            "sdf_valid": bool(source_record.get("sdf_valid")),
        }
        if prediction_file:
            prediction_path = benchmark_dir / prediction_file
            try:
                comparison = compare_molecular_graphs(
                    read_mol(expected_path),
                    read_mol(prediction_path),
                )
                record.update(comparison)
                record["prediction_parseable"] = True
            except Exception as error:
                record["error"] = str(error)
        records.append(record)

    summary = summarize(records)
    write_jsonl(benchmark_dir / "object_decomposition_records.jsonl", records)
    write_json(benchmark_dir / "object_decomposition.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args()

    summary = evaluate_benchmark_directory(args.benchmark_dir, args.manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "compare_molecular_graphs",
    "evaluate_benchmark_directory",
    "molecular_graph",
    "summarize",
]
