from chemvector.evaluation.audits.bonds.build_v1p2d2e3a_style_endpoint_manifest import (
    assign_scaffold_splits,
)


def test_style_endpoint_split_keeps_scaffolds_and_molecules_together() -> None:
    rows = []
    for scaffold_index in range(1_510):
        for molecule_index in range(2):
            molecule_id = f"m{scaffold_index}-{molecule_index}"
            for variant in ("coordgen", "original", "rdkit", "stress"):
                rows.append(
                    {
                        "sample_key": f"{molecule_id}-{variant}",
                        "molecule_id": molecule_id,
                        "scaffold": f"s{scaffold_index}",
                        "variant": variant,
                    }
                )
    assigned = assign_scaffold_splits(rows)
    scaffold_splits = {}
    molecule_splits = {}
    for row in assigned:
        scaffold_splits.setdefault(row["scaffold"], set()).add(row["split"])
        molecule_splits.setdefault(row["molecule_id"], set()).add(row["split"])
    assert all(len(values) == 1 for values in scaffold_splits.values())
    assert all(len(values) == 1 for values in molecule_splits.values())
    assert {row["split"] for row in assigned} == {"fit", "calibration", "holdout"}
