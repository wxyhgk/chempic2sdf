"""CPU-only data compatibility audit for the self-developed ChemVector route.

This command contains no model loading, optimizer, GPU initialization or training
entrypoint. Passing the audit does not admit quarantined data for training.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import torch
from rdkit import Chem

from .data import CurriculumCollator, PAYLOADS, load_sample


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def run(data_root, output):
    root, output = Path(data_root), Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh CPU preflight output directory")
    input_files = {name: (root / name).read_bytes() for name in (
        "summary.json", "samples.jsonl", "sample-sha256.json")}
    summary = json.loads(input_files["summary.json"])
    hashes = json.loads(input_files["sample-sha256.json"])
    rows = [json.loads(line) for line in input_files["samples.jsonl"].splitlines() if line.strip()]
    if (not rows or len(rows) != summary["accepted_images"]
            or len({r["sample_key"] for r in rows}) != len(rows)):
        raise ValueError("Curriculum sample manifest count/identity mismatch")
    collator = CurriculumCollator(purpose="diagnostic")
    results, elements, chemical_bonds, display_bonds = [], Counter(), Counter(), Counter()
    for index, row in enumerate(rows, 1):
        key = row["sample_key"]
        result = {"sample_key": key, "molecule_id": row["molecule_id"], "compatible": False}
        try:
            # load_sample validates safe paths and same-render bindings before
            # any following key-derived path is read.
            sample = load_sample(root, key, purpose="diagnostic")
            for suffix in (*PAYLOADS, ".meta.json"):
                name = "samples/" + key + suffix
                if sha256((root / name).read_bytes()).hexdigest() != hashes.get(name):
                    raise ValueError("Corpus-level artifact hash mismatch")
            if (sample["meta"]["molecule_id"] != row["molecule_id"]
                    or sample["meta"]["profile_id"] != row["profile_id"]):
                raise ValueError("Sample row identity mismatch")
            sdf = (root / "samples" / (key + ".sdf")).read_text()
            mol = Chem.MolFromMolBlock(sdf, sanitize=True, removeHs=False, strictParsing=True)
            graph = sample["graph"]
            if (mol is None or Chem.MolToSmiles(mol) != graph["canonical_smiles"]
                    or mol.GetNumAtoms() != len(graph["atoms"]) or mol.GetNumBonds() != len(graph["bonds"])):
                raise ValueError("SDF graph identity mismatch")
            for atom in graph["atoms"]:
                restored = mol.GetAtomWithIdx(atom["atom_id"])
                if (restored.GetSymbol(), restored.GetIsotope(), restored.GetFormalCharge()) != (
                        atom["element"], atom["isotope"], atom["formal_charge"]):
                    raise ValueError("SDF atom index/chemistry mismatch")
            for bond in graph["bonds"]:
                restored = mol.GetBondWithIdx(bond["bond_id"])
                if ({restored.GetBeginAtomIdx(), restored.GetEndAtomIdx()} != {bond["begin_atom"], bond["end_atom"]}
                        or str(restored.GetBondType()) != bond["chemical_type"]):
                    raise ValueError("SDF bond index/chemistry mismatch")
            batch = collator([sample])
            target_hashes = {name: sha256(t.contiguous().numpy().tobytes()).hexdigest()
                             for name, t in batch.items() if isinstance(t, torch.Tensor) and name != "pixel_values"}
            collisions = batch["spatial_cell_collision_pairs"][0]
            result.update(compatible=collisions == 0, spatial_cell_collision_pairs=collisions,
                          atoms=int(batch["atom_mask"].sum()), bonds=int(batch["bond_mask"].sum()),
                          text_atoms=int(batch["atom_text_mask"].sum()), target_sha256=target_hashes)
            if collisions:
                result["error"] = "Existing spatial atom loss cannot represent colliding cells"
            elements.update(a["element"] for a in graph["atoms"])
            chemical_bonds.update(b["chemical_type"] for b in graph["bonds"])
            display_bonds.update(b["drawing"]["bond_type"] for b in sample["layers"]["bonds"])
        except (ValueError, KeyError, OSError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
        if index % 100 == 0 or index == len(rows):
            print(f"CPU target audit: {index}/{len(rows)}", flush=True)
    sources = {str(p): sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
    report = {
        "schema_version": "chemvector-curriculum-cpu-preflight-0.1",
        "route": "self_developed_chemvector", "model_architecture_changed": False,
        "model_loaded": False, "gpu_used": False, "training_performed": False,
        "training_authorized": False, "target_compatible_images": sum(r["compatible"] for r in results),
        "checked_images": len(results), "failed_images": sum(not r["compatible"] for r in results),
        "unique_molecules_with_images": len({r["molecule_id"] for r in results}),
        "planned_images": summary["planned_images"], "renderer_rejected_images": summary["rejected_images"],
        "external_overlap_status": summary["external_overlap_status"],
        "target_schema": "chemvector-curriculum-targets-0.1",
        "supervision": ["atom presence/anchor", "visible text role/ink bbox", "bond order/drawing direction", "ID-based bond endpoints"],
        "not_trained_by_this_adapter": ["OCR transcription", "element/charge/isotope classification", "functional-group classifier"],
        "chemical_attributes": "Preserved in chemical_graphs for semantic targets and scoring, not discarded or inferred from OCR",
        "semantic_mask": "bitset input is not passed to legacy categorical losses",
        "bond_ports": "existing q1023 text-bbox-ray supervision, not measured stroke endpoints",
        "elements_by_image_occurrence": dict(elements), "chemical_bonds_by_image_occurrence": dict(chemical_bonds),
        "display_bonds_by_image_occurrence": dict(display_bonds),
        "blockers_before_training": [
            "Pilot remains quarantine; independent identity admission and an authorized grouped split are required.",
            "This data-only audit cannot authorize a launch; a released mixture and bounded training protocol are required.",
        ] + (["Systematic render rejections must be fixed before scaling coverage."]
             if summary["rejected_images"] else []),
        "input_sha256": {name: sha256(data).hexdigest() for name, data in input_files.items()},
        "source_sha256": sources,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    write_json(output / "samples.json", results)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    report = run(args.data_root, args.output)
    print(json.dumps({k: report[k] for k in ("checked_images", "target_compatible_images", "failed_images", "training_authorized")}, indent=2))
    return 0 if report["failed_images"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
