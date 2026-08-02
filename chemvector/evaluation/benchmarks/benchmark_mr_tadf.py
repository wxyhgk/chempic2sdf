#!/usr/bin/env python3
"""Benchmark parallel ChemVector inference on rendered MR-TADF structures."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

from PIL import Image
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
import torch

from Data.sdf.chemsvg_sequence_v4 import (
    diagnostic_mol_from_chemsvg_v4,
    mol_from_chemsvg_v4,
    parse_chemsvg_v4,
)
from chemvector.benchmark_diagnostics import diagnose_chemsvg, summarize_diagnostics
from chemvector.evaluate import load_model
from chemvector.image_io import image_tensor
from chemvector.parallel_v6 import generate_parallel_chemsvg_v6, strip_autoregressive_modules
from chemvector.proposal_keep import load_proposal_keep_checkpoint
from chemvector.proposal_graph import load_proposal_graph_checkpoint
from chemvector.repair_v4 import InkGraphRepairConfig, repair_chemsvg_v4_from_image
from chemvector.structure_gate import (
    expert_route_probability,
    load_packed_structure_expert,
    load_structure_gate,
)


def failure_category(error: Exception) -> str:
    message = str(error).lower()
    if "not connected" in message or "component" in message:
        return "disconnected"
    if "valence" in message:
        return "valence"
    if "kekul" in message:
        return "kekulize"
    return "other"


def summarize_records(records: list[dict]) -> dict:
    samples = len(records)
    difficulty_groups: dict[str, list[dict]] = defaultdict(list)
    failure_counts = Counter()
    for record in records:
        difficulty_groups[record["difficulty"]].append(record)
        if record.get("failure_category"):
            failure_counts[record["failure_category"]] += 1

    def rates(group: list[dict]) -> dict[str, float | int]:
        count = len(group)
        return {
            "samples": count,
            "sdf_valid_rate": sum(bool(row["sdf_valid"]) for row in group)
            / max(count, 1),
            "formula_match_rate": sum(bool(row["formula_match"]) for row in group)
            / max(count, 1),
            "nonstereo_structure_match_rate": sum(
                bool(row["nonstereo_structure_match"]) for row in group
            )
            / max(count, 1),
            "chemical_match_rate": sum(bool(row["chemical_match"]) for row in group)
            / max(count, 1),
        }

    summary = rates(records)
    summary.update(
        {
            "average_seconds": sum(float(row["seconds"]) for row in records)
            / max(samples, 1),
            "atom_count_match_rate": sum(
                bool(row["atom_count_match"]) for row in records
            )
            / max(samples, 1),
            "bond_count_match_rate": sum(
                bool(row["bond_count_match"]) for row in records
            )
            / max(samples, 1),
            "failure_categories": dict(sorted(failure_counts.items())),
            "difficulty_rates": {
                name: rates(group) for name, group in sorted(difficulty_groups.items())
            },
        }
    )
    if any(record.get("diagnostics") for record in records):
        summary["error_diagnostics"] = summarize_diagnostics(records)
    return summary


def nonstereo_inchikey(mol: Chem.Mol) -> str:
    copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(copy)
    return Chem.MolToInchiKey(copy)


def molecular_similarity(expected: Chem.Mol, predicted: Chem.Mol) -> float:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    expected_fingerprint = generator.GetFingerprint(expected)
    predicted_fingerprint = generator.GetFingerprint(predicted)
    return float(
        DataStructs.TanimotoSimilarity(
            expected_fingerprint,
            predicted_fingerprint,
        )
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_benchmark_config(config_path: Path, config: dict) -> None:
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            keys = sorted(set(existing) | set(config))
            mismatches = {
                key: (existing.get(key), config.get(key))
                for key in keys
                if existing.get(key) != config.get(key)
            }
            raise ValueError(
                "Existing benchmark output uses a different configuration: "
                f"{mismatches}"
            )
        return
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_prediction_sdf(
    mol: Chem.Mol,
    path: Path,
    *,
    sanitize: bool,
) -> Chem.Mol:
    """Write the delivered SDF and return the molecule read back from that file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        writer = Chem.SDWriter(str(temporary))
        if not sanitize:
            writer.SetKekulize(False)
        try:
            writer.write(mol)
        finally:
            writer.close()
        written = Chem.MolFromMolFile(
            str(temporary),
            sanitize=sanitize,
            removeHs=False,
            strictParsing=False,
        )
        if written is None:
            raise ValueError("RDKit could not read back the generated SDF")
        temporary.replace(path)
        return written
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_manual_diagnostic_sdf(sequence: str, path: Path) -> tuple[int, int]:
    """Write a V3000 diagnostic SDF when RDKit cannot serialize an invalid graph."""

    parsed = parse_chemsvg_v4(sequence)
    coordinate_to_atom = {
        (vertex.x, vertex.y): index
        for index, vertex in enumerate(parsed.vertices, start=1)
    }
    bond_types = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
    lines = [
        "ChemVector diagnostic prediction",
        "  ChemVector",
        "",
        "  0  0  0  0  0  0  0  0  0  0999 V3000",
        "M  V30 BEGIN CTAB",
        f"M  V30 COUNTS {len(parsed.vertices)} {len(parsed.strokes)} 0 0 0",
        "M  V30 BEGIN ATOM",
    ]
    for atom_id, vertex in enumerate(parsed.vertices, start=1):
        properties = []
        if vertex.formal_charge:
            properties.append(f"CHG={vertex.formal_charge}")
        if vertex.isotope:
            properties.append(f"MASS={vertex.isotope}")
        if vertex.radical_electrons:
            properties.append(f"RAD={min(vertex.radical_electrons + 1, 3)}")
        suffix = f" {' '.join(properties)}" if properties else ""
        lines.append(
            f"M  V30 {atom_id} {vertex.element} {vertex.x / 100.0:.6f} "
            f"{-vertex.y / 100.0:.6f} 0.000000 0{suffix}"
        )
    lines.extend(("M  V30 END ATOM", "M  V30 BEGIN BOND"))
    for bond_id, stroke in enumerate(parsed.strokes, start=1):
        lines.append(
            f"M  V30 {bond_id} {bond_types[stroke.bond_type]} "
            f"{coordinate_to_atom[stroke.begin]} {coordinate_to_atom[stroke.end]}"
        )
    lines.extend(
        (
            "M  V30 END BOND",
            "M  V30 END CTAB",
            "M  END",
            ">  <CHEMVECTOR_STATUS>",
            "invalid",
            "",
            ">  <CHEMSVG_SEQUENCE>",
            sequence.rstrip(),
            "",
            "$$$$",
            "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return len(parsed.vertices), len(parsed.strokes)


def write_invalid_prediction_sdf(sequence: str, path: Path) -> tuple[int, int]:
    try:
        diagnostic_mol = diagnostic_mol_from_chemsvg_v4(sequence)
        written = write_prediction_sdf(diagnostic_mol, path, sanitize=False)
        return written.GetNumAtoms(), written.GetNumBonds()
    except Exception:
        return write_manual_diagnostic_sdf(sequence, path)


def backfill_prediction_artifacts(records: list[dict], output_dir: Path) -> bool:
    """Fill SDF artifacts from saved ChemSVG without rerunning model inference."""

    changed = False
    sdf_dir = output_dir / "sdf"
    for record in records:
        molecule_id = record["molecule_id"]
        raw_sequence_path = (
            output_dir / "raw_chemsvg" / f"{molecule_id}.chemsvg.txt"
        )
        repaired_sequence_path = (
            output_dir / "repaired_chemsvg" / f"{molecule_id}.chemsvg.txt"
        )
        if raw_sequence_path.is_file() and "raw_strokes" not in record:
            parsed = parse_chemsvg_v4(raw_sequence_path.read_text(encoding="utf-8"))
            record["raw_vertices"] = len(parsed.vertices)
            record["raw_strokes"] = len(parsed.strokes)
            changed = True
        if repaired_sequence_path.is_file() and "repaired_strokes" not in record:
            parsed = parse_chemsvg_v4(
                repaired_sequence_path.read_text(encoding="utf-8")
            )
            record["repaired_vertices"] = len(parsed.vertices)
            record["repaired_strokes"] = len(parsed.strokes)
            changed = True
        relative = record.get("prediction_sdf_file")
        if relative and (output_dir / relative).is_file():
            continue
        if record.get("sdf_valid"):
            sequence_path = (
                output_dir / "repaired_chemsvg" / f"{molecule_id}.chemsvg.txt"
            )
            prefix = "correct" if record.get("chemical_match") else "wrong"
            sanitize = True
        else:
            sequence_path = (
                repaired_sequence_path
                if repaired_sequence_path.is_file()
                else raw_sequence_path
            )
            prefix = "invalid"
            sanitize = False
        if not sequence_path.is_file():
            continue
        try:
            sequence = sequence_path.read_text(encoding="utf-8")
            final_sdf = sdf_dir / f"{prefix}_{molecule_id}.sdf"
            if sanitize:
                mol = mol_from_chemsvg_v4(sequence)
                written = write_prediction_sdf(mol, final_sdf, sanitize=True)
            else:
                atom_count, bond_count = write_invalid_prediction_sdf(
                    sequence,
                    final_sdf,
                )
                written = None
            record["predicted_atoms"] = (
                written.GetNumAtoms() if written is not None else atom_count
            )
            record["predicted_bonds"] = (
                written.GetNumBonds() if written is not None else bond_count
            )
            record["prediction_sdf_file"] = str(final_sdf.relative_to(output_dir))
            record.pop("diagnostic_sdf_error", None)
            changed = True
        except Exception as error:
            record["diagnostic_sdf_error"] = str(error)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/chemvector-v5-vertex-10k/last.pt",
    )
    parser.add_argument("--gate-checkpoint")
    parser.add_argument("--gate-threshold", type=float, default=0.9)
    parser.add_argument(
        "--benchmark-manifest",
        default="Data/benchmarks/mr_tadf_v1/manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluations/mr-tadf-v1-v071",
    )
    parser.add_argument("--samples", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--vertex-threshold", type=float, default=0.3)
    parser.add_argument("--vertex-nms-kernel-size", type=int, default=3)
    parser.add_argument("--proposal-keep-checkpoint")
    parser.add_argument("--proposal-keep-threshold", type=float, default=0.74)
    parser.add_argument("--proposal-graph-checkpoint")
    parser.add_argument("--proposal-graph-threshold", type=float, default=0.87)
    parser.add_argument("--proposal-graph-edge-prior-threshold", type=float)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--ocr-roi-radius", type=float, default=20.0)
    parser.add_argument("--maximum-vertices", type=int, default=128)
    parser.add_argument(
        "--no-save-artifacts",
        action="store_true",
        help="Do not retain predicted ChemSVG and valid SDF artifacts.",
    )
    parser.add_argument("--save-sdf", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--repair-candidate-mode",
        choices=("all", "model", "model_connect"),
        default="all",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.samples is not None and args.samples < 1:
        raise ValueError("samples must be positive")
    if args.cpu_threads < 1 or args.maximum_vertices < 1:
        raise ValueError("CPU threads and maximum vertices must be positive")
    if args.vertex_nms_kernel_size < 1 or args.vertex_nms_kernel_size % 2 == 0:
        raise ValueError("vertex-nms-kernel-size must be a positive odd integer")
    if not 0.0 < args.proposal_keep_threshold < 1.0:
        raise ValueError("proposal-keep-threshold must be in (0, 1)")
    if not 0.0 < args.proposal_graph_threshold < 1.0:
        raise ValueError("proposal-graph-threshold must be in (0, 1)")
    if (
        args.proposal_graph_edge_prior_threshold is not None
        and not 0.0 < args.proposal_graph_edge_prior_threshold < 1.0
    ):
        raise ValueError("proposal-graph-edge-prior-threshold must be in (0, 1)")
    if args.proposal_graph_edge_prior_threshold is not None and not args.proposal_graph_checkpoint:
        raise ValueError("Proposal graph edge prior requires proposal graph checkpoint")
    if not 0.0 <= args.gate_threshold <= 1.0:
        raise ValueError("gate-threshold must be in [0, 1]")
    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)

    benchmark_manifest = Path(args.benchmark_manifest)
    benchmark_root = benchmark_manifest.parent
    with benchmark_manifest.open(newline="", encoding="utf-8") as handle:
        benchmark_rows = list(csv.DictReader(handle))
    if args.samples is not None:
        benchmark_rows = benchmark_rows[: args.samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    records = []
    if records_path.is_file():
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if backfill_prediction_artifacts(records, output_dir):
        temporary_records = records_path.with_suffix(".jsonl.tmp")
        temporary_records.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n" for record in records
            ),
            encoding="utf-8",
        )
        temporary_records.replace(records_path)
    completed = {record["molecule_id"] for record in records}
    checkpoint_path = Path(args.checkpoint)
    config = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "benchmark_manifest": str(benchmark_manifest),
        "benchmark_manifest_sha256": sha256_file(benchmark_manifest),
        "device": args.device,
        "vertex_threshold": args.vertex_threshold,
        "vertex_nms_kernel_size": args.vertex_nms_kernel_size,
        "proposal_keep_checkpoint": args.proposal_keep_checkpoint,
        "proposal_keep_checkpoint_sha256": (
            sha256_file(Path(args.proposal_keep_checkpoint))
            if args.proposal_keep_checkpoint
            else None
        ),
        "proposal_keep_threshold": (
            args.proposal_keep_threshold if args.proposal_keep_checkpoint else None
        ),
        "proposal_graph_checkpoint": args.proposal_graph_checkpoint,
        "proposal_graph_checkpoint_sha256": (
            sha256_file(Path(args.proposal_graph_checkpoint))
            if args.proposal_graph_checkpoint
            else None
        ),
        "proposal_graph_threshold": (
            args.proposal_graph_threshold if args.proposal_graph_checkpoint else None
        ),
        "proposal_graph_edge_prior_threshold": (
            args.proposal_graph_edge_prior_threshold
            if args.proposal_graph_checkpoint
            else None
        ),
        "edge_threshold": args.edge_threshold,
        "ocr_roi_radius": args.ocr_roi_radius,
        "maximum_vertices": args.maximum_vertices,
        "repair_candidate_mode": args.repair_candidate_mode,
        "gate_checkpoint": args.gate_checkpoint,
        "gate_threshold": args.gate_threshold if args.gate_checkpoint else None,
    }
    config_path = output_dir / "benchmark_config.json"
    ensure_benchmark_config(config_path, config)

    model, _ = load_model(checkpoint_path, device)
    strip_autoregressive_modules(model)
    proposal_keep_head = None
    if args.proposal_keep_checkpoint:
        proposal_keep_head, proposal_payload = load_proposal_keep_checkpoint(
            args.proposal_keep_checkpoint,
            device,
        )
        if Path(proposal_payload["dense_checkpoint"]).resolve() != checkpoint_path.resolve():
            raise ValueError("Proposal keep dense checkpoint does not match --checkpoint")
    proposal_graph_head = None
    if args.proposal_graph_checkpoint:
        proposal_graph_head, graph_payload = load_proposal_graph_checkpoint(
            args.proposal_graph_checkpoint,
            device,
        )
        if Path(graph_payload["dense_checkpoint"]).resolve() != checkpoint_path.resolve():
            raise ValueError("Proposal graph dense checkpoint does not match --checkpoint")
        expected_keep = graph_payload.get("proposal_keep_checkpoint")
        if expected_keep and not args.proposal_keep_checkpoint:
            raise ValueError("Proposal graph checkpoint requires proposal keep checkpoint")
        if expected_keep and Path(expected_keep).resolve() != Path(
            args.proposal_keep_checkpoint
        ).resolve():
            raise ValueError("Proposal graph and keep checkpoints do not match")
    gate = None
    expert_model = None
    if args.gate_checkpoint:
        gate, gate_payload = load_structure_gate(args.gate_checkpoint, device)
        if Path(gate_payload["base_checkpoint"]).resolve() != checkpoint_path.resolve():
            raise ValueError("Gate base checkpoint does not match --checkpoint")
        if "expert_structure_tensors" in gate_payload:
            expert_model = load_packed_structure_expert(model, gate_payload, device)
        else:
            expert_model, _ = load_model(
                Path(gate_payload["expert_checkpoint"]), device
            )
            strip_autoregressive_modules(expert_model)
    raw_sequence_dir = output_dir / "raw_chemsvg"
    repaired_sequence_dir = output_dir / "repaired_chemsvg"
    sdf_dir = output_dir / "sdf"
    save_artifacts = not args.no_save_artifacts or args.save_sdf
    if save_artifacts:
        raw_sequence_dir.mkdir(parents=True, exist_ok=True)
        repaired_sequence_dir.mkdir(parents=True, exist_ok=True)
        sdf_dir.mkdir(parents=True, exist_ok=True)

    for row in benchmark_rows:
        molecule_id = row["molecule_id"]
        if molecule_id in completed:
            continue
        image_path = benchmark_root / row["image_file"]
        image = Image.open(image_path).convert("RGB")
        pixels = image_tensor(image).unsqueeze(0).to(device)
        expected = Chem.MolFromMolFile(
            str(benchmark_root / row["sdf_file"]),
            removeHs=False,
        )
        if expected is None:
            raise ValueError(f"Cannot read benchmark ground truth: {row['sdf_file']}")
        annotation = json.loads(
            (benchmark_root / row["annotation_file"]).read_text(encoding="utf-8")
        )
        expected_nonstereo_key = nonstereo_inchikey(expected)
        record = {
            "molecule_id": molecule_id,
            "difficulty": row["difficulty"],
            "maximum_to_median_bond_ratio": float(
                row["maximum_to_median_bond_ratio"]
            ),
            "expected_atoms": int(row["atoms"]),
            "expected_bonds": int(row["bonds"]),
            "predicted_atoms": 0,
            "predicted_bonds": 0,
            "sdf_valid": False,
            "formula_match": False,
            "atom_count_match": False,
            "bond_count_match": False,
            "nonstereo_structure_match": False,
            "chemical_match": False,
        }
        inference_model = model
        if gate is not None:
            route_probability = expert_route_probability(model, gate, pixels)
            record["expert_route_probability"] = route_probability
            record["selected_expert"] = route_probability >= args.gate_threshold
            if record["selected_expert"]:
                inference_model = expert_model
        diagnostic_sequence = None
        pending_sdf = None
        started = time.perf_counter()
        try:
            generated = generate_parallel_chemsvg_v6(
                inference_model,
                pixels,
                vertex_threshold=args.vertex_threshold,
                vertex_nms_kernel_size=args.vertex_nms_kernel_size,
                proposal_keep_head=proposal_keep_head,
                proposal_keep_threshold=args.proposal_keep_threshold,
                proposal_graph_head=proposal_graph_head,
                proposal_graph_threshold=args.proposal_graph_threshold,
                proposal_graph_edge_prior_threshold=(
                    args.proposal_graph_edge_prior_threshold
                ),
                maximum_vertices=args.maximum_vertices,
                edge_threshold=args.edge_threshold,
                ocr_roi_radius=args.ocr_roi_radius,
            )
            diagnostic_sequence = generated.sequence
            if save_artifacts:
                (raw_sequence_dir / f"{molecule_id}.chemsvg.txt").write_text(
                    generated.sequence,
                    encoding="utf-8",
                )
            record["raw_vertices"] = len(generated.vertices)
            record["raw_strokes"] = len(generated.strokes)
            repaired = repair_chemsvg_v4_from_image(
                generated.sequence,
                image,
                pair_predictions=generated.pair_predictions,
                config=InkGraphRepairConfig(
                    aromatic_vote_source="vertices",
                    candidate_pair_mode=args.repair_candidate_mode,
                ),
                validate_chemistry=False,
            )
            diagnostic_sequence = repaired.sequence
            if save_artifacts:
                (
                    repaired_sequence_dir / f"{molecule_id}.chemsvg.txt"
                ).write_text(
                    repaired.sequence,
                    encoding="utf-8",
                )
            record["repaired_vertices"] = repaired.vertices
            record["repaired_strokes"] = repaired.strokes
            record["model_bridge_strokes"] = repaired.model_bridge_strokes
            predicted = mol_from_chemsvg_v4(repaired.sequence)
            pending_sdf = sdf_dir / f"pending_{molecule_id}.sdf"
            delivered = (
                write_prediction_sdf(predicted, pending_sdf, sanitize=True)
                if save_artifacts
                else predicted
            )
            actual_inchikey = Chem.MolToInchiKey(delivered)
            actual_nonstereo_key = nonstereo_inchikey(delivered)
            record.update(
                {
                    "sdf_valid": True,
                    "predicted_atoms": delivered.GetNumAtoms(),
                    "predicted_bonds": delivered.GetNumBonds(),
                    "actual_formula": rdMolDescriptors.CalcMolFormula(delivered),
                    "actual_inchikey": actual_inchikey,
                    "fingerprint_similarity": molecular_similarity(
                        expected,
                        delivered,
                    ),
                    "formula_match": rdMolDescriptors.CalcMolFormula(delivered)
                    == row["formula"],
                    "atom_count_match": delivered.GetNumAtoms() == int(row["atoms"]),
                    "bond_count_match": delivered.GetNumBonds() == int(row["bonds"]),
                    "nonstereo_structure_match": actual_nonstereo_key
                    == expected_nonstereo_key,
                    "chemical_match": actual_inchikey == row["inchikey"],
                }
            )
            if save_artifacts:
                prefix = "correct" if record["chemical_match"] else "wrong"
                final_sdf = sdf_dir / f"{prefix}_{molecule_id}.sdf"
                pending_sdf.replace(final_sdf)
                record["prediction_sdf_file"] = str(
                    final_sdf.relative_to(output_dir)
                )
        except Exception as error:
            if pending_sdf is not None:
                pending_sdf.unlink(missing_ok=True)
            record["error"] = str(error)
            record["failure_category"] = failure_category(error)
            if save_artifacts and diagnostic_sequence is not None:
                invalid_sdf = sdf_dir / f"invalid_{molecule_id}.sdf"
                try:
                    atom_count, bond_count = write_invalid_prediction_sdf(
                        diagnostic_sequence,
                        invalid_sdf,
                    )
                    record["predicted_atoms"] = atom_count
                    record["predicted_bonds"] = bond_count
                    record["prediction_sdf_file"] = str(
                        invalid_sdf.relative_to(output_dir)
                    )
                except Exception as diagnostic_error:
                    record["diagnostic_sdf_error"] = str(diagnostic_error)
        if diagnostic_sequence is not None:
            try:
                diagnostics = diagnose_chemsvg(
                    annotation,
                    diagnostic_sequence,
                    sdf_valid=bool(record["sdf_valid"]),
                    chemical_match=bool(record["chemical_match"]),
                    nonstereo_structure_match=bool(
                        record["nonstereo_structure_match"]
                    ),
                    failure_category=record.get("failure_category"),
                )
                record["diagnostics"] = diagnostics
                record["primary_error"] = diagnostics["primary_error"]
                record["error_flags"] = diagnostics["error_flags"]
            except Exception as diagnostic_error:
                record["graph_diagnostic_error"] = str(diagnostic_error)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record["seconds"] = time.perf_counter() - started
        records.append(record)
        completed.add(molecule_id)
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if len(records) % 25 == 0:
            summary = summarize_records(records)
            (output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    summary = summarize_records(records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
