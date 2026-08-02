#!/usr/bin/env python3
"""Evaluate a ChemVector checkpoint and export generated SDF files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import time
from dataclasses import fields
from pathlib import Path

import torch
import networkx as nx
from rdkit import Chem

from Data.sdf.chemmol_sequence import mol_from_chemmolseq
from Data.sdf.chemmol_sequence_v3 import mol_from_chemmolseq_v3
from Data.sdf.chemmol_tokenizer import ChemMolTokenizer
from Data.sdf.chemmol_tokenizer_v3 import ChemMolTokenizerV3
from Data.sdf.chemsvg_sequence_v4 import mol_from_chemsvg_v4
from Data.sdf.chemsvg_tokenizer_v4 import ChemSVGTokenizerV4
from Data.sdf.chemtraverse_tokenizer_v0_1 import ChemTraverseTokenizerV01
from Data.sdf.chemtraverse_v0_1 import (
    mol_from_chemtraverse_v0_1,
    parse_chemtraverse_v0_1,
)
from chemvector.generation import generate_chemmolseq, write_sdf
from chemvector.generation_v3 import generate_chemmolseq_v3, write_sdf_v3
from chemvector.generation_v4 import generate_chemsvg_v4, write_sdf_v4
from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.train import expand_shards, load_sample_keys, make_loader, slice_batch
from chemvector.chemtraverse_teacher_metrics import (
    ChemTraverseGrammarMaskedTeacherCounts,
)

try:
    from chemvector.generation_chemtraverse_v0_1 import (
        generate_chemtraverse_v0_1,
        write_sdf as write_sdf_chemtraverse_v0_1,
    )
except ModuleNotFoundError as error:
    if error.name != "chemvector.generation_chemtraverse_v0_1":
        raise
    generate_chemtraverse_v0_1 = None
    write_sdf_chemtraverse_v0_1 = None


def _load_tokenizer(tokenizer_path: Path) -> object:
    if not tokenizer_path.is_file():
        return ChemMolTokenizer()
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    sequence_schema = tokenizer_payload.get("sequence_schema_version")
    if sequence_schema == "chemsvg-0.4":
        return ChemSVGTokenizerV4.load(tokenizer_path)
    if sequence_schema == "chemmolseq-0.3":
        return ChemMolTokenizerV3.load(tokenizer_path)
    if sequence_schema == "chemtraverse-0.1":
        return ChemTraverseTokenizerV01.load(tokenizer_path)
    return ChemMolTokenizer.load(tokenizer_path)


def _chemtraverse_generation_api():
    if generate_chemtraverse_v0_1 is None or write_sdf_chemtraverse_v0_1 is None:
        raise RuntimeError(
            "ChemTraverse generation requires "
            "chemvector.generation_chemtraverse_v0_1"
        )
    return generate_chemtraverse_v0_1, write_sdf_chemtraverse_v0_1


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ChemVectorModel, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    valid_config_keys = {item.name for item in fields(ChemVectorConfig)}
    config = ChemVectorConfig(
        **{key: value for key, value in checkpoint["config"].items() if key in valid_config_keys}
    )
    tokenizer = _load_tokenizer(checkpoint_path.parent / "tokenizer.json")
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("Checkpoint config and tokenizer vocabulary sizes differ")
    model = ChemVectorModel(config).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys)
    allowed_missing = {
        "visual_aux_head.weight",
        "visual_aux_head.bias",
        "vertex_head.role_heatmap.weight",
        "vertex_head.role_heatmap.bias",
    }
    if unexpected or missing - allowed_missing:
        raise ValueError(
            f"Checkpoint architecture mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def evaluate_teacher_forcing(
    model: ChemVectorModel,
    loader,
    device: torch.device,
    sample_limit: int,
    tokenizer: object | None = None,
) -> dict:
    loss_sum = 0.0
    correct_tokens = 0
    token_count = 0
    exact_sequences = 0
    sample_count = 0
    token_totals = None
    token_correct = None
    chemtraverse_counts = None
    if tokenizer is not None:
        token_totals = torch.zeros(tokenizer.vocab_size, dtype=torch.long)
        token_correct = torch.zeros(tokenizer.vocab_size, dtype=torch.long)
        if isinstance(tokenizer, ChemTraverseTokenizerV01):
            chemtraverse_counts = ChemTraverseGrammarMaskedTeacherCounts()
    started = time.perf_counter()
    for batch in loader:
        remaining = sample_limit - sample_count
        if remaining <= 0:
            break
        if batch["input_ids"].size(0) > remaining:
            batch = slice_batch(batch, stop=remaining)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(pixel_values, input_ids, attention_mask)
        labels = input_ids[:, 1:]
        valid = labels != model.config.pad_token_id
        predictions = output["logits"].argmax(dim=-1)
        if chemtraverse_counts is not None:
            chemtraverse_counts.update(
                output["logits"],
                input_ids,
                tokenizer,
                pad_token_id=model.config.pad_token_id,
            )
        matches = (predictions == labels) | ~valid
        batch_tokens = int(valid.sum())
        loss_sum += float(output["loss"]) * batch_tokens
        correct_tokens += int(((predictions == labels) & valid).sum())
        if token_totals is not None:
            valid_labels = labels[valid].detach().cpu()
            correct_labels = labels[(predictions == labels) & valid].detach().cpu()
            token_totals += torch.bincount(
                valid_labels,
                minlength=tokenizer.vocab_size,
            )
            token_correct += torch.bincount(
                correct_labels,
                minlength=tokenizer.vocab_size,
            )
        token_count += batch_tokens
        exact_sequences += int(matches.all(dim=1).sum())
        sample_count += input_ids.size(0)
    elapsed = time.perf_counter() - started
    result = {
        "samples": sample_count,
        "tokens": token_count,
        "loss": loss_sum / max(token_count, 1),
        "token_accuracy": correct_tokens / max(token_count, 1),
        "exact_sequence_accuracy": exact_sequences / max(sample_count, 1),
        "samples_per_second": sample_count / max(elapsed, 1e-9),
    }
    if token_totals is not None:
        result.update(_teacher_token_breakdown(tokenizer, token_totals, token_correct))
    if chemtraverse_counts is not None:
        result["chemtraverse_grammar_masked"] = chemtraverse_counts.metrics()
    return result


def _teacher_token_breakdown(
    tokenizer: object,
    totals: torch.Tensor,
    correct: torch.Tensor,
) -> dict:
    traversal_controls = {
        "[COMPONENT]",
        "[DOWN]",
        "[UP]",
        "[LINK]",
        "[COMPONENT_END]",
        "[CHEMTRAVERSE_END]",
        "[EOS]",
    }

    def category(token: str) -> str:
        if token in traversal_controls:
            return "traversal_control"
        for prefix, name in (
            ("n:", "numeric"),
            ("role:", "atom_role"),
            ("el:", "element"),
            ("vis:", "display_bond_type"),
            ("bt:", "chemical_bond_type"),
            ("dir:", "bond_direction"),
            ("st:", "bond_stereo"),
            ("chi:", "atom_chirality"),
            ("ori:", "bond_orientation"),
            ("byte:", "text_byte"),
        ):
            if token.startswith(prefix):
                return name
        return "grammar_marker" if token.startswith("[") else "other"

    category_counts: dict[str, list[int]] = {}
    control_tokens = {}
    for token_id, token in enumerate(tokenizer.id_to_token):
        total = int(totals[token_id])
        if not total:
            continue
        matched = int(correct[token_id])
        values = category_counts.setdefault(category(token), [0, 0])
        values[0] += matched
        values[1] += total
        if token in traversal_controls:
            control_tokens[token] = {
                "correct": matched,
                "total": total,
                "accuracy": matched / total,
            }
    return {
        "token_categories": {
            name: {
                "correct": values[0],
                "total": values[1],
                "accuracy": values[0] / values[1],
            }
            for name, values in sorted(category_counts.items())
        },
        "control_tokens": dict(sorted(control_tokens.items())),
    }


def normalized_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _scene_graph(scene: dict) -> nx.Graph:
    graph = nx.Graph()
    coordinate_to_index = {}
    for index, junction in enumerate(scene["junctions"]):
        coordinate_to_index[tuple(junction["coordinate"])] = index
        atom = junction["atom"]
        graph.add_node(
            index,
            element=atom["element"],
            formal_charge=atom["formal_charge"],
            isotope=atom["isotope"],
            aromatic=atom["aromatic"],
            explicit_h_count=atom["explicit_h_count"],
            no_implicit=atom["no_implicit"],
            radical_electrons=atom["radical_electrons"],
            canonical_chiral_tag=atom["canonical_chiral_tag"],
        )
    for bond in scene["bond_objects"]:
        begin, end = (
            coordinate_to_index[tuple(coordinate)] for coordinate in bond["endpoints"]
        )
        graph.add_edge(
            begin,
            end,
            chemical_type=bond["chemical_type"],
            direction=bond["direction"],
            stereo=bond["stereo"],
        )
    return graph


def compare_chemtraverse_scenes(expected: dict, predicted: dict) -> dict[str, bool | int]:
    expected_graph = _scene_graph(expected)
    predicted_graph = _scene_graph(predicted)
    atom_keys = (
        "element",
        "formal_charge",
        "isotope",
        "aromatic",
        "explicit_h_count",
        "no_implicit",
        "radical_electrons",
        "canonical_chiral_tag",
    )
    bond_keys = ("chemical_type", "direction", "stereo")

    def atom_match(left: dict, right: dict) -> bool:
        return all(left[key] == right[key] for key in atom_keys)

    def bond_match(left: dict, right: dict) -> bool:
        return all(left[key] == right[key] for key in bond_keys)

    skeleton_exact = nx.is_isomorphic(expected_graph, predicted_graph)
    atom_typed = skeleton_exact and nx.is_isomorphic(
        expected_graph,
        predicted_graph,
        node_match=atom_match,
    )
    bond_typed = skeleton_exact and nx.is_isomorphic(
        expected_graph,
        predicted_graph,
        edge_match=bond_match,
    )
    return {
        "expected_atom_count": expected_graph.number_of_nodes(),
        "predicted_atom_count": predicted_graph.number_of_nodes(),
        "expected_bond_count": expected_graph.number_of_edges(),
        "predicted_bond_count": predicted_graph.number_of_edges(),
        "atom_count_match": expected_graph.number_of_nodes()
        == predicted_graph.number_of_nodes(),
        "bond_count_match": expected_graph.number_of_edges()
        == predicted_graph.number_of_edges(),
        "count_pair_exact": (
            expected_graph.number_of_nodes() == predicted_graph.number_of_nodes()
            and expected_graph.number_of_edges() == predicted_graph.number_of_edges()
        ),
        "skeleton_exact": skeleton_exact,
        "atom_typed_skeleton_exact": atom_typed,
        "bond_typed_skeleton_exact": bond_typed,
        "typed_graph_exact": atom_typed
        and bond_typed
        and nx.is_isomorphic(
            expected_graph,
            predicted_graph,
            node_match=atom_match,
            edge_match=bond_match,
        ),
    }


def _failure_bucket(record: dict) -> str:
    if not record["completed"]:
        return "generation_incomplete"
    if not record["grammar_valid"]:
        return "grammar_invalid"
    if not record["scene_valid"]:
        return "scene_invalid"
    if not record["count_pair_exact"]:
        return "count_mismatch"
    if not record["skeleton_exact"]:
        return "topology_mismatch"
    if not record["atom_typed_skeleton_exact"]:
        return "atom_type_mismatch"
    if not record["bond_typed_skeleton_exact"]:
        return "bond_type_mismatch"
    if not record["mol_valid"]:
        return "chemical_invalid"
    if not record["sdf_valid"]:
        return "sdf_invalid"
    if not record["chemical_match"]:
        return "chemical_mismatch"
    return "correct"


def evaluate_generation(
    model: ChemVectorModel,
    loader,
    tokenizer: object,
    device: torch.device,
    sample_limit: int,
    output_dir: Path,
    *,
    beam_width: int = 1,
    visual_first: bool = False,
    ocr_fusion: bool = False,
    stroke_scoring: bool = False,
    stroke_threshold: float = 0.5,
    oracle_counts: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    is_v3 = isinstance(tokenizer, ChemMolTokenizerV3)
    is_v4 = isinstance(tokenizer, ChemSVGTokenizerV4)
    is_traverse = isinstance(tokenizer, ChemTraverseTokenizerV01)
    traverse_generator = None
    traverse_sdf_writer = None
    if is_traverse:
        traverse_generator, traverse_sdf_writer = _chemtraverse_generation_api()
    elif oracle_counts:
        raise ValueError("Oracle counts require ChemTraverse v0.1")
    records = []
    started = time.perf_counter()
    for batch in loader:
        if len(records) >= sample_limit:
            break
        key = batch["keys"][0]
        expected_sequence = batch["target_sequences"][0]
        expected_smiles = batch["expected_smiles"][0]
        expected_inchikey = batch["expected_inchikeys"][0]
        expected_scene = (
            parse_chemtraverse_v0_1(expected_sequence) if is_traverse else None
        )
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        if is_traverse:
            generation_kwargs = {"beam_width": beam_width}
            if oracle_counts:
                generation_kwargs.update(
                    expected_atom_count=len(expected_scene["junctions"]),
                    expected_bond_count=len(expected_scene["bond_objects"]),
                )
            result = traverse_generator(
                model,
                pixel_values,
                tokenizer,
                **generation_kwargs,
            )
        elif is_v4:
            result = generate_chemsvg_v4(
                model,
                pixel_values,
                tokenizer,
                beam_width=beam_width,
                visual_first=visual_first,
                ocr_fusion=ocr_fusion,
                stroke_scoring=stroke_scoring,
                stroke_threshold=stroke_threshold,
            )
        elif is_v3:
            result = generate_chemmolseq_v3(
                model,
                pixel_values,
                tokenizer,
                beam_width=beam_width,
            )
        else:
            result = generate_chemmolseq(model, pixel_values, tokenizer)
        record = {
            "key": key,
            "completed": result.completed,
            "chemically_valid": result.chemically_valid,
            "tokens": len(result.input_ids),
            "exact_sequence": result.sequence == expected_sequence,
            "chemical_match": False,
            "error": result.error,
            "oracle_counts": oracle_counts,
        }
        if is_traverse:
            record.update(
                {
                    "grammar_valid": False,
                    "scene_valid": False,
                    "mol_valid": False,
                    "sdf_valid": False,
                    "sdf_roundtrip_match": False,
                    "grammar_error": None,
                    "scene_error": None,
                    "mol_error": None,
                    "sdf_error": None,
                    "expected_atom_count": len(expected_scene["junctions"]),
                    "predicted_atom_count": 0,
                    "expected_bond_count": len(expected_scene["bond_objects"]),
                    "predicted_bond_count": 0,
                    "atom_count_match": False,
                    "bond_count_match": False,
                    "count_pair_exact": False,
                    "skeleton_exact": False,
                    "atom_typed_skeleton_exact": False,
                    "bond_typed_skeleton_exact": False,
                    "typed_graph_exact": False,
                }
            )
        if result.sequence is not None:
            if is_traverse:
                sequence_suffix = "chemtraverse.txt"
            elif is_v4:
                sequence_suffix = "chemsvg.txt"
            else:
                sequence_suffix = "chemmolseq.txt"
            (output_dir / f"{key}.{sequence_suffix}").write_text(
                result.sequence,
                encoding="utf-8",
            )
            expected_ids = tokenizer.encode(expected_sequence)
            divergence = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(result.input_ids, expected_ids)
                    )
                    if actual != expected
                ),
                min(len(result.input_ids), len(expected_ids)),
            )
            record["first_divergence_token"] = divergence
            if divergence < len(result.input_ids):
                record["actual_token_at_divergence"] = tokenizer.id_to_token[
                    result.input_ids[divergence]
                ]
            if divergence < len(expected_ids):
                record["expected_token_at_divergence"] = tokenizer.id_to_token[
                    expected_ids[divergence]
                ]
            if is_traverse:
                try:
                    tokenizer.encode(result.sequence)
                    record["grammar_valid"] = True
                except Exception as error:
                    record["grammar_error"] = str(error)
                    if record["error"] is None:
                        record["error"] = str(error)

                scene = None
                try:
                    scene = parse_chemtraverse_v0_1(result.sequence)
                    record["scene_valid"] = True
                    record.update(compare_chemtraverse_scenes(expected_scene, scene))
                    (output_dir / f"{key}.chemscene.json").write_text(
                        json.dumps(scene, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                except Exception as error:
                    record["scene_error"] = str(error)
                    if record["error"] is None:
                        record["error"] = str(error)

                mol = None
                if scene is not None:
                    try:
                        mol = mol_from_chemtraverse_v0_1(result.sequence)
                        record["mol_valid"] = True
                    except Exception as error:
                        record["mol_error"] = str(error)
                        if record["error"] is None:
                            record["error"] = str(error)

                if mol is not None:
                    sdf_path = output_dir / f"{key}.sdf"
                    try:
                        traverse_sdf_writer(result.sequence, sdf_path)
                        sdf_molecules = [
                            item
                            for item in Chem.SDMolSupplier(str(sdf_path), removeHs=False)
                            if item is not None
                        ]
                        if len(sdf_molecules) != 1:
                            raise ValueError(
                                "ChemTraverse SDF must contain exactly one valid molecule"
                            )
                        sdf_mol = sdf_molecules[0]
                        record["sdf_valid"] = True
                        actual_smiles = normalized_smiles(sdf_mol)
                        actual_inchikey = Chem.MolToInchiKey(sdf_mol)
                        record["sdf_roundtrip_match"] = (
                            actual_smiles == normalized_smiles(mol)
                            and actual_inchikey == Chem.MolToInchiKey(mol)
                        )
                        expected_mol = Chem.MolFromSmiles(expected_smiles)
                        record.update(
                            {
                                "actual_smiles": actual_smiles,
                                "actual_inchikey": actual_inchikey,
                                "chemical_match": (
                                    expected_mol is not None
                                    and actual_smiles == normalized_smiles(expected_mol)
                                    and actual_inchikey == expected_inchikey
                                ),
                            }
                        )
                    except Exception as error:
                        record["sdf_error"] = str(error)
                        if record["error"] is None:
                            record["error"] = str(error)
            else:
                try:
                    if is_v4:
                        mol = mol_from_chemsvg_v4(result.sequence)
                    elif is_v3:
                        mol = mol_from_chemmolseq_v3(result.sequence)
                    else:
                        mol = mol_from_chemmolseq(result.sequence)
                    actual_smiles = normalized_smiles(mol)
                    actual_inchikey = Chem.MolToInchiKey(mol)
                    expected_mol = Chem.MolFromSmiles(expected_smiles)
                    record.update(
                        {
                            "actual_smiles": actual_smiles,
                            "actual_inchikey": actual_inchikey,
                            "chemical_match": (
                                expected_mol is not None
                                and actual_smiles == normalized_smiles(expected_mol)
                                and actual_inchikey == expected_inchikey
                            ),
                        }
                    )
                    if is_v4:
                        sdf_writer = write_sdf_v4
                    elif is_v3:
                        sdf_writer = write_sdf_v3
                    else:
                        sdf_writer = write_sdf
                    sdf_writer(result.sequence, output_dir / f"{key}.sdf")
                except Exception as error:
                    record["error"] = str(error)
        records.append(record)
        if is_traverse:
            record["failure_bucket"] = _failure_bucket(record)
        print(
            f"generate key={key} completed={record['completed']} "
            f"tokens={record['tokens']} chemical_match={record['chemical_match']}"
        )
    elapsed = time.perf_counter() - started
    summary = {
        "samples": len(records),
        "oracle_counts": oracle_counts,
        "beam_width": beam_width,
        "visual_first": visual_first,
        "ocr_fusion": ocr_fusion,
        "stroke_scoring": stroke_scoring,
        "stroke_threshold": stroke_threshold,
        "completed": sum(record["completed"] for record in records),
        "chemically_valid": sum(record["chemically_valid"] for record in records),
        "exact_sequences": sum(record["exact_sequence"] for record in records),
        "chemical_matches": sum(record["chemical_match"] for record in records),
        "seconds": elapsed,
        "records": records,
    }
    if is_traverse:
        samples = max(len(records), 1)
        grammar_valid = sum(record["grammar_valid"] for record in records)
        scene_valid = sum(record["scene_valid"] for record in records)
        mol_valid = sum(record["mol_valid"] for record in records)
        sdf_valid = sum(record["sdf_valid"] for record in records)
        sdf_roundtrip_matches = sum(
            record["sdf_roundtrip_match"] for record in records
        )
        summary.update(
            {
                "sequence_schema_version": "chemtraverse-0.1",
                "grammar_valid": grammar_valid,
                "scene_valid": scene_valid,
                "mol_valid": mol_valid,
                "sdf_valid": sdf_valid,
                "sdf_roundtrip_matches": sdf_roundtrip_matches,
                "grammar_valid_rate": grammar_valid / samples,
                "scene_valid_rate": scene_valid / samples,
                "mol_valid_rate": mol_valid / samples,
                "sdf_valid_rate": sdf_valid / samples,
                "sdf_roundtrip_match_rate": sdf_roundtrip_matches / samples,
                "chemical_match_rate": summary["chemical_matches"] / samples,
                "count_pair_exact": sum(
                    record["count_pair_exact"] for record in records
                ),
                "skeleton_exact": sum(record["skeleton_exact"] for record in records),
                "atom_typed_skeleton_exact": sum(
                    record["atom_typed_skeleton_exact"] for record in records
                ),
                "bond_typed_skeleton_exact": sum(
                    record["bond_typed_skeleton_exact"] for record in records
                ),
                "typed_graph_exact": sum(
                    record["typed_graph_exact"] for record in records
                ),
                "count_pair_exact_rate": sum(
                    record["count_pair_exact"] for record in records
                )
                / samples,
                "skeleton_exact_rate": sum(
                    record["skeleton_exact"] for record in records
                )
                / samples,
                "atom_typed_skeleton_exact_rate": sum(
                    record["atom_typed_skeleton_exact"] for record in records
                )
                / samples,
                "bond_typed_skeleton_exact_rate": sum(
                    record["bond_typed_skeleton_exact"] for record in records
                )
                / samples,
                "typed_graph_exact_rate": sum(
                    record["typed_graph_exact"] for record in records
                )
                / samples,
                "failure_buckets": dict(
                    sorted(Counter(record["failure_bucket"] for record in records).items())
                ),
            }
        )
    (output_dir / "generation_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--generate-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument(
        "--oracle-counts",
        action="store_true",
        help="Diagnostic oracle: constrain ChemTraverse to GT atom/bond counts",
    )
    parser.add_argument("--visual-first", action="store_true")
    parser.add_argument("--ocr-fusion", action="store_true")
    parser.add_argument("--stroke-scoring", action="store_true")
    parser.add_argument("--stroke-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="outputs/evaluations/chemvector-evaluation")
    parser.add_argument("--reject-keys-file")
    parser.add_argument(
        "--sample-manifest",
        help="JSONL manifest whose sample_key values are the only evaluated samples",
    )
    parser.add_argument(
        "--target-molecule-id",
        help="Replace unobservable ChemTraverse molecule_id with a fixed target value",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    if args.samples < 1 or args.generate_samples < 0 or args.beam_width < 1:
        raise ValueError(
            "--samples and --beam-width must be positive; "
            "--generate-samples must be nonnegative"
        )
    device = torch.device("cuda")
    model, tokenizer = load_model(Path(args.checkpoint), device)
    if args.target_molecule_id is not None and not isinstance(
        tokenizer, ChemTraverseTokenizerV01
    ):
        raise ValueError("Fixed target molecule ID requires ChemTraverse v0.1")
    if args.target_molecule_id == "":
        raise ValueError("--target-molecule-id must be non-empty")
    if args.oracle_counts and not isinstance(tokenizer, ChemTraverseTokenizerV01):
        raise ValueError("--oracle-counts requires ChemTraverse v0.1")
    if args.beam_width > 1 and not isinstance(
        tokenizer,
        (ChemMolTokenizerV3, ChemSVGTokenizerV4, ChemTraverseTokenizerV01),
    ):
        raise ValueError(
            "Beam search requires ChemMolSeq v0.3, ChemSVG v0.4, "
            "or ChemTraverse v0.1"
        )
    if (args.visual_first or args.ocr_fusion or args.stroke_scoring) and not isinstance(
        tokenizer,
        ChemSVGTokenizerV4,
    ):
        raise ValueError("Visual-first and OCR fusion require ChemSVG v0.4")
    shards = expand_shards(args.shards)
    reject_keys = set()
    if args.reject_keys_file:
        reject_keys = {
            line.strip()
            for line in Path(args.reject_keys_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    include_keys = load_sample_keys(args.sample_manifest)
    print(f"include_keys={len(include_keys) if include_keys is not None else 'all'}")
    teacher_loader = make_loader(
        shards,
        tokenizer,
        model.config,
        batch_size=args.batch_size,
        image_size=args.image_size,
        workers=args.workers,
        shuffle=False,
        reject_keys=reject_keys,
        include_keys=include_keys,
        target_molecule_id=args.target_molecule_id,
    )
    teacher_metrics = evaluate_teacher_forcing(
        model,
        teacher_loader,
        device,
        args.samples,
        tokenizer,
    )
    print("teacher_forcing=" + json.dumps(teacher_metrics, ensure_ascii=False))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "teacher_forcing_metrics.json").write_text(
        json.dumps(teacher_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.generate_samples:
        generation_loader = make_loader(
            shards,
            tokenizer,
            model.config,
            batch_size=1,
            image_size=args.image_size,
            workers=args.workers,
            shuffle=False,
            reject_keys=reject_keys,
            include_keys=include_keys,
            target_molecule_id=args.target_molecule_id,
        )
        generation_metrics = evaluate_generation(
            model,
            generation_loader,
            tokenizer,
            device,
            args.generate_samples,
            output_dir / "generated",
            beam_width=args.beam_width,
            visual_first=args.visual_first,
            ocr_fusion=args.ocr_fusion,
            stroke_scoring=args.stroke_scoring,
            stroke_threshold=args.stroke_threshold,
            oracle_counts=args.oracle_counts,
        )
        print(
            "generation="
            + json.dumps(
                {key: value for key, value in generation_metrics.items() if key != "records"},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
