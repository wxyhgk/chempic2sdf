#!/usr/bin/env python3
"""Evaluate a ChemVector checkpoint and export generated SDF files."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

import torch
from rdkit import Chem

from Data.sdf.chemmol_sequence import mol_from_chemmolseq
from Data.sdf.chemmol_sequence_v3 import mol_from_chemmolseq_v3
from Data.sdf.chemmol_tokenizer import ChemMolTokenizer
from Data.sdf.chemmol_tokenizer_v3 import ChemMolTokenizerV3
from chemvector.generation import generate_chemmolseq, write_sdf
from chemvector.generation_v3 import generate_chemmolseq_v3, write_sdf_v3
from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.train import expand_shards, make_loader, slice_batch


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ChemVectorModel, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    valid_config_keys = {item.name for item in fields(ChemVectorConfig)}
    config = ChemVectorConfig(
        **{key: value for key, value in checkpoint["config"].items() if key in valid_config_keys}
    )
    tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    if tokenizer_path.is_file():
        tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        tokenizer = (
            ChemMolTokenizerV3.load(tokenizer_path)
            if tokenizer_payload.get("sequence_schema_version") == "chemmolseq-0.3"
            else ChemMolTokenizer.load(tokenizer_path)
        )
    else:
        tokenizer = ChemMolTokenizer()
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("Checkpoint config and tokenizer vocabulary sizes differ")
    model = ChemVectorModel(config).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys)
    allowed_missing = {"visual_aux_head.weight", "visual_aux_head.bias"}
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
) -> dict[str, float | int]:
    loss_sum = 0.0
    correct_tokens = 0
    token_count = 0
    exact_sequences = 0
    sample_count = 0
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
        matches = (predictions == labels) | ~valid
        batch_tokens = int(valid.sum())
        loss_sum += float(output["loss"]) * batch_tokens
        correct_tokens += int(((predictions == labels) & valid).sum())
        token_count += batch_tokens
        exact_sequences += int(matches.all(dim=1).sum())
        sample_count += input_ids.size(0)
    elapsed = time.perf_counter() - started
    return {
        "samples": sample_count,
        "tokens": token_count,
        "loss": loss_sum / max(token_count, 1),
        "token_accuracy": correct_tokens / max(token_count, 1),
        "exact_sequence_accuracy": exact_sequences / max(sample_count, 1),
        "samples_per_second": sample_count / max(elapsed, 1e-9),
    }


def normalized_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def evaluate_generation(
    model: ChemVectorModel,
    loader,
    tokenizer: ChemMolTokenizer,
    device: torch.device,
    sample_limit: int,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    is_v3 = isinstance(tokenizer, ChemMolTokenizerV3)
    records = []
    started = time.perf_counter()
    for batch in loader:
        if len(records) >= sample_limit:
            break
        key = batch["keys"][0]
        expected_sequence = batch["target_sequences"][0]
        expected_smiles = batch["expected_smiles"][0]
        expected_inchikey = batch["expected_inchikeys"][0]
        generator = generate_chemmolseq_v3 if is_v3 else generate_chemmolseq
        result = generator(
            model, batch["pixel_values"].to(device, non_blocking=True), tokenizer
        )
        record = {
            "key": key,
            "completed": result.completed,
            "chemically_valid": result.chemically_valid,
            "tokens": len(result.input_ids),
            "exact_sequence": result.sequence == expected_sequence,
            "chemical_match": False,
            "error": result.error,
        }
        if result.sequence is not None:
            (output_dir / f"{key}.chemmolseq.txt").write_text(
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
            try:
                mol = (
                    mol_from_chemmolseq_v3(result.sequence)
                    if is_v3
                    else mol_from_chemmolseq(result.sequence)
                )
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
                sdf_writer = write_sdf_v3 if is_v3 else write_sdf
                sdf_writer(result.sequence, output_dir / f"{key}.sdf")
            except Exception as error:
                record["error"] = str(error)
        records.append(record)
        print(
            f"generate key={key} completed={record['completed']} "
            f"tokens={record['tokens']} chemical_match={record['chemical_match']}"
        )
    elapsed = time.perf_counter() - started
    summary = {
        "samples": len(records),
        "completed": sum(record["completed"] for record in records),
        "chemically_valid": sum(record["chemically_valid"] for record in records),
        "exact_sequences": sum(record["exact_sequence"] for record in records),
        "chemical_matches": sum(record["chemical_match"] for record in records),
        "seconds": elapsed,
        "records": records,
    }
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
    parser.add_argument("--output-dir", default="outputs/chemvector-evaluation")
    parser.add_argument("--reject-keys-file")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    if args.samples < 1 or args.generate_samples < 0:
        raise ValueError("Sample limits must be nonnegative and --samples must be positive")
    device = torch.device("cuda")
    model, tokenizer = load_model(Path(args.checkpoint), device)
    shards = expand_shards(args.shards)
    reject_keys = set()
    if args.reject_keys_file:
        reject_keys = {
            line.strip()
            for line in Path(args.reject_keys_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    teacher_loader = make_loader(
        shards,
        tokenizer,
        model.config,
        batch_size=args.batch_size,
        image_size=args.image_size,
        workers=args.workers,
        shuffle=False,
        reject_keys=reject_keys,
    )
    teacher_metrics = evaluate_teacher_forcing(
        model,
        teacher_loader,
        device,
        args.samples,
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
        )
        generation_metrics = evaluate_generation(
            model,
            generation_loader,
            tokenizer,
            device,
            args.generate_samples,
            output_dir / "generated",
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
