#!/usr/bin/env python3
"""Run a short real-data training smoke test on one GPU."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from Data.sdf.chemmol_tokenizer import ChemMolTokenizer
from chemvector.data import ChemVectorCollator, ChemVectorIterableDataset
from chemvector.model import ChemVectorConfig, ChemVectorModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard",
        default="Data/sdf/webdataset_full_v0/shards/train-000000.tar",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ChemVector smoke test")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    tokenizer = ChemMolTokenizer()
    if args.tiny:
        config = ChemVectorConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            d_model=256,
            decoder_layers=2,
            decoder_heads=4,
            decoder_ffn_dim=1024,
            vision_dims=(48, 96, 192, 256),
            vision_depths=(1, 1, 2, 1),
        )
    else:
        config = ChemVectorConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
        )
    model = ChemVectorModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    dataset = ChemVectorIterableDataset(args.shard)
    collator = ChemVectorCollator(
        tokenizer,
        image_size=args.image_size,
        max_length=config.max_target_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=collator,
    )

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"parameters={model.parameter_count():,}")
    print(f"vocab_size={tokenizer.vocab_size}")
    print(f"image_size={args.image_size}")
    model.train()
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        batch = next(iterator)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(pixel_values, input_ids, attention_mask)
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(
            f"step={step} loss={output['loss'].item():.6f} "
            f"tokens={int(attention_mask.sum())} "
            f"seconds={time.perf_counter() - started:.3f} "
            f"max_memory_gib={allocated:.3f}"
        )


if __name__ == "__main__":
    main()
