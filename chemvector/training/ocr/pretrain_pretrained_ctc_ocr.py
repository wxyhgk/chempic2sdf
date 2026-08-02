#!/usr/bin/env python3
"""Train bbox CTC OCR while freezing the v0.8.6 pixel encoder."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from chemvector.ctc_ocr import (
    PretrainedPixelCTCConfig,
    PretrainedPixelCTCOCR,
    ctc_loss,
)
from chemvector.pretrain_ctc_ocr import (
    ChemicalTextCropDataset,
    collate,
    evaluate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--legacy-ocr-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-crops", type=int, default=1000)
    parser.add_argument("--validation-crops", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--color-probability", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--unfreeze-pixel-encoder", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PretrainedPixelCTCConfig(
        freeze_pixel_encoder=not args.unfreeze_pixel_encoder
    )
    model = PretrainedPixelCTCOCR(config)
    initialization = model.load_legacy_checkpoint(args.legacy_ocr_checkpoint)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    make_collate = partial(collate, maximum_bytes=config.maximum_bytes)
    validation_loader = DataLoader(
        ChemicalTextCropDataset(
            args.validation_shards,
            seed=args.seed + 1,
            crop_limit=args.validation_crops,
            color_probability=args.color_probability,
            training=False,
        ),
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=make_collate,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run = {
        "config": asdict(config),
        "initialization": initialization,
        "trainable_parameters": model.trainable_parameter_count(),
        "training_args": vars(args),
    }
    (output / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("run=" + json.dumps(run, ensure_ascii=False), flush=True)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        loader = DataLoader(
            ChemicalTextCropDataset(
                args.train_shards,
                seed=args.seed + epoch,
                crop_limit=args.train_crops,
                color_probability=args.color_probability,
                training=True,
            ),
            batch_size=args.batch_size,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            collate_fn=make_collate,
        )
        model.train()
        loss_sum = 0.0
        batches = 0
        for batch in loader:
            logits = model(batch["pixels"].to(device))
            loss = ctc_loss(
                logits,
                batch["targets"].to(device),
                batch["target_lengths"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                1.0,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach())
            batches += 1
        metrics = evaluate(model, validation_loader, device)
        metrics["train_loss"] = loss_sum / max(batches, 1)
        print(f"epoch={epoch} " + json.dumps(metrics, ensure_ascii=False), flush=True)
        payload = {
            "pretrained_ctc_ocr": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "validation": metrics,
            "initialization": initialization,
            "training_args": vars(args),
        }
        torch.save(payload, output / "last.pt")
        if metrics["exact_rate"] > best:
            best = metrics["exact_rate"]
            torch.save(payload, output / "best.pt")
            (output / "best-validation.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
