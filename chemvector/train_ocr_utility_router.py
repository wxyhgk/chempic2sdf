#!/usr/bin/env python3
"""Train a conservative router that selectively adopts the newer OCR output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from chemvector.ocr_utility_router import OCRUtilityRouter, conservative_frontier


LABEL_WEIGHTS = {
    "new_rescue": 1.0,
    "new_break": 8.0,
    "both_exact": 0.1,
    "both_wrong": 1.0,
}


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line
        )
    return [row for row in rows if row["old_text"] != row["new_text"]]


def pack_rows(rows: list[dict], names: list[str] | None = None):
    if not rows:
        raise ValueError("OCR utility dataset has no disagreements")
    feature_names = names or sorted(rows[0]["features"])
    features = torch.tensor(
        [[float(row["features"][name]) for name in feature_names] for row in rows],
        dtype=torch.float32,
    )
    labels = [str(row["label"]) for row in rows]
    targets = torch.tensor(
        [label == "new_rescue" for label in labels], dtype=torch.float32
    )
    weights = torch.tensor([LABEL_WEIGHTS[label] for label in labels])
    domains = [str(row["domain"]) for row in rows]
    return feature_names, features, targets, weights, labels, domains


@torch.no_grad()
def evaluate(model, features, labels, domains):
    probabilities = model(features).sigmoid().cpu()
    return probabilities, conservative_frontier(probabilities, labels, domains)


def score(frontier: dict) -> tuple:
    return tuple(
        frontier[key]["overall"]["net_text_exact_gain"]
        for key in ("0", "1", "2", "5", "10")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", nargs="+", required=True)
    parser.add_argument("--validation-data", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_rows = load_rows(args.train_data)
    validation_rows = load_rows(args.validation_data)
    names, train_x, train_y, train_w, _, _ = pack_rows(train_rows)
    _, validation_x, _, _, validation_labels, validation_domains = pack_rows(
        validation_rows, names
    )
    mean = train_x.mean(0)
    std = train_x.std(0).clamp_min(1e-5)
    train_x = ((train_x - mean) / std).to(device)
    train_y = train_y.to(device)
    train_w = train_w.to(device)
    validation_x = ((validation_x - mean) / std).to(device)
    model = OCRUtilityRouter(len(names), args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    best_score = None
    best_payload = None
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            len(train_x), (args.batch_size,), generator=generator
        ).to(device)
        logits = model(train_x[indices])
        losses = F.binary_cross_entropy_with_logits(
            logits, train_y[indices], reduction="none"
        )
        loss = (losses * train_w[indices]).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % args.validate_every and step != args.steps:
            continue
        model.eval()
        probabilities, frontier = evaluate(
            model, validation_x, validation_labels, validation_domains
        )
        current_score = score(frontier)
        print(
            f"step={step} loss={float(loss.detach()):.6f} "
            f"break0_gain={frontier['0']['overall']['net_text_exact_gain']} "
            f"break2_gain={frontier['2']['overall']['net_text_exact_gain']}",
            flush=True,
        )
        if best_score is None or current_score > best_score:
            best_score = current_score
            best_payload = {
                "checkpoint_schema_version": "ocr-utility-router-0.1",
                "router": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "input_dim": len(names),
                "hidden_dim": args.hidden_dim,
                "feature_names": names,
                "feature_mean": mean,
                "feature_std": std,
                "step": step,
                "validation_frontier": frontier,
                "validation_probabilities": probabilities,
                "training_args": vars(args),
                "train_disagreements": len(train_rows),
                "validation_disagreements": len(validation_rows),
            }
        model.train()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"OCR router output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best_payload, output / "best.pt")
    (output / "validation.json").write_text(
        json.dumps(best_payload["validation_frontier"], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(best_payload["validation_frontier"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
