#!/usr/bin/env python3
"""Evaluate a frozen Bond utility router on untouched internal data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from chemvector.bond_utility_router import BondUtilityRouter, route_summary
from chemvector.train_bond_utility_router import load_rows, pack_rows


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-data", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--break-budget", choices=("0", "1", "2", "5", "10"), default="0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    rows = load_rows(args.test_data)
    _, features, _, _, labels, domains = pack_rows(rows, payload["feature_names"])
    device = torch.device(args.device)
    model = BondUtilityRouter(payload["input_dim"], payload["hidden_dim"]).to(device)
    model.load_state_dict(payload["router"])
    model.eval().requires_grad_(False)
    normalized = (
        features - payload["feature_mean"]
    ) / payload["feature_std"]
    probabilities = model(normalized.to(device)).sigmoid().cpu()
    selected_validation = payload["validation_frontier"][args.break_budget]
    threshold = float(selected_validation["threshold"])
    summary = route_summary(probabilities, labels, domains, threshold)
    summary["protocol"] = {
        "checkpoint": args.checkpoint,
        "test_data": args.test_data,
        "break_budget_selected_on_validation": int(args.break_budget),
        "threshold_selected_on_validation": threshold,
        "uses_test_labels_for_threshold_selection": False,
        "uses_uspto_for_training_or_threshold_selection": False,
        "uses_mr_tadf_for_training_or_threshold_selection": False,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
