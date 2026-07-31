#!/usr/bin/env python3
"""Evaluate a frozen OCR utility router on untouched internal teacher rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from chemvector.ocr_utility_router import OCRUtilityRouter, route_summary


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-data", nargs="+", required=True)
    parser.add_argument("--break-budget", default="0")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema_version") != "ocr-utility-router-0.1":
        raise ValueError("OCR utility router checkpoint schema differs")
    frontier = payload["validation_frontier"].get(args.break_budget)
    if frontier is None:
        raise ValueError("Requested OCR router break budget is unavailable")
    rows = []
    for path in args.test_data:
        rows.extend(
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line
        )
    disagreements = [row for row in rows if row["old_text"] != row["new_text"]]
    names = payload["feature_names"]
    features = torch.tensor(
        [[float(row["features"][name]) for name in names] for row in disagreements],
        dtype=torch.float32,
    )
    device = torch.device(args.device)
    model = OCRUtilityRouter(payload["input_dim"], payload["hidden_dim"]).to(device)
    model.load_state_dict(payload["router"])
    model.eval().requires_grad_(False)
    normalized = (
        (features - payload["feature_mean"]) / payload["feature_std"]
    ).to(device)
    with torch.no_grad():
        probabilities = model(normalized).sigmoid().cpu()
    labels = [str(row["label"]) for row in disagreements]
    domains = [str(row["domain"]) for row in disagreements]
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(frontier["threshold"])
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("OCR router threshold must lie in [0, 1]")
    summary = route_summary(probabilities, labels, domains, threshold)
    gate = {
        "zero_break": summary["overall"]["new_break"] == 0,
        "positive_rescue": summary["overall"]["new_rescue"] > 0,
        "positive_rescue_each_domain": all(
            value["new_rescue"] > 0 for value in summary["domains"].values()
        ),
    }
    gate["passed"] = all(gate.values())
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "break_budget": args.break_budget,
        "validation_threshold": float(frontier["threshold"]),
        "deployed_threshold": threshold,
        "threshold_source": (
            "explicit_fixed_floor"
            if args.threshold is not None
            else "validation_frontier"
        ),
        "test_rows": len(rows),
        "test_disagreements": len(disagreements),
        "summary": summary,
        "gate": gate,
        "uses_uspto": False,
        "uses_mr_tadf": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
