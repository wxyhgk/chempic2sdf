#!/usr/bin/env python3
"""Evaluate a proposal graph checkpoint without mutating model state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chemvector.data import ChemVectorIterableDataset, load_reject_keys
from chemvector.pretrain_proposal_graph import evaluate
from chemvector.pretrain_proposal_keep import parse_thresholds
from chemvector.pretrain_vertex import load_vertex_model
from chemvector.proposal_graph import ProposalGraphCollator, load_proposal_graph_checkpoint
from chemvector.proposal_keep import load_proposal_keep_checkpoint
from chemvector.train import expand_shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--proposal-graph-checkpoint", required=True)
    parser.add_argument("--proposal-keep-checkpoint")
    parser.add_argument("--validation-shards", nargs="+", required=True)
    parser.add_argument("--validation-reject-keys-file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--validation-batches", type=int, default=0)
    parser.add_argument(
        "--node-thresholds",
        type=parse_thresholds,
        default=parse_thresholds(
            "0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.81,0.82,0.83,0.84,"
            "0.85,0.86,0.87,0.88,0.89,0.9,0.95"
        ),
    )
    parser.add_argument(
        "--edge-thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.5,0.6,0.7,0.8,0.85,0.9,0.95"),
    )
    parser.add_argument("--edge-loss-weight", type=float, default=1.0)
    parser.add_argument("--negative-ratio", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if min(args.batch_size, args.negative_ratio) < 1:
        raise ValueError("Batch size and negative ratio must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device(args.device)
    dense_path = Path(args.dense_checkpoint)
    dense_model = load_vertex_model(dense_path, device).eval()
    dense_model.requires_grad_(False)
    graph_head, graph_payload = load_proposal_graph_checkpoint(
        args.proposal_graph_checkpoint,
        device,
    )
    if Path(graph_payload["dense_checkpoint"]).resolve() != dense_path.resolve():
        raise ValueError("Proposal graph and dense checkpoints do not match")

    keep_head = None
    expected_keep = graph_payload.get("proposal_keep_checkpoint")
    if expected_keep:
        if not args.proposal_keep_checkpoint:
            raise ValueError("Proposal graph requires a proposal keep checkpoint")
        if Path(expected_keep).resolve() != Path(args.proposal_keep_checkpoint).resolve():
            raise ValueError("Proposal graph and proposal keep checkpoints do not match")
    if args.proposal_keep_checkpoint:
        keep_head, keep_payload = load_proposal_keep_checkpoint(
            args.proposal_keep_checkpoint,
            device,
        )
        if Path(keep_payload["dense_checkpoint"]).resolve() != dense_path.resolve():
            raise ValueError("Proposal keep and dense checkpoints do not match")

    loader = DataLoader(
        ChemVectorIterableDataset(
            expand_shards(args.validation_shards),
            sequence_version="0.4",
            reject_keys=load_reject_keys(args.validation_reject_keys_file),
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=args.device == "cuda",
        collate_fn=ProposalGraphCollator(),
    )
    metrics = evaluate(
        dense_model,
        graph_head,
        loader,
        device,
        node_thresholds=args.node_thresholds,
        edge_thresholds=args.edge_thresholds,
        maximum_batches=args.validation_batches,
        keep_head=keep_head,
        edge_loss_weight=args.edge_loss_weight,
        negative_ratio=args.negative_ratio,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
