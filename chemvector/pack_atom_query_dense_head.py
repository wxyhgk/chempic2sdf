#!/usr/bin/env python3
"""Pack a trained dense vertex head into an Atom Query checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chemvector.pretrain_vertex import load_vertex_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-checkpoint", required=True)
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    query_path = Path(args.query_checkpoint)
    dense_path = Path(args.dense_checkpoint)
    payload = torch.load(query_path, map_location="cpu", weights_only=False)
    dense_model = load_vertex_model(dense_path, torch.device("cpu")).eval()
    payload["dense_vertex_head"] = {
        name: value.detach().cpu().clone()
        for name, value in dense_model.vertex_head.state_dict().items()
    }
    payload["dense_vertex_head_source"] = str(dense_path)
    payload["training_kind"] = "atom_query_with_shared_dense_head"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(
        f"packed_dense_head_parameters="
        f"{sum(value.numel() for value in payload['dense_vertex_head'].values()):,}"
    )
    print(f"output={output}")


if __name__ == "__main__":
    main()
