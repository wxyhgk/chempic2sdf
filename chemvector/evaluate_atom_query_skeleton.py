#!/usr/bin/env python3
"""Evaluate Atom Query plus dense snapping with the existing relation head."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from Data.sdf.chemsvg_sequence_v4 import VertexRecordV4, parse_chemsvg_v4
from chemvector.atom_query import (
    AtomQueryConfig,
    AtomQueryDetector,
    apply_atom_query_dense_head,
    apply_atom_query_vision_updates,
    decode_atom_queries,
    decode_atom_queries_by_set_count,
    load_atom_query_state,
    snap_queries_to_dense_candidates,
)
from chemvector.data import ChemVectorIterableDataset
from chemvector.evaluate import load_model
from chemvector.evaluate_skeleton import (
    compare_skeleton,
    parse_thresholds,
    skeleton_collate,
    summarize,
)
from chemvector.grounded_selector import (
    GroundedQuerySelector,
    GroundedSelectorConfig,
    decode_grounded_queries,
    fuse_dense_candidates_with_queries,
)
from chemvector.parallel_v6 import score_vertex_pairs_parallel
from chemvector.pretrain_atom_query import query_features
from chemvector.pretrain_vertex import load_vertex_model
from chemvector.train import expand_shards
from chemvector.vertex import decode_vertices


def vertex_records(detections: list[dict[str, float | bool]]) -> list[VertexRecordV4]:
    records = []
    seen = set()
    for item in detections:
        coordinate = (
            max(0, min(1023, round(float(item["x"]) * 1023))),
            max(0, min(1023, round(float(item["y"]) * 1023))),
        )
        if coordinate in seen:
            continue
        seen.add(coordinate)
        records.append(
            VertexRecordV4(
                x=coordinate[0],
                y=coordinate[1],
                element="C",
                formal_charge=0,
                isotope=0,
                aromatic=bool(item["aromatic"]),
                explicit_h_count=0,
                no_implicit=False,
                radical_electrons=0,
                canonical_chiral_tag="CHI_UNSPECIFIED",
            )
        )
    return records


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation-checkpoint", required=True)
    parser.add_argument("--query-checkpoint", required=True)
    parser.add_argument("--dense-checkpoint")
    parser.add_argument("--grounded-selector-checkpoint")
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--feature-level", choices=("stroke", "high"), default="stroke")
    parser.add_argument("--query-threshold", type=float, default=0.5)
    parser.add_argument(
        "--query-decode-mode",
        choices=("threshold", "set_count"),
        default="set_count",
    )
    parser.add_argument("--dense-threshold", type=float, default=0.2)
    parser.add_argument("--snap-distance-pixels", type=float, default=64.0)
    parser.add_argument("--vertex-tolerance-pixels", type=float, default=6.0)
    parser.add_argument(
        "--edge-thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.30,0.40,0.50,0.60,0.70"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(args.device)
    relation_model, tokenizer = load_model(Path(args.relation_checkpoint), device)
    relation_model.eval()
    payload = torch.load(args.query_checkpoint, map_location="cpu", weights_only=False)
    query_config = AtomQueryConfig(**payload["atom_query_config"])
    query_detector = AtomQueryDetector(query_config).to(device)
    load_atom_query_state(query_detector, payload["atom_query"])
    query_detector.eval()
    apply_atom_query_vision_updates(relation_model, payload)
    dense_model = None
    if args.dense_checkpoint:
        dense_model = load_vertex_model(Path(args.dense_checkpoint), device).eval()
    else:
        apply_atom_query_dense_head(relation_model, payload)
    grounded_selector = None
    grounded_payload = None
    if args.grounded_selector_checkpoint:
        grounded_payload = torch.load(
            args.grounded_selector_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if Path(grounded_payload["query_checkpoint"]).resolve() != Path(
            args.query_checkpoint
        ).resolve():
            raise ValueError("Grounded selector query checkpoint does not match")
        grounded_config = GroundedSelectorConfig(
            **grounded_payload["grounded_selector_config"]
        )
        grounded_selector = GroundedQuerySelector(grounded_config).to(device)
        grounded_selector.load_state_dict(grounded_payload["grounded_selector"])
        grounded_selector.eval()
    sequence_version = tokenizer.sequence_schema_version.removeprefix(
        "chemmolseq-"
    ).removeprefix("chemsvg-")
    loader = DataLoader(
        ChemVectorIterableDataset(
            expand_shards(args.shards),
            sequence_version=sequence_version,
        ),
        batch_size=1,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=lambda samples: skeleton_collate(samples, 768),
    )
    records = []
    for batch in loader:
        if len(records) >= args.samples:
            break
        pixels = batch["pixel_values"].to(device)
        expected = parse_chemsvg_v4(batch["target_sequences"][0])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            features = query_features(
                relation_model,
                pixels,
                feature_level=args.feature_level,
            )
            query_output = query_detector(features)
            relation_features = (
                features
                if args.feature_level == "stroke"
                else relation_model.vision_encoder.forward_stroke_resolution(pixels)
            )
            if dense_model is None:
                dense_output = relation_model.vertex_head(relation_features)
            else:
                dense_features = dense_model.vision_encoder.forward_stroke_resolution(
                    pixels
                )
                dense_output = dense_model.vertex_head(dense_features)
        candidates = decode_vertices(
            dense_output,
            threshold=args.dense_threshold,
            maximum_vertices=128,
        )
        if grounded_selector is not None:
            grounded = fuse_dense_candidates_with_queries(
                query_output,
                candidates,
                image_size=768,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                grounded_logits = grounded_selector(
                    grounded["query_features"],
                    grounded["scalar_features"],
                    grounded["grounded_centers"],
                    grounded["base_logits"],
                    grounded["candidate_mask"],
                )
            snapped = decode_grounded_queries(
                grounded_logits,
                grounded["grounded_centers"],
                grounded["aromatic_logits"],
                candidate_mask=grounded["candidate_mask"],
            )[0]
        elif args.query_decode_mode == "set_count":
            queries = decode_atom_queries_by_set_count(query_output)[0]
            snapped = snap_queries_to_dense_candidates(
                queries,
                candidates[0],
                image_size=768,
                maximum_distance_pixels=args.snap_distance_pixels,
            )
        else:
            queries = decode_atom_queries(
                query_output,
                threshold=args.query_threshold,
            )[0]
            snapped = snap_queries_to_dense_candidates(
                queries,
                candidates[0],
                image_size=768,
                maximum_distance_pixels=args.snap_distance_pixels,
            )
        vertices = vertex_records(snapped)
        pairs, probabilities = score_vertex_pairs_parallel(
            relation_model,
            relation_features,
            vertices,
            pixels.shape[-2:],
        )
        records.append(
            {
                "key": batch["keys"][0],
                "modes": {
                    "query_dense_snap": {
                        f"{threshold:.4f}": asdict(
                            compare_skeleton(
                                expected,
                                vertices,
                                pairs,
                                probabilities,
                                edge_threshold=threshold,
                                tolerance_pixels=args.vertex_tolerance_pixels,
                                image_size=768,
                            )
                        )
                        for threshold in args.edge_thresholds
                    }
                },
            }
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(records, ("query_dense_snap",), args.edge_thresholds)
    summary.update(
        {
            "relation_checkpoint": args.relation_checkpoint,
            "query_checkpoint": args.query_checkpoint,
            "dense_checkpoint": args.dense_checkpoint,
            "query_decode_mode": args.query_decode_mode,
            "grounded_selector_checkpoint": args.grounded_selector_checkpoint,
            "query_config": asdict(query_config),
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
