#!/usr/bin/env python3
"""Build frozen two-domain line-only AtomPair training caches."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

from PIL import Image
import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.audit_line_segment_rescue import _proposal_to_target_atoms
from chemvector.line_atom_pair_head import (
    LineAtomPairHeadConfig,
    build_line_atom_pair_features,
)
from chemvector.line_segment_rescue import detect_atom_pair_line_candidates
from chemvector.predicted_incidence import (
    PredictedProposalConfig,
    associate_predicted_proposals,
    decode_predicted_atom_proposals,
    decode_predicted_bond_proposals,
)
from chemvector.train_atom_bond_scene import AtomBondSceneCollator, stack_targets
from chemvector.train_atom_pair_bond_group import (
    TRAIN_DOMAINS,
    VALIDATION_DOMAINS,
)
from chemvector.train_bond_presence_calibration import load_components
from chemvector.train_joint_proposal_objects_full import load_validation_records
from chemvector.train_spatial_incidence import DEFAULT_GRAPH, forward_frozen_bond_head
from chemvector.train_stable_bond_presence import valid_stream


DEFAULT_SOURCE = (
    "outputs/checkpoints/chemvector-v0972-bond-presence-calibration-10k/best.pt"
)
DEFAULT_OUTPUT = "outputs/cache/chemvector-v0985-line-atom-pair"


def _batches(iterator: Iterable[dict], batch_size: int):
    batch = []
    for sample in iterator:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _limited_stream(
    shards: list[str],
    *,
    samples: int,
    config: AtomBondSceneConfig,
    seed: int,
    shuffle_buffer: int,
):
    stream = valid_stream(
        shards,
        config=config,
        seed=seed,
        shuffle_buffer=shuffle_buffer,
    )
    for index, sample in enumerate(stream):
        if index >= samples:
            break
        yield sample


@torch.no_grad()
def build_cache(
    *,
    rows: Iterable[dict],
    samples: int,
    batch_size: int,
    collator: AtomBondSceneCollator,
    vision,
    atom_head,
    bond_head,
    set_head,
    proposal_config: PredictedProposalConfig,
    feature_config: LineAtomPairHeadConfig,
    device: torch.device,
    atom_threshold: float,
    minimum_support_segments: int,
    domain: str,
    split: str,
) -> dict:
    feature_rows = []
    presence_rows = []
    display_rows = []
    chemical_rows = []
    rescue_rows = []
    unknown_rows = []
    sample_rows = []
    sample_keys = []
    stats = Counter()
    accepted = 0
    for samples_batch in _batches(rows, batch_size):
        if accepted >= samples:
            break
        samples_batch = samples_batch[: samples - accepted]
        batch = collator(samples_batch)
        pixels = batch["pixel_values"].to(device)
        targets = stack_targets(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vision_features = vision.forward_stroke_resolution(pixels)
            atom_output = atom_head(vision_features)
            bond_output = forward_frozen_bond_head(
                vision, bond_head, vision_features
            )
            atom_proposals = decode_predicted_atom_proposals(
                atom_output, proposal_config
            )
            bond_proposals = decode_predicted_bond_proposals(
                bond_output, proposal_config
            )
            proposals = associate_predicted_proposals(
                atom_proposals, bond_proposals, targets, proposal_config
            )
            set_output = set_head(
                proposals["atom_features"],
                proposals["atom_geometry"],
                proposals["atom_scores"],
                proposals["atom_mask"],
                proposals["bond_features"],
                proposals["bond_endpoints"],
                proposals["bond_types"],
                proposals["bond_scores"],
                proposals["bond_mask"],
            )

        line_only_candidates = []
        labels = []
        for batch_index, sample in enumerate(samples_batch):
            predicted_pixels = (
                proposals["atom_geometry"][batch_index, :, :2]
                .float()
                .cpu()
                .numpy()
                * 768.0
            )
            proposal_mask = (
                proposals["atom_mask"][batch_index].cpu().numpy().astype(bool)
            )
            selected_mask = (
                proposals["atom_mask"][batch_index]
                & (
                    set_output["atom_presence_logits"][batch_index]
                    >= atom_threshold
                )
            ).cpu().numpy()
            predicted_refs = (
                set_output["incidence_logits"][batch_index].argmax(-1).cpu().tolist()
            )
            existing_pairs = {
                tuple(sorted((int(begin), int(end))))
                for proposal_index, (begin, end) in enumerate(predicted_refs)
                if bool(proposals["bond_mask"][batch_index, proposal_index])
                and int(begin) != int(end)
            }
            candidates = tuple(
                candidate
                for candidate in detect_atom_pair_line_candidates(
                    sample["image"].convert("RGB"),
                    predicted_pixels,
                    atom_mask=selected_mask,
                )
                if len(candidate.support_segment_indices)
                >= minimum_support_segments
                and candidate.atom_pair not in existing_pairs
            )
            line_only_candidates.append(candidates)

            atom_count = int(targets["atom_mask"][batch_index].sum())
            target_pixels = (
                targets["atom_anchors"][batch_index, :atom_count]
                .float()
                .cpu()
                .numpy()
                * 768.0
            )
            proposal_to_target = _proposal_to_target_atoms(
                predicted_pixels,
                proposal_mask,
                target_pixels,
                radius_pixels=proposal_config.atom_match_radius_px,
            )
            bond_count = int(targets["bond_mask"][batch_index].sum())
            references = targets["bond_atom_refs"][batch_index, :bond_count].cpu().tolist()
            expected = {
                tuple(sorted((int(begin), int(end)))): index
                for index, (begin, end) in enumerate(references)
            }
            matched = proposals["bond_target_matched"][
                batch_index, :bond_count
            ].cpu()
            dense_pairs = {
                pair for pair, index in expected.items() if bool(matched[index])
            }
            sample_labels = []
            for candidate in candidates:
                begin, end = candidate.atom_pair
                mapped = (
                    tuple(
                        sorted(
                            (proposal_to_target[begin], proposal_to_target[end])
                        )
                    )
                    if begin in proposal_to_target and end in proposal_to_target
                    else None
                )
                target_index = expected.get(mapped) if mapped is not None else None
                positive = target_index is not None
                sample_labels.append(
                    {
                        "presence": positive,
                        "display": (
                            int(targets["bond_display"][batch_index, target_index])
                            if positive
                            else 0
                        ),
                        "chemical": (
                            int(targets["bond_chemical"][batch_index, target_index])
                            if positive
                            else 0
                        ),
                        "rescue": bool(positive and mapped not in dense_pairs),
                        "unknown": mapped is None,
                    }
                )
            labels.append(sample_labels)

        line_features, local_sample_indices = build_line_atom_pair_features(
            bond_output["object_features"].float(),
            proposals["atom_features"].float(),
            proposals["atom_geometry"][..., :2].float(),
            line_only_candidates,
            config=feature_config,
        )
        flat_labels = [item for sample_labels in labels for item in sample_labels]
        if len(flat_labels) != len(line_features):
            raise RuntimeError("Line candidate feature and label counts differ")
        if len(line_features):
            feature_rows.append(line_features.cpu().half())
            presence_rows.append(
                torch.tensor([item["presence"] for item in flat_labels], dtype=torch.bool)
            )
            display_rows.append(
                torch.tensor([item["display"] for item in flat_labels], dtype=torch.long)
            )
            chemical_rows.append(
                torch.tensor([item["chemical"] for item in flat_labels], dtype=torch.long)
            )
            rescue_rows.append(
                torch.tensor([item["rescue"] for item in flat_labels], dtype=torch.bool)
            )
            unknown_rows.append(
                torch.tensor([item["unknown"] for item in flat_labels], dtype=torch.bool)
            )
            sample_rows.append(local_sample_indices.cpu() + accepted)
        for sample_index, (sample, sample_labels) in enumerate(
            zip(samples_batch, labels)
        ):
            sample_keys.append(str(sample["key"]))
            stats["samples"] += 1
            stats["candidates"] += len(sample_labels)
            stats["samples_with_candidates"] += int(bool(sample_labels))
            stats["positives"] += sum(item["presence"] for item in sample_labels)
            stats["negatives"] += sum(not item["presence"] for item in sample_labels)
            stats["unknown_negatives"] += sum(item["unknown"] for item in sample_labels)
            stats["rescue_positives"] += sum(item["rescue"] for item in sample_labels)
        accepted += len(samples_batch)
        if accepted % 1000 < len(samples_batch):
            print(
                json.dumps(
                    {
                        "domain": domain,
                        "split": split,
                        "samples": accepted,
                        "candidates": stats["candidates"],
                        "positives": stats["positives"],
                        "negatives": stats["negatives"],
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
    if accepted != samples:
        raise ValueError(f"{domain}/{split} yielded {accepted}, expected {samples}")
    empty_features = torch.empty((0, feature_config.input_dim), dtype=torch.float16)
    return {
        "schema_version": "line-atom-pair-cache-0.1",
        "domain": domain,
        "split": split,
        "features": torch.cat(feature_rows) if feature_rows else empty_features,
        "presence_targets": (
            torch.cat(presence_rows) if presence_rows else torch.empty(0, dtype=torch.bool)
        ),
        "display_targets": (
            torch.cat(display_rows) if display_rows else torch.empty(0, dtype=torch.long)
        ),
        "chemical_targets": (
            torch.cat(chemical_rows) if chemical_rows else torch.empty(0, dtype=torch.long)
        ),
        "rescue_targets": (
            torch.cat(rescue_rows) if rescue_rows else torch.empty(0, dtype=torch.bool)
        ),
        "unknown_targets": (
            torch.cat(unknown_rows) if unknown_rows else torch.empty(0, dtype=torch.bool)
        ),
        "sample_indices": (
            torch.cat(sample_rows) if sample_rows else torch.empty(0, dtype=torch.long)
        ),
        "sample_keys": sample_keys,
        "stats": dict(stats),
        "feature_config": asdict(feature_config),
        "atom_threshold": atom_threshold,
        "minimum_support_segments": minimum_support_segments,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE)
    parser.add_argument("--grounded-graph-checkpoint", default=DEFAULT_GRAPH)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="all")
    parser.add_argument("--train-samples-per-domain", type=int, default=8967)
    parser.add_argument("--validation-samples-per-domain", type=int, default=498)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--minimum-support-segments", type=int, default=2)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Line AtomPair cache build requires CUDA")
    if min(
        args.train_samples_per_domain,
        args.validation_samples_per_domain,
        args.batch_size,
        args.minimum_support_segments,
    ) < 1:
        raise ValueError("Sample, batch, and support counts must be positive")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    vision, atom_head, bond_head, set_head, source = load_components(
        args.source_checkpoint, args.grounded_graph_checkpoint, device
    )
    for module in (vision, atom_head, bond_head, set_head):
        module.eval().requires_grad_(False)
    proposal_config = PredictedProposalConfig(**source["proposal_config"])
    scene_config = AtomBondSceneConfig(
        atom_queries=proposal_config.maximum_atoms,
        bond_queries=proposal_config.maximum_bonds,
    )
    collator = AtomBondSceneCollator(scene_config)
    atom_threshold = float(source["training_args"]["atom_decision_threshold"])
    feature_config = LineAtomPairHeadConfig(
        feature_channels=atom_head.config.hidden_channels
    )
    summary = {
        "schema_version": "line-atom-pair-cache-build-0.1",
        "source_checkpoint": args.source_checkpoint,
        "proposal_config": asdict(proposal_config),
        "feature_config": asdict(feature_config),
        "splits": {},
    }
    split_names = ("train", "validation") if args.split == "all" else (args.split,)
    for split in split_names:
        domains = TRAIN_DOMAINS if split == "train" else VALIDATION_DOMAINS
        count = (
            args.train_samples_per_domain
            if split == "train"
            else args.validation_samples_per_domain
        )
        summary["splits"][split] = {}
        for domain_index, (domain, shards) in enumerate(domains.items()):
            shard_list = [shards] if isinstance(shards, str) else list(shards)
            if split == "train":
                rows = _limited_stream(
                    shard_list,
                    samples=count,
                    config=scene_config,
                    seed=args.seed + domain_index,
                    shuffle_buffer=args.shuffle_buffer,
                )
            else:
                rows, _ = load_validation_records(
                    shard_list, per_shard=count, config=scene_config
                )
            cache = build_cache(
                rows=rows,
                samples=count,
                batch_size=args.batch_size,
                collator=collator,
                vision=vision,
                atom_head=atom_head,
                bond_head=bond_head,
                set_head=set_head,
                proposal_config=proposal_config,
                feature_config=feature_config,
                device=device,
                atom_threshold=atom_threshold,
                minimum_support_segments=args.minimum_support_segments,
                domain=domain,
                split=split,
            )
            path = output / f"{split}-{domain}.pt"
            torch.save(cache, path)
            summary["splits"][split][domain] = {
                "path": str(path),
                **cache["stats"],
            }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
