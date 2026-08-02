#!/usr/bin/env python3
"""Train and evaluate byte CTC OCR on Open Text v0.1 crops."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import tarfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from chemvector.ctc_ocr import (
    PretrainedPixelCTCConfig,
    PretrainedPixelCTCOCR,
    ctc_loss,
    decode_ctc,
    encode_text,
)
from chemvector.pretrain_ctc_ocr import is_modifier_label
from chemvector.paths import PROJECT_ROOT


DEFAULT_DATA_ROOT = PROJECT_ROOT / "Data/ocr/open_text_v0_1"
DEFAULT_INITIAL_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/checkpoints/chemvector-v088-pretrained-bbox-ctc-10k/best.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/checkpoints/chemvector-v0811-open-text-balanced-20k"
)
OUTPUT_SIZE = (192, 48)
SOURCE_BUCKET_TARGETS = {
    "synthetic_random": 0.30,
    "synthetic_curated": 0.20,
    "molecule_gt": 0.10,
    "molecule_sampled": 0.10,
    "molecule_real": 0.30,
}
CONFUSION_LABELS = {"N", "NH", "NH2", "N+", "B", "Br", "P", "Ph"}
COMMON_SINGLE_LABELS = {"B", "C", "F", "I", "N", "O", "P", "S"}


@dataclass(frozen=True, slots=True)
class OpenTextEntry:
    key: str
    shard: Path
    text: str
    split: str
    source_kind: str
    bbox_protocol: str
    string_group: str
    color_mode: str
    font_family_id: str
    counterfactual_pair_id: str | None = None
    counterfactual_pair_group: str | None = None


def letterbox_crop(
    image: Image.Image,
    output_size: tuple[int, int] = OUTPUT_SIZE,
) -> Image.Image:
    image = image.convert("RGB")
    output_width, output_height = output_size
    scale = min(
        output_width / max(image.width, 1),
        output_height / max(image.height, 1),
    )
    resized = image.resize(
        (
            max(1, min(output_width, round(image.width * scale))),
            max(1, min(output_height, round(image.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", output_size, "white")
    canvas.paste(
        resized,
        (
            (output_width - resized.width) // 2,
            (output_height - resized.height) // 2,
        ),
    )
    return canvas


def source_bucket(entry: OpenTextEntry) -> str:
    if entry.source_kind == "synthetic_compositional":
        return (
            "synthetic_random"
            if entry.string_group == "random_composition"
            else "synthetic_curated"
        )
    if entry.bbox_protocol == "real_predicted":
        return "molecule_real"
    if entry.bbox_protocol == "gt":
        return "molecule_gt"
    return "molecule_sampled"


def random_structure_key(text: str) -> str:
    classes = []
    if any(character.isupper() for character in text):
        classes.append("upper")
    if any(character.islower() for character in text):
        classes.append("lower")
    if any(character.isdigit() for character in text):
        classes.append("digit")
    if any(not character.isalnum() and not character.isspace() for character in text):
        classes.append("symbol")
    if any(character.isspace() for character in text):
        classes.append("space")
    return f"len{len(text)}:{'+'.join(classes) or 'other'}"


def balance_key(entry: OpenTextEntry) -> str:
    if entry.string_group == "random_composition":
        return random_structure_key(entry.text)
    return entry.text


class OpenTextDataset(Dataset):
    def __init__(self, root: str | Path, *, split: str) -> None:
        self.root = Path(root)
        self.split = split
        self.entries: list[OpenTextEntry] = []
        self._archives: dict[Path, tarfile.TarFile] = {}
        pair_index_path = self.root / "counterfactual_pairs.json"
        pair_index = {}
        if pair_index_path.is_file():
            pair_payload = json.loads(pair_index_path.read_text(encoding="utf-8"))
            if pair_payload.get("status") != "passed":
                raise ValueError("Open Text counterfactual pair index is not passed")
            pair_index = {
                record["sample_key"]: record
                for record in pair_payload["records"]
            }
        for group in ("synthetic_compositional", "molecule_bbox"):
            group_root = self.root / group
            manifest = group_root / "samples.csv"
            if not manifest.is_file():
                raise FileNotFoundError(f"Missing Open Text manifest: {manifest}")
            with manifest.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row["split"] != split:
                        continue
                    shard = group_root / row["shard"]
                    if not shard.is_file():
                        raise FileNotFoundError(f"Missing Open Text shard: {shard}")
                    pair = pair_index.get(row["sample_key"]) if group == "synthetic_compositional" else None
                    if pair is not None and (
                        pair["split"] != row["split"] or pair["text"] != row["text"]
                    ):
                        raise ValueError(
                            f"Counterfactual index mismatch for {row['sample_key']}"
                        )
                    self.entries.append(
                        OpenTextEntry(
                            key=row["sample_key"],
                            shard=shard,
                            text=row["text"],
                            split=row["split"],
                            source_kind=row["source_kind"],
                            bbox_protocol=row["bbox_protocol"],
                            string_group=row["string_group"],
                            color_mode=row["color_mode"],
                            font_family_id=row["font_family_id"],
                            counterfactual_pair_id=(
                                str(pair["pair_id"]) if pair is not None else None
                            ),
                            counterfactual_pair_group=(
                                str(pair["pair_group"]) if pair is not None else None
                            ),
                        )
                    )
        if not self.entries:
            raise ValueError(f"Open Text split {split!r} is empty")

    def __len__(self) -> int:
        return len(self.entries)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archives"] = {}
        return state

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def _archive(self, path: Path) -> tarfile.TarFile:
        archive = self._archives.get(path)
        if archive is None:
            archive = tarfile.open(path)
            self._archives[path] = archive
        return archive

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        extracted = self._archive(entry.shard).extractfile(f"{entry.key}.png")
        if extracted is None:
            raise RuntimeError(f"Missing PNG {entry.shard}:{entry.key}")
        with Image.open(extracted) as opened:
            image = letterbox_crop(opened)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return {
            "pixels": (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5,
            "text": entry.text,
            "key": f"{entry.source_kind}:{entry.key}",
            "source_kind": entry.source_kind,
            "bbox_protocol": entry.bbox_protocol,
            "string_group": entry.string_group,
            "color_mode": entry.color_mode,
            "font_family_id": entry.font_family_id,
            "source_bucket": source_bucket(entry),
            "counterfactual_pair_id": entry.counterfactual_pair_id,
            "counterfactual_pair_group": entry.counterfactual_pair_group,
        }


def training_weights(
    entries: list[OpenTextEntry],
) -> tuple[torch.Tensor, dict]:
    bucket_counts = Counter(source_bucket(entry) for entry in entries)
    missing = set(SOURCE_BUCKET_TARGETS) - set(bucket_counts)
    if missing:
        raise ValueError(f"Training split is missing source buckets: {sorted(missing)}")
    key_counts = Counter(
        (source_bucket(entry), balance_key(entry)) for entry in entries
    )
    raw = [
        1.0 / math.sqrt(key_counts[(source_bucket(entry), balance_key(entry))])
        for entry in entries
    ]
    raw_mass = defaultdict(float)
    for entry, value in zip(entries, raw, strict=True):
        raw_mass[source_bucket(entry)] += value
    weights = torch.tensor(
        [
            value
            * SOURCE_BUCKET_TARGETS[source_bucket(entry)]
            / raw_mass[source_bucket(entry)]
            for entry, value in zip(entries, raw, strict=True)
        ],
        dtype=torch.double,
    )
    probability = defaultdict(float)
    total = float(weights.sum())
    for entry, value in zip(entries, weights.tolist(), strict=True):
        probability[source_bucket(entry)] += value / total
    audit = {
        "samples": len(entries),
        "source_counts": dict(sorted(bucket_counts.items())),
        "balance_key_counts": len(key_counts),
        "target_probability": SOURCE_BUCKET_TARGETS,
        "observed_probability": {
            key: round(value, 8) for key, value in sorted(probability.items())
        },
    }
    return weights, audit


def collate_open_text(samples: list[dict], maximum_bytes: int) -> dict:
    encoded = [encode_text(sample["text"], maximum_bytes) for sample in samples]
    return {
        "pixels": torch.stack([sample["pixels"] for sample in samples]),
        "targets": torch.cat(encoded),
        "target_lengths": torch.tensor(
            [len(item) for item in encoded], dtype=torch.long
        ),
        "texts": [sample["text"] for sample in samples],
        "keys": [sample["key"] for sample in samples],
        "source_kinds": [sample["source_kind"] for sample in samples],
        "bbox_protocols": [sample["bbox_protocol"] for sample in samples],
        "string_groups": [sample["string_group"] for sample in samples],
        "color_modes": [sample["color_mode"] for sample in samples],
        "font_family_ids": [sample["font_family_id"] for sample in samples],
        "source_buckets": [sample["source_bucket"] for sample in samples],
        "counterfactual_pair_ids": [
            sample["counterfactual_pair_id"] for sample in samples
        ],
        "counterfactual_pair_groups": [
            sample["counterfactual_pair_group"] for sample in samples
        ],
    }


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + int(left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def new_metric_totals() -> dict:
    return {
        "samples": 0,
        "exact": 0,
        "edit_distance": 0,
        "reference_characters": 0,
        "simple_samples": 0,
        "false_modifier": 0,
    }


def update_metric_totals(total: dict, expected: str, predicted: str) -> None:
    modifier = is_modifier_label(expected)
    total["samples"] += 1
    total["exact"] += int(expected == predicted)
    total["edit_distance"] += edit_distance(expected, predicted)
    total["reference_characters"] += len(expected)
    if not modifier:
        total["simple_samples"] += 1
        total["false_modifier"] += int(is_modifier_label(predicted))


def summarize_metric_totals(total: dict) -> dict:
    return {
        **total,
        "exact_rate": total["exact"] / max(total["samples"], 1),
        "edit_similarity": max(
            0.0,
            1.0
            - total["edit_distance"] / max(total["reference_characters"], 1),
        ),
        "false_modifier_rate": total["false_modifier"]
        / max(total["simple_samples"], 1),
    }


def custom_segments(
    *,
    source_kind: str,
    bbox_protocol: str,
    string_group: str,
    text: str,
) -> list[str]:
    result = []
    if string_group == "random_composition":
        result.append("open_random")
    else:
        result.append("curated")
    if source_kind == "molecule_bbox":
        result.append("molecule_all")
        if bbox_protocol == "real_predicted":
            result.append("real_predicted")
        if is_modifier_label(text):
            result.append("molecule_modifier")
        else:
            result.append("molecule_simple")
        if text in COMMON_SINGLE_LABELS:
            result.append("molecule_common_single")
    return result


@torch.inference_mode()
def evaluate_open_text(
    model: PretrainedPixelCTCOCR,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    overall = new_metric_totals()
    segment_totals: dict[str, dict[str, dict]] = {
        category: defaultdict(new_metric_totals)
        for category in (
            "source_kind",
            "bbox_protocol",
            "string_group",
            "color_mode",
            "font_family_id",
            "length",
            "source_bucket",
            "custom",
        )
    }
    label_totals: dict[str, dict] = defaultdict(new_metric_totals)
    label_groups: dict[str, str] = {}
    confusion: dict[str, Counter] = defaultdict(Counter)
    failures = []
    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(pixels)
        predictions = [decode_ctc(row) for row in logits.argmax(-1).cpu()]
        for index, (expected, predicted) in enumerate(
            zip(batch["texts"], predictions, strict=True)
        ):
            source_kind = batch["source_kinds"][index]
            bbox_protocol = batch["bbox_protocols"][index]
            string_group = batch["string_groups"][index]
            values = {
                "source_kind": source_kind,
                "bbox_protocol": bbox_protocol,
                "string_group": string_group,
                "color_mode": batch["color_modes"][index],
                "font_family_id": batch["font_family_ids"][index],
                "length": str(len(expected)) if len(expected) <= 4 else "5+",
                "source_bucket": batch["source_buckets"][index],
            }
            update_metric_totals(overall, expected, predicted)
            for category, value in values.items():
                update_metric_totals(
                    segment_totals[category][value], expected, predicted
                )
            for name in custom_segments(
                source_kind=source_kind,
                bbox_protocol=bbox_protocol,
                string_group=string_group,
                text=expected,
            ):
                update_metric_totals(
                    segment_totals["custom"][name], expected, predicted
                )
            update_metric_totals(label_totals[expected], expected, predicted)
            label_groups[expected] = string_group
            if expected in CONFUSION_LABELS:
                confusion[expected][predicted] += 1
            if expected != predicted and len(failures) < 100:
                failures.append(
                    {
                        "key": batch["keys"][index],
                        "expected": expected,
                        "predicted": predicted,
                        **values,
                    }
                )

    labels = {
        label: summarize_metric_totals(total)
        for label, total in sorted(label_totals.items())
        if label_groups[label] != "random_composition" or total["samples"] >= 2
    }
    curated_rates = [
        total["exact"] / total["samples"]
        for label, total in label_totals.items()
        if label_groups[label] != "random_composition"
    ]
    supported_rates = [
        total["exact"] / total["samples"]
        for total in label_totals.values()
        if total["samples"] >= 20
    ]
    result = summarize_metric_totals(overall)
    result.update(
        {
            "curated_macro_exact_rate": sum(curated_rates)
            / max(len(curated_rates), 1),
            "supported_macro_exact_rate": sum(supported_rates)
            / max(len(supported_rates), 1),
            "curated_label_count": len(curated_rates),
            "supported_label_count": len(supported_rates),
            "segments": {
                category: {
                    name: summarize_metric_totals(total)
                    for name, total in sorted(values.items())
                }
                for category, values in segment_totals.items()
            },
            "labels": labels,
            "confusion": {
                expected: dict(values.most_common(20))
                for expected, values in sorted(confusion.items())
            },
            "failure_examples": failures,
        }
    )
    custom = result["segments"]["custom"]
    result["modifier_exact_rate"] = custom.get(
        "molecule_modifier", {"exact_rate": 0.0}
    )["exact_rate"]
    result["simple_exact_rate"] = custom.get(
        "molecule_simple", {"exact_rate": 0.0}
    )["exact_rate"]
    return result


def load_initial_model(
    checkpoint: Path,
    *,
    arm: str,
) -> tuple[PretrainedPixelCTCOCR, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config_values = dict(payload["config"])
    config_values["freeze_pixel_encoder"] = arm == "frozen"
    model = PretrainedPixelCTCOCR(PretrainedPixelCTCConfig(**config_values))
    model.load_state_dict(payload["pretrained_ctc_ocr"], strict=True)
    model.pixel_encoder_loaded = True
    if arm == "partial":
        model.pixel_encoder.requires_grad_(False)
        for index in (2, 4):
            model.pixel_encoder[index].requires_grad_(True)
    elif arm == "full":
        model.pixel_encoder.requires_grad_(True)
    elif arm != "frozen":
        raise ValueError(f"Unknown Open Text OCR arm: {arm}")
    return model, payload


def optimizer_for_arm(
    model: PretrainedPixelCTCOCR,
    *,
    learning_rate: float,
    pixel_learning_rate: float,
) -> torch.optim.Optimizer:
    pixel_parameters = [
        parameter
        for parameter in model.pixel_encoder.parameters()
        if parameter.requires_grad
    ]
    pixel_ids = {id(parameter) for parameter in pixel_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in pixel_ids
    ]
    groups = [{"params": head_parameters, "lr": learning_rate}]
    if pixel_parameters:
        groups.append({"params": pixel_parameters, "lr": pixel_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=0.01)


def metric_score(metrics: dict) -> float:
    custom = metrics["segments"]["custom"]
    real = custom.get("real_predicted", {"exact_rate": 0.0})["exact_rate"]
    modifier = custom.get("molecule_modifier", {"exact_rate": 0.0})[
        "exact_rate"
    ]
    common_single = custom.get(
        "molecule_common_single", {"exact_rate": 0.0}
    )["exact_rate"]
    false_modifier = custom.get(
        "molecule_simple", {"false_modifier_rate": 1.0}
    )["false_modifier_rate"]
    random_similarity = custom.get("open_random", {"edit_similarity": 0.0})[
        "edit_similarity"
    ]
    return (
        0.30 * real
        + 0.20 * modifier
        + 0.15 * common_single
        + 0.15 * metrics["curated_macro_exact_rate"]
        + 0.10 * random_similarity
        + 0.10 * (1.0 - false_modifier)
    )


def make_validation_loader(
    dataset: OpenTextDataset,
    *,
    batch_size: int,
    workers: int,
    maximum_bytes: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        collate_fn=lambda samples: collate_open_text(samples, maximum_bytes),
    )


def train_arm(
    arm: str,
    args,
    train_dataset: OpenTextDataset,
    validation_loader: DataLoader,
    weights: torch.Tensor,
    device: torch.device,
) -> dict:
    model, initial_payload = load_initial_model(args.initial_checkpoint, arm=arm)
    model.to(device)
    optimizer = optimizer_for_arm(
        model,
        learning_rate=args.learning_rate,
        pixel_learning_rate=args.pixel_learning_rate,
    )
    output = args.output_dir / arm
    output.mkdir(parents=True, exist_ok=True)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    samples_per_epoch = math.ceil(args.train_crops / args.epochs)
    history = []
    best_score = -math.inf
    samples_seen = 0
    for epoch in range(1, args.epochs + 1):
        remaining = args.train_crops - samples_seen
        current_samples = min(samples_per_epoch, remaining)
        if current_samples <= 0:
            break
        generator = torch.Generator().manual_seed(args.seed + epoch)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=current_samples,
            replacement=True,
            generator=generator,
        )
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            collate_fn=lambda samples: collate_open_text(
                samples, model.config.maximum_bytes
            ),
        )
        model.train()
        loss_sum = 0.0
        batches = 0
        for batch in loader:
            pixels = batch["pixels"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(pixels)
            loss = ctc_loss(
                logits.float(),
                batch["targets"].to(device),
                batch["target_lengths"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                1.0,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach())
            batches += 1
        samples_seen += current_samples
        metrics = evaluate_open_text(model, validation_loader, device)
        metrics.update(
            {
                "epoch": epoch,
                "samples_seen": samples_seen,
                "train_loss": loss_sum / max(batches, 1),
                "selection_score": metric_score(metrics),
            }
        )
        history.append(metrics)
        print(
            f"arm={arm} epoch={epoch} samples={samples_seen} "
            + json.dumps(
                {
                    key: metrics[key]
                    for key in (
                        "exact_rate",
                        "curated_macro_exact_rate",
                        "supported_macro_exact_rate",
                        "modifier_exact_rate",
                        "simple_exact_rate",
                        "false_modifier_rate",
                        "train_loss",
                        "selection_score",
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        payload = {
            "pretrained_ctc_ocr": model.state_dict(),
            "config": asdict(model.config),
            "arm": arm,
            "epoch": epoch,
            "samples_seen": samples_seen,
            "validation": metrics,
            "initial_checkpoint": str(args.initial_checkpoint),
            "initial_validation": initial_payload.get("validation"),
            "training_args": serializable_args(args),
        }
        torch.save(payload, output / "last.pt")
        if metrics["selection_score"] > best_score:
            best_score = metrics["selection_score"]
            torch.save(payload, output / "best.pt")
            write_json(output / "best-validation.json", metrics)
    write_json(output / "history.json", {"arm": arm, "history": history})
    return {
        "arm": arm,
        "trainable_parameters": trainable,
        "best": json.loads((output / "best-validation.json").read_text()),
        "checkpoint": str(output / "best.pt"),
    }


def evaluate_checkpoint(
    checkpoint: Path,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PretrainedPixelCTCOCR(PretrainedPixelCTCConfig(**payload["config"]))
    model.load_state_dict(payload["pretrained_ctc_ocr"], strict=True)
    model.pixel_encoder_loaded = True
    model.to(device)
    return evaluate_open_text(model, loader, device)


def gate_results(validation: dict, internal: dict) -> dict:
    validation_custom = validation["segments"]["custom"]
    internal_custom = internal["segments"]["custom"]
    checks = {
        "real_predicted_exact_at_least_80": validation_custom.get(
            "real_predicted", {"exact_rate": 0.0}
        )["exact_rate"]
        >= 0.80,
        "molecule_modifier_exact_at_least_70": validation_custom.get(
            "molecule_modifier", {"exact_rate": 0.0}
        )["exact_rate"]
        >= 0.70,
        "molecule_common_single_at_least_95": validation_custom.get(
            "molecule_common_single", {"exact_rate": 0.0}
        )["exact_rate"]
        >= 0.95,
        "molecule_simple_false_modifier_at_most_2": validation_custom.get(
            "molecule_simple", {"false_modifier_rate": 1.0}
        )["false_modifier_rate"]
        <= 0.02,
        "internal_real_predicted_within_8pp": internal_custom.get(
            "real_predicted", {"exact_rate": 0.0}
        )["exact_rate"]
        >= validation_custom.get("real_predicted", {"exact_rate": 0.0})[
            "exact_rate"
        ]
        - 0.08,
        "internal_molecule_all_within_8pp": internal_custom.get(
            "molecule_all", {"exact_rate": 0.0}
        )["exact_rate"]
        >= validation_custom.get("molecule_all", {"exact_rate": 0.0})[
            "exact_rate"
        ]
        - 0.08,
        "internal_supported_macro_within_12pp": internal[
            "supported_macro_exact_rate"
        ]
        >= validation["supported_macro_exact_rate"] - 0.12,
    }
    return {
        "status": "passed" if all(checks.values()) else "pilot_only",
        "checks": checks,
    }


def serializable_args(args) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--initial-checkpoint", type=Path, default=DEFAULT_INITIAL_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("frozen", "partial", "full"),
        default=("frozen", "partial"),
    )
    parser.add_argument("--train-crops", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--pixel-learning-rate", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if min(
        args.train_crops,
        args.epochs,
        args.batch_size,
        args.evaluation_batch_size,
    ) < 1:
        parser.error("crop, epoch and batch counts must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = OpenTextDataset(args.data_root, split="train")
    validation_dataset = OpenTextDataset(args.data_root, split="validation")
    internal_dataset = OpenTextDataset(args.data_root, split="internal_test")
    weights, sampler_audit = training_weights(train_dataset.entries)
    initial_payload = torch.load(
        args.initial_checkpoint, map_location="cpu", weights_only=False
    )
    maximum_bytes = int(initial_payload["config"]["maximum_bytes"])
    validation_loader = make_validation_loader(
        validation_dataset,
        batch_size=args.evaluation_batch_size,
        workers=args.workers,
        maximum_bytes=maximum_bytes,
    )
    internal_loader = make_validation_loader(
        internal_dataset,
        batch_size=args.evaluation_batch_size,
        workers=args.workers,
        maximum_bytes=maximum_bytes,
    )
    write_json(
        args.output_dir / "run.json",
        {
            "protocol": "v0.8.11-open-text-controlled-pilot",
            "args": serializable_args(args),
            "dataset_sizes": {
                "train": len(train_dataset),
                "validation": len(validation_dataset),
                "internal_test": len(internal_dataset),
            },
            "sampler": sampler_audit,
        },
    )

    baseline_model, _ = load_initial_model(args.initial_checkpoint, arm="frozen")
    baseline_model.to(device)
    baseline_validation = evaluate_open_text(
        baseline_model, validation_loader, device
    )
    baseline_internal = evaluate_open_text(baseline_model, internal_loader, device)
    write_json(args.output_dir / "baseline-validation.json", baseline_validation)
    write_json(args.output_dir / "baseline-internal-test.json", baseline_internal)
    print(
        "baseline="
        + json.dumps(
            {
                "validation_exact": baseline_validation["exact_rate"],
                "validation_curated_macro": baseline_validation[
                    "curated_macro_exact_rate"
                ],
                "validation_real_predicted": baseline_validation["segments"][
                    "custom"
                ]["real_predicted"]["exact_rate"],
                "internal_exact": baseline_internal["exact_rate"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    arms = []
    for arm in args.arms:
        arms.append(
            train_arm(
                arm,
                args,
                train_dataset,
                validation_loader,
                weights,
                device,
            )
        )
    best_arm = max(arms, key=lambda item: item["best"]["selection_score"])
    best_checkpoint = Path(best_arm["checkpoint"])
    best_internal = evaluate_checkpoint(best_checkpoint, internal_loader, device)
    write_json(
        args.output_dir / best_arm["arm"] / "best-internal-test.json",
        best_internal,
    )
    gates = gate_results(best_arm["best"], best_internal)
    summary = {
        "protocol": "v0.8.11-open-text-controlled-pilot",
        "status": gates["status"],
        "baseline": {
            "validation": baseline_validation,
            "internal_test": baseline_internal,
        },
        "arms": arms,
        "selected_arm": best_arm["arm"],
        "selected_checkpoint": str(best_checkpoint),
        "selected_validation": best_arm["best"],
        "selected_internal_test": best_internal,
        "gates": gates,
        "sampler": sampler_audit,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(
        "summary="
        + json.dumps(
            {
                "status": summary["status"],
                "selected_arm": summary["selected_arm"],
                "validation_exact": summary["selected_validation"]["exact_rate"],
                "validation_real_predicted": summary["selected_validation"][
                    "segments"
                ]["custom"]["real_predicted"]["exact_rate"],
                "internal_exact": summary["selected_internal_test"]["exact_rate"],
                "gates": gates["checks"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
