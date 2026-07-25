"""OCR dataset backed by the rendered ChemVision semantic-layer package."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import hashlib
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

from Data.sdf.chemmol_webdataset import ChemMolWebDataset
from chemvector.model import OCR_ROLE_LABELS


ROLE_TO_ID = {label: index for index, label in enumerate(OCR_ROLE_LABELS)}
CPK_TEXT_COLORS = {
    "B": (235, 120, 145),
    "N": (35, 75, 210),
    "O": (215, 35, 35),
    "F": (30, 155, 45),
    "Cl": (30, 155, 45),
    "Br": (145, 55, 35),
    "I": (110, 45, 155),
    "P": (220, 110, 25),
    "S": (190, 150, 20),
    "Se": (175, 85, 30),
}


@dataclass(frozen=True, slots=True)
class ChemVisionLayerOCRConfig:
    maximum_blank_anchors: int = 4
    maximum_visible_anchors: int = 12
    maximum_common_visible_anchors: int = 4
    visible_jitter_copies: int = 0
    visible_jitter_elements: tuple[str, ...] = ("B", "N")
    n_jitter_probability: float = 0.05
    priority_elements: tuple[str, ...] = (
        "B",
        "F",
        "Cl",
        "Br",
        "I",
        "P",
        "Si",
        "Se",
    )
    priority_texts: tuple[str, ...] = ()
    color_probability: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.maximum_blank_anchors,
            self.maximum_visible_anchors,
            self.maximum_common_visible_anchors,
        ) < 0:
            raise ValueError("OCR anchor limits must be nonnegative")
        if self.visible_jitter_copies < 0:
            raise ValueError("Visible jitter copies must be nonnegative")
        if not 0.0 <= self.n_jitter_probability <= 1.0:
            raise ValueError("N jitter probability must be in [0, 1]")
        if not 0.0 <= self.color_probability <= 1.0:
            raise ValueError("OCR color probability must be in [0, 1]")


def colorize_visible_text(image, atoms: list[dict]):
    """Apply element colors inside known text boxes while preserving white background."""

    array = np.asarray(image.convert("RGB")).copy()
    height, width = array.shape[:2]
    for atom in atoms:
        bbox = atom.get("bbox")
        color = CPK_TEXT_COLORS.get(str(atom.get("element", "")))
        if not atom.get("visible_text") or bbox is None or color is None:
            continue
        x0, y0, x1, y1 = bbox
        left = max(0, int(np.floor(x0)) - 1)
        top = max(0, int(np.floor(y0)) - 1)
        right = min(width, int(np.ceil(x1)) + 2)
        bottom = min(height, int(np.ceil(y1)) + 2)
        region = array[top:bottom, left:right]
        darkness = 255 - region.mean(axis=-1)
        alpha = np.clip(darkness / 220.0, 0.0, 1.0)[..., None]
        target = np.asarray(color, dtype=np.float32)
        recolored = 255.0 * (1.0 - alpha) + target * alpha
        region[:] = recolored.round().clip(0, 255).astype(np.uint8)
    from PIL import Image

    return Image.fromarray(array, mode="RGB")


def stable_seed(seed: int, key: str) -> int:
    payload = f"{seed}\0{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def font_bucket(font_size: int) -> str:
    if font_size <= 12:
        return "small"
    if font_size <= 19:
        return "normal"
    if font_size <= 28:
        return "large"
    return "extreme"


def build_layer_ocr_sample(
    sample: dict,
    *,
    seed: int,
    config: ChemVisionLayerOCRConfig,
) -> dict:
    layers = sample.get("layers")
    if layers is None:
        raise ValueError(f"ChemVision sample has no layers.json: {sample['key']}")
    rng = random.Random(stable_seed(seed, sample["key"]))
    atoms = sorted(layers["atoms"], key=lambda atom: atom["atom_id"])
    colored = rng.random() < config.color_probability
    image = colorize_visible_text(sample["image"], atoms) if colored else sample["image"]
    color_suffix = (
        f"_{'color' if colored else 'bw'}" if config.color_probability > 0.0 else ""
    )
    priority = [
        atom
        for atom in atoms
        if atom["visible_text"]
        and (
            atom["element"] in config.priority_elements
            or atom.get("text", "") in config.priority_texts
        )
    ]
    common = [
        atom
        for atom in atoms
        if atom["visible_text"]
        and atom["element"] not in config.priority_elements
        and atom.get("text", "") not in config.priority_texts
    ]
    blank = [atom for atom in atoms if not atom["visible_text"]]
    if len(priority) > config.maximum_visible_anchors:
        priority = rng.sample(priority, config.maximum_visible_anchors)
    common_limit = min(
        config.maximum_common_visible_anchors,
        max(0, config.maximum_visible_anchors - len(priority)),
    )
    if len(common) > common_limit:
        common = rng.sample(common, common_limit)
    if len(blank) > config.maximum_blank_anchors:
        blank = rng.sample(blank, config.maximum_blank_anchors)
    selected = sorted(priority + common + blank, key=lambda atom: atom["atom_id"])
    if not selected:
        selected = [rng.choice(atoms)]
    width = float(layers["width"])
    height = float(layers["height"])
    size = int(layers["render_parameters"]["font_size"])
    bucket = font_bucket(size)
    entries = []
    for atom in selected:
        entries.append(
            {
                "atom_id": atom["atom_id"],
                "point": atom["center"],
                "text": atom["text"],
                "category": f"{bucket}_{'visible' if atom['visible_text'] else 'blank'}{color_suffix}",
            }
        )
        if (
            atom["visible_text"]
            and atom["element"] in config.visible_jitter_elements
            and atom.get("bbox") is not None
            and (
                atom["element"] != "N"
                or rng.random() < config.n_jitter_probability
            )
        ):
            x0, y0, x1, y1 = atom["bbox"]
            for _ in range(config.visible_jitter_copies):
                entries.append(
                    {
                        "atom_id": atom["atom_id"],
                        "point": [rng.uniform(x0, x1), rng.uniform(y0, y1)],
                        "text": atom["text"],
                        "category": f"{bucket}_visible_jitter{color_suffix}",
                    }
                )
    return {
        "key": sample["key"],
        "image": image,
        "anchors": [
            (entry["point"][0] / width, entry["point"][1] / height)
            for entry in entries
        ],
        "texts": [entry["text"] for entry in entries],
        "roles": [
            "atom_text" if entry["text"] else "implicit_c"
            for entry in entries
        ],
        "categories": [entry["category"] for entry in entries],
        "atom_ids": [entry["atom_id"] for entry in entries],
        "font_bucket": bucket,
        "font_family_id": layers["render_parameters"]["font_family_id"],
        "layer_atoms": atoms,
        "layer_width": layers["width"],
        "layer_height": layers["height"],
    }


def labels_for_predicted_points(
    points: list[tuple[float, float]],
    atoms: list[dict],
    *,
    tolerance_pixels: float = 8.0,
    bbox_margin: float = 3.0,
) -> list[str]:
    """Assign actual detector peaks to text instances or implicit-carbon blanks."""

    labels = []
    for x, y in points:
        inside = []
        for atom in atoms:
            bbox = atom.get("bbox")
            if not atom["visible_text"] or bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            if (
                x0 - bbox_margin <= x <= x1 + bbox_margin
                and y0 - bbox_margin <= y <= y1 + bbox_margin
            ):
                distance = float(
                    np.hypot(x - atom["center"][0], y - atom["center"][1])
                )
                inside.append((distance, atom["text"]))
        if inside:
            labels.append(min(inside, key=lambda item: item[0])[1])
            continue
        nearest = min(
            (
                float(np.hypot(x - atom["center"][0], y - atom["center"][1])),
                atom["text"],
            )
            for atom in atoms
        )
        labels.append(nearest[1] if nearest[0] <= tolerance_pixels else "")
    return labels


def assign_predicted_points(
    points: list[tuple[float, float]],
    atoms: list[dict],
    *,
    tolerance_pixels: float = 8.0,
    bbox_margin: float = 3.0,
) -> tuple[list[str], list[str]]:
    """One-to-one match detector peaks to atoms and mark extra peaks as non-atoms."""

    candidates = []
    for point_index, (x, y) in enumerate(points):
        for atom_index, atom in enumerate(atoms):
            center_x, center_y = atom["center"]
            distance = float(np.hypot(x - center_x, y - center_y))
            bbox = atom.get("bbox")
            inside_text = False
            if atom["visible_text"] and bbox is not None:
                x0, y0, x1, y1 = bbox
                inside_text = (
                    x0 - bbox_margin <= x <= x1 + bbox_margin
                    and y0 - bbox_margin <= y <= y1 + bbox_margin
                )
            if inside_text or distance <= tolerance_pixels:
                candidates.append((distance, point_index, atom_index))
    labels = ["" for _ in points]
    roles = ["non_molecule_text" for _ in points]
    assigned_points = set()
    assigned_atoms = set()
    for _, point_index, atom_index in sorted(candidates):
        if point_index in assigned_points or atom_index in assigned_atoms:
            continue
        atom = atoms[atom_index]
        assigned_points.add(point_index)
        assigned_atoms.add(atom_index)
        labels[point_index] = atom["text"]
        roles[point_index] = "atom_text" if atom["visible_text"] else "implicit_c"
    return labels, roles


def collate_layer_ocr(samples: list[dict], codec):
    from chemvector.replacement_ocr import collate_replacement_ocr

    batch = collate_replacement_ocr(samples, codec)
    maximum_anchors = batch["ocr_anchor_mask"].size(1)
    role_targets = torch.full(
        (len(samples), maximum_anchors),
        -1,
        dtype=torch.long,
    )
    for sample_index, sample in enumerate(samples):
        role_targets[sample_index, : len(sample["roles"])] = torch.tensor(
            [ROLE_TO_ID[role] for role in sample["roles"]],
            dtype=torch.long,
        )
    batch["ocr_role_targets"] = role_targets
    batch["layer_atoms"] = [sample["layer_atoms"] for sample in samples]
    batch["layer_widths"] = [sample["layer_width"] for sample in samples]
    batch["layer_heights"] = [sample["layer_height"] for sample in samples]
    return batch


class ChemVisionLayerOCRDataset(IterableDataset):
    def __init__(
        self,
        shards: str | list[str],
        *,
        seed: int = 20260715,
        config: ChemVisionLayerOCRConfig | None = None,
        sample_limit: int | None = None,
        shardshuffle: int | bool = False,
        sample_shuffle_buffer: int = 0,
    ) -> None:
        super().__init__()
        if sample_limit is not None and sample_limit < 1:
            raise ValueError("sample_limit must be positive")
        patterns = [shards] if isinstance(shards, str) else shards
        expanded = []
        for pattern in patterns:
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"No ChemVision OCR shards matched: {pattern}")
            expanded.extend(matches)
        self.shards = expanded
        self.seed = seed
        self.config = config or ChemVisionLayerOCRConfig()
        self.sample_limit = sample_limit
        self.shardshuffle = shardshuffle
        self.sample_shuffle_buffer = sample_shuffle_buffer

    def __iter__(self):
        dataset = ChemMolWebDataset(
            self.shards,
            validate_legacy_hash=False,
            shardshuffle=self.shardshuffle,
            sample_shuffle_buffer=self.sample_shuffle_buffer,
            sequence_version="0.4",
        )
        yielded = 0
        for sample in dataset:
            if self.sample_limit is not None and yielded >= self.sample_limit:
                break
            yield build_layer_ocr_sample(
                sample,
                seed=self.seed,
                config=self.config,
            )
            yielded += 1
