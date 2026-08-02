"""Train-only structural data adapter for the G3 P0 capacity gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from chemvector.bond_object import BOND_TYPE_TO_ID
from chemvector.data import ChemVectorIterableDataset, structure_labels


OVERFIT32_MANIFEST = Path(
    "Data/sdf/chemtraverse_v0_1_capacity/overfit_32.jsonl"
)
OVERFIT32_MANIFEST_SHA256 = (
    "8fbb55b26ed170a64d9f258f702432d5ec626049976f6d9194bcf6751033b8a0"
)
OVERFIT32_SAMPLE_COUNT = 32
OVERFIT32_MOLECULE_COUNT = 8
OVERFIT32_VARIANTS = frozenset({"coordgen", "original", "rdkit", "stress"})
QUANTIZATION_MAX = 1023.0


@dataclass(frozen=True, slots=True)
class G3SampleManifestRecord:
    sample_key: str
    molecule_id: str
    inchikey: str
    shard: str
    variant: str


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_sha256(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def load_g3_sample_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_samples: int,
    expected_molecules: int,
    expected_variants: frozenset[str],
) -> tuple[list[G3SampleManifestRecord], dict]:
    """Load and bind a train-only sample manifest before opening any shard."""

    path = Path(path)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"G3 sample manifest SHA256 differs: {actual_sha256} != {expected_sha256}"
        )
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        required = {"sample_key", "molecule_id", "inchikey", "shard", "variant"}
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(
                f"G3 manifest line {line_number} is missing fields: {missing}"
            )
        records.append(
            G3SampleManifestRecord(
                sample_key=str(payload["sample_key"]),
                molecule_id=str(payload["molecule_id"]),
                inchikey=str(payload["inchikey"]),
                shard=str(payload["shard"]),
                variant=str(payload["variant"]),
            )
        )
    if len(records) != expected_samples:
        raise ValueError(
            f"G3 manifest contains {len(records)} samples, expected {expected_samples}"
        )
    sample_keys = [record.sample_key for record in records]
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("G3 manifest contains duplicate sample keys")
    molecule_ids = [record.molecule_id for record in records]
    if len(set(molecule_ids)) != expected_molecules:
        raise ValueError(
            "G3 manifest molecule count differs: "
            f"{len(set(molecule_ids))} != {expected_molecules}"
        )
    variants = {record.variant for record in records}
    if variants != set(expected_variants):
        raise ValueError(
            f"G3 manifest variants differ: {sorted(variants)} != "
            f"{sorted(expected_variants)}"
        )
    shards = list(dict.fromkeys(record.shard for record in records))
    if not shards or any(not Path(shard).is_file() for shard in shards):
        raise ValueError("G3 manifest references a missing training shard")
    audit = {
        "schema_version": "g3-sample-manifest-audit-0.1",
        "path": str(path.resolve()),
        "sha256": actual_sha256,
        "sample_count": len(records),
        "molecule_count": len(set(molecule_ids)),
        "sample_keys_sha256": _sequence_sha256(sample_keys),
        "molecule_ids_sha256": _sequence_sha256(sorted(set(molecule_ids))),
        "variants": sorted(variants),
        "shards": [str(Path(shard).resolve()) for shard in shards],
    }
    return records, audit


def load_g3_overfit32_manifest(
    path: str | Path = OVERFIT32_MANIFEST,
) -> tuple[list[G3SampleManifestRecord], dict]:
    return load_g3_sample_manifest(
        path,
        expected_sha256=OVERFIT32_MANIFEST_SHA256,
        expected_samples=OVERFIT32_SAMPLE_COUNT,
        expected_molecules=OVERFIT32_MOLECULE_COUNT,
        expected_variants=OVERFIT32_VARIANTS,
    )


def load_g3_manifest_samples(
    manifest: list[G3SampleManifestRecord],
) -> list[dict]:
    """Read exactly the committed sample keys and restore manifest order."""

    expected = {record.sample_key: record for record in manifest}
    shards = list(dict.fromkeys(record.shard for record in manifest))
    loaded: dict[str, dict] = {}
    dataset = ChemVectorIterableDataset(
        shards,
        sequence_version="0.4",
        include_keys=set(expected),
        serialize_target=False,
        load_raster_layers=False,
    )
    for sample in dataset:
        key = str(sample["key"])
        if key in loaded:
            raise ValueError(f"G3 training shard yielded duplicate sample {key}")
        if key not in expected:
            raise ValueError(f"G3 training shard yielded uncommitted sample {key}")
        record = expected[key]
        molecule_id = str(sample["chemmol"]["molecule_id"])
        variant = str(sample["render"]["variant"])
        if molecule_id != record.molecule_id or variant != record.variant:
            raise ValueError(f"G3 manifest identity differs for {key}")
        split = sample.get("meta", {}).get("split")
        source_split = sample.get("meta", {}).get("source_split")
        if split != "train" and source_split != "train":
            raise ValueError(f"G3 P0 may only load train samples, found {key}")
        loaded[key] = sample
    missing = sorted(set(expected) - loaded.keys())
    if missing:
        raise ValueError(f"G3 training shards are missing committed samples: {missing}")
    return [loaded[record.sample_key] for record in manifest]


class G3StructuralCollator:
    """Collate only observable geometry and Bond types for structural G3."""

    TARGET_FIELDS = (
        "atom_anchors",
        "atom_mask",
        "bond_atom_refs",
        "bond_types",
        "bond_display",
        "bond_mask",
    )

    def __init__(
        self,
        *,
        maximum_atoms: int = 128,
        maximum_bonds: int = 128,
        image_size: int = 768,
    ) -> None:
        if min(maximum_atoms, maximum_bonds, image_size) < 1:
            raise ValueError("G3 structural collator dimensions must be positive")
        self.maximum_atoms = maximum_atoms
        self.maximum_bonds = maximum_bonds
        self.image_size = image_size

    def _image_tensor(self, image: Image.Image) -> torch.Tensor:
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return (torch.from_numpy(array).permute(2, 0, 1) - 0.5) / 0.5

    @staticmethod
    def _normalized_atom_anchors(
        sample: dict, labels: dict
    ) -> dict[int, tuple[float, float]]:
        render_positions = {
            int(item["atom_id"]): item
            for item in sample.get("render", {}).get("atoms", [])
        }
        width = int(labels["width"])
        height = int(labels["height"])
        anchors = {}
        for atom in labels["atoms"]:
            atom_id = int(atom["atom_id"])
            rendered = render_positions.get(atom_id)
            if rendered is not None and {
                "quantized_x",
                "quantized_y",
            } <= rendered.keys():
                anchor = (
                    int(rendered["quantized_x"]) / QUANTIZATION_MAX,
                    int(rendered["quantized_y"]) / QUANTIZATION_MAX,
                )
            else:
                center = atom["center"]
                anchor = (
                    float(center[0]) / max(width - 1, 1),
                    float(center[1]) / max(height - 1, 1),
                )
            if not 0.0 <= anchor[0] <= 1.0 or not 0.0 <= anchor[1] <= 1.0:
                raise ValueError(f"G3 Atom anchor is outside the canvas: {atom_id}")
            anchors[atom_id] = anchor
        return anchors

    def __call__(self, samples: list[dict]) -> dict:
        if not samples:
            raise ValueError("G3 structural collator requires at least one sample")
        batch = len(samples)
        atom_anchors = torch.zeros(batch, self.maximum_atoms, 2)
        atom_mask = torch.zeros(batch, self.maximum_atoms, dtype=torch.bool)
        bond_atom_refs = torch.zeros(
            batch, self.maximum_bonds, 2, dtype=torch.long
        )
        bond_types = torch.zeros(batch, self.maximum_bonds, dtype=torch.long)
        bond_display = torch.zeros(batch, self.maximum_bonds, dtype=torch.long)
        bond_mask = torch.zeros(batch, self.maximum_bonds, dtype=torch.bool)
        molecule_ids = []
        variants = []

        for batch_index, sample in enumerate(samples):
            labels = structure_labels(sample)
            atoms = sorted(labels["atoms"], key=lambda item: int(item["atom_id"]))
            bonds = sorted(labels["bonds"], key=lambda item: int(item["bond_id"]))
            if len(atoms) > self.maximum_atoms:
                raise ValueError(
                    f"G3 sample {sample['key']} has {len(atoms)} Atoms, "
                    f"capacity is {self.maximum_atoms}"
                )
            if len(bonds) > self.maximum_bonds:
                raise ValueError(
                    f"G3 sample {sample['key']} has {len(bonds)} Bonds, "
                    f"capacity is {self.maximum_bonds}"
                )
            anchors = self._normalized_atom_anchors(sample, labels)
            if len(anchors) != len(atoms):
                raise ValueError(f"G3 sample {sample['key']} has duplicate Atom IDs")
            slot_by_atom_id = {
                int(atom["atom_id"]): slot for slot, atom in enumerate(atoms)
            }
            for slot, atom in enumerate(atoms):
                atom_anchors[batch_index, slot] = torch.tensor(
                    anchors[int(atom["atom_id"])], dtype=torch.float32
                )
                atom_mask[batch_index, slot] = True
            seen_pairs = set()
            for slot, bond in enumerate(bonds):
                begin_id = int(bond["begin_atom"])
                end_id = int(bond["end_atom"])
                if begin_id not in slot_by_atom_id or end_id not in slot_by_atom_id:
                    raise ValueError(
                        f"G3 sample {sample['key']} Bond references an unknown Atom"
                    )
                pair = tuple(sorted((begin_id, end_id)))
                if begin_id == end_id or pair in seen_pairs:
                    raise ValueError(
                        f"G3 sample {sample['key']} has an invalid duplicate Bond"
                    )
                seen_pairs.add(pair)
                chemical = str(bond["chemical_type"])
                display = str(bond.get("display_type", chemical))
                if chemical not in BOND_TYPE_TO_ID or display not in BOND_TYPE_TO_ID:
                    raise ValueError(
                        f"G3 sample {sample['key']} has an unsupported Bond type"
                    )
                bond_atom_refs[batch_index, slot] = torch.tensor(
                    (slot_by_atom_id[begin_id], slot_by_atom_id[end_id]),
                    dtype=torch.long,
                )
                bond_types[batch_index, slot] = BOND_TYPE_TO_ID[chemical]
                bond_display[batch_index, slot] = BOND_TYPE_TO_ID[display]
                bond_mask[batch_index, slot] = True
            molecule_ids.append(str(sample["chemmol"]["molecule_id"]))
            variants.append(str(sample["render"]["variant"]))

        return {
            "pixel_values": torch.stack(
                [self._image_tensor(sample["image"]) for sample in samples]
            ),
            "atom_anchors": atom_anchors,
            "atom_mask": atom_mask,
            "bond_atom_refs": bond_atom_refs,
            "bond_types": bond_types,
            "bond_display": bond_display,
            "bond_mask": bond_mask,
            "atom_counts": atom_mask.sum(dim=1),
            "bond_counts": bond_mask.sum(dim=1),
            "keys": [str(sample["key"]) for sample in samples],
            "molecule_ids": molecule_ids,
            "variants": variants,
        }


def g3_targets_to_device(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in G3StructuralCollator.TARGET_FIELDS
    }
