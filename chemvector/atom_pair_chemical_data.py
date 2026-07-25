"""Train-only molecule-disjoint streams for chemical action pilots."""

from __future__ import annotations

import hashlib
import itertools
from typing import Iterator

from chemvector.train_atom_pair_bond_group import TRAIN_DOMAINS
from chemvector.train_stable_bond_presence import valid_stream


PARTITION_MODULUS = 5
DEV_BUCKET = 0


def sample_molecule_id(sample: dict) -> str:
    molecule_id = sample.get("chemmol", {}).get("molecule_id")
    if not molecule_id:
        molecule_id = sample.get("meta", {}).get("molecule_id")
    if not molecule_id:
        raise ValueError("Chemical action sample has no molecule_id")
    return str(molecule_id)


def molecule_partition(molecule_id: str) -> str:
    digest = hashlib.sha256(molecule_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % PARTITION_MODULUS
    return "train_dev" if bucket == DEV_BUCKET else "train_fit"


def partitioned_domain_stream(
    domain: str,
    *,
    partition: str,
    config,
    data_seed: int,
    shuffle_buffer: int,
) -> Iterator[dict]:
    if domain not in TRAIN_DOMAINS:
        raise ValueError(f"Unknown chemical action domain: {domain}")
    if partition not in {"train_fit", "train_dev"}:
        raise ValueError(f"Unknown chemical action partition: {partition}")
    domain_offset = tuple(TRAIN_DOMAINS).index(domain)
    stream = valid_stream(
        TRAIN_DOMAINS[domain],
        config=config,
        seed=data_seed + domain_offset,
        shuffle_buffer=shuffle_buffer,
    )
    for sample in stream:
        if molecule_partition(sample_molecule_id(sample)) == partition:
            yield sample


def load_partition_records(
    domain: str,
    *,
    partition: str,
    count: int,
    config,
    data_seed: int,
    shuffle_buffer: int,
) -> list[dict]:
    if count < 1:
        raise ValueError("Chemical partition count must be positive")
    records = list(
        itertools.islice(
            partitioned_domain_stream(
                domain,
                partition=partition,
                config=config,
                data_seed=data_seed,
                shuffle_buffer=shuffle_buffer,
            ),
            count,
        )
    )
    if len(records) != count:
        raise RuntimeError(
            f"Domain {domain} partition {partition} yielded {len(records)} of {count}"
        )
    return records


def balanced_fit_batches(
    *,
    samples_per_domain: int,
    batch_size: int,
    config,
    data_seed: int,
    shuffle_buffer: int,
):
    if batch_size % len(TRAIN_DOMAINS):
        raise ValueError("Batch size must be divisible by the domain count")
    per_batch = batch_size // len(TRAIN_DOMAINS)
    streams = {
        domain: iter(
            partitioned_domain_stream(
                domain,
                partition="train_fit",
                config=config,
                data_seed=data_seed,
                shuffle_buffer=shuffle_buffer,
            )
        )
        for domain in TRAIN_DOMAINS
    }
    seen = {domain: 0 for domain in TRAIN_DOMAINS}
    while min(seen.values()) < samples_per_domain:
        batch = []
        for domain, stream in streams.items():
            take = min(per_batch, samples_per_domain - seen[domain])
            for _ in range(take):
                try:
                    batch.append(next(stream))
                except StopIteration as error:
                    raise RuntimeError(
                        f"Domain {domain} train_fit exhausted at {seen[domain]}"
                    ) from error
            seen[domain] += take
        if batch:
            yield batch, dict(seen)
