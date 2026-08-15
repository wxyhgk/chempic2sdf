"""Determinism and worker-order replay for the R0 constructor."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .build_observation import build_observation
from .candidate_domain import build_candidate_domain
from .common import canonical_bytes, sha256


ROOT = Path(__file__).resolve().parent
SEQUENTIAL_REPLAYS = 1000
PARALLEL_REPLAYS = 32


def _build_digest(_: int) -> str:
    return sha256(canonical_bytes(build_candidate_domain(build_observation())))


def run_replay() -> dict:
    expected = ROOT.joinpath("r0_candidate_domain.json").read_bytes()
    expected_digest = sha256(expected)
    sequential = [_build_digest(index) for index in range(SEQUENTIAL_REPLAYS)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        parallel = list(executor.map(_build_digest, reversed(range(PARALLEL_REPLAYS))))
    all_digests = sequential + parallel
    if any(digest != expected_digest for digest in all_digests):
        raise RuntimeError("R0_REPLAY_MISMATCH")
    return {
        "artifact_sha256": expected_digest,
        "parallel_replay_count": PARALLEL_REPLAYS,
        "schema_version": "owner-candidate-domain-v0.5-r0-replay",
        "sequential_replay_count": SEQUENTIAL_REPLAYS,
        "status": "passed",
        "training_authorized": False,
        "unique_digest_count": len(set(all_digests)),
        "worker_count": 8,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true",
        help="freeze the canonical result to r0_replay.json",
    )
    args = parser.parse_args()
    result = canonical_bytes(run_replay())
    if args.write:
        ROOT.joinpath("r0_replay.json").write_bytes(result)
    print(result.decode("ascii"), end="")


if __name__ == "__main__":
    main()
