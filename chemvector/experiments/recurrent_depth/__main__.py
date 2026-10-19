"""Run a non-promoting candidate selection experiment in a new directory."""

import argparse
import fcntl
import json
from pathlib import Path

import torch

from chemvector.external_ocr_r0_contracts import write_new_json
from .data import build_cache
from .protocol import freeze
from .run import fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this protocol")
    torch.set_num_threads(4)
    # Do not interrupt another experiment using the GPU.
    with open("/tmp/chempic2sdf-gpu0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = freeze(output)
        try:
            cache = build_cache(output, protocol)
            write_new_json(output / "cache_receipt.json", cache)
            report = fit(output, protocol, cache)
            print(json.dumps(report["decision"], indent=2))
        except Exception as error:
            write_new_json(output / "failure.json", {"completed": False, "error": str(error)})
            raise


if __name__ == "__main__":
    main()
