"""Freeze and run the finite image-to-SDF oracle factorial experiment."""

import argparse
import fcntl
import json
from pathlib import Path

import torch
from rdkit import RDLogger

from chemvector.external_ocr_r0_contracts import write_new_json
from .protocol import freeze
from .runtime import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen CUDA protocol required")
    # RDKit errors are retained per result; avoid thousands of stderr duplicates.
    RDLogger.DisableLog("rdApp.*")
    with open("/tmp/chempic2sdf-gpu0.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = freeze(output)
        try:
            result = run(output, config)
            print(json.dumps({k: result[k] for k in ("arms", "conditional_transitions", "full_oracle_control_passed")}, indent=2))
        except Exception as error:
            write_new_json(output / "failure.json", {"completed": False, "error": str(error)})
            raise


if __name__ == "__main__":
    main()
