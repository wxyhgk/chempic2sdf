"""Compare frozen frontend profiles; never train, tune, promote, or overwrite."""

import argparse
import fcntl
import json
from pathlib import Path

from chemvector.experiments.id_graph.roundtrip import write_json
from .protocol import freeze
from .runtime import predict
from .scoring import score


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for native profile comparison')
    with open('/tmp/chempic2sdf-gpu0.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = freeze(args.output_dir)
        print('Protocol frozen; beginning native profile inference', flush=True)
        try:
            predict(args.output_dir, protocol)
            report = score(args.output_dir, protocol)
        except Exception as error:
            write_json(args.output_dir / 'failure.json', {'completed': False, 'type': type(error).__name__, 'error': str(error)})
            raise
    print(json.dumps({k: report[k] for k in ('denominator', 'arms', 'cold_parity_count')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
