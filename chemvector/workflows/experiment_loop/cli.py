"""Command-line interface for experiment campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chemvector.experiment_loop.runner import CampaignRunner
from chemvector.experiment_loop.schema import load_campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    run = subparsers.add_parser("run")
    status = subparsers.add_parser("status")
    for current in (validate, run, status):
        current.add_argument("campaign")
        current.add_argument("--project-root", default=".")
    run.add_argument("--allow-draft", action="store_true")
    run.add_argument("--gpu-lock", default="/tmp/chempic2sdf-gpu0.lock")
    args = parser.parse_args()

    if args.command == "validate":
        spec = load_campaign(args.campaign)
        print(
            json.dumps(
                {
                    "campaign_id": spec.campaign_id,
                    "status": spec.status,
                    "trials": len(spec.trials),
                },
                ensure_ascii=False,
            )
        )
        return
    if args.command == "status":
        spec = load_campaign(args.campaign)
        path = (
            Path(args.project_root).resolve()
            / "outputs"
            / "experiments"
            / spec.campaign_id
            / "state.json"
        )
        if not path.is_file():
            print(json.dumps({"campaign_id": spec.campaign_id, "status": "not_started"}))
        else:
            print(path.read_text(encoding="utf-8"), end="")
        return
    runner = CampaignRunner(
        args.campaign,
        project_root=args.project_root,
        allow_draft=args.allow_draft,
        gpu_lock_path=args.gpu_lock,
    )
    print(json.dumps(runner.run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
