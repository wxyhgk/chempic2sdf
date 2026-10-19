"""Command-line entry point for the external OCR R0 candidate audit."""

from __future__ import annotations

import argparse
import json
from typing import Any

import chemvector.external_ocr_r0_contracts as contract
import chemvector.external_ocr_r0_atom_reachability as atom_reachability
import chemvector.external_ocr_r0_crop_bundle as crop_bundle
import chemvector.external_ocr_r0_peak_precision as peak_precision
import chemvector.external_ocr_r0_recognizers as recognizers
import chemvector.external_ocr_r0_report as report
import chemvector.external_ocr_r0_selection as selection
import chemvector.external_ocr_r0_truth as truth


def _arm_paths(args: argparse.Namespace) -> dict[str, str]:
    return {
        "ctc_v0109_greedy": args.ctc_greedy,
        "ctc_v0109_top8": args.ctc_top8,
        "ppocrv5_mobile": args.paddle_mobile,
        "ppocrv5_server": args.paddle_server,
    }


def _run_select(args: argparse.Namespace) -> dict[str, Any]:
    result = selection.write_selection(args.output)
    return {
        "output": args.output,
        "molecules": result["molecules"],
        "samples": result["samples"],
    }


def _run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    return crop_bundle.prepare_predicted_crops(args.selection, args.output_dir)


def _run_recognize(args: argparse.Namespace) -> dict[str, Any]:
    return recognizers.recognize_arm(args.arm, args.dataset_dir, args.output)


def _run_reveal_truth(args: argparse.Namespace) -> dict[str, Any]:
    return truth.reveal_truth(args.dataset_dir, _arm_paths(args), args.output)


def _run_summarize(args: argparse.Namespace) -> dict[str, Any]:
    result = report.summarize(
        args.dataset_dir,
        args.truth_manifest,
        _arm_paths(args),
        args.output,
    )
    return {"decision": result["status"], "output": args.output}


def _run_proposal_ladder(args: argparse.Namespace) -> dict[str, Any]:
    return atom_reachability.run_atom_reachability(
        args.dataset_dir,
        args.truth_manifest,
        args.output_dir,
    )


def _run_peak_precision(args: argparse.Namespace) -> dict[str, Any]:
    return peak_precision.run_peak_precision_audit(
        args.dataset_dir,
        args.truth_manifest,
        args.baseline_ladder_dir,
        args.output_dir,
    )


def _add_arm_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ctc-greedy", required=True)
    parser.add_argument("--ctc-top8", required=True)
    parser.add_argument("--paddle-mobile", required=True)
    parser.add_argument("--paddle-server", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser(
        "select", help="Freeze the molecule-grouped fresh_dev selection"
    )
    select.add_argument("--output", required=True)
    select.set_defaults(function=_run_select)

    prepare = commands.add_parser(
        "prepare", help="Generate frozen predicted-v0.10.8 crops"
    )
    prepare.add_argument("--selection", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.set_defaults(function=_run_prepare)

    recognize = commands.add_parser(
        "recognize", help="Run one frozen recognizer arm"
    )
    recognize.add_argument("--arm", choices=contract.ARM_NAMES, required=True)
    recognize.add_argument("--dataset-dir", required=True)
    recognize.add_argument("--output", required=True)
    recognize.set_defaults(function=_run_recognize)

    reveal = commands.add_parser(
        "reveal-truth", help="Read GT only after all candidate bundles are frozen"
    )
    reveal.add_argument("--dataset-dir", required=True)
    _add_arm_artifact_arguments(reveal)
    reveal.add_argument("--output", required=True)
    reveal.set_defaults(function=_run_reveal_truth)

    report = commands.add_parser(
        "summarize", help="Measure raw candidate unions without replacement"
    )
    report.add_argument("--dataset-dir", required=True)
    report.add_argument("--truth-manifest", required=True)
    _add_arm_artifact_arguments(report)
    report.add_argument("--output", required=True)
    report.set_defaults(function=_run_summarize)

    ladder = commands.add_parser(
        "proposal-ladder", help="Audit frozen AtomObject proposal reachability"
    )
    ladder.add_argument("--dataset-dir", required=True)
    ladder.add_argument("--truth-manifest", required=True)
    ladder.add_argument("--output-dir", required=True)
    ladder.set_defaults(function=_run_proposal_ladder)

    precision = commands.add_parser(
        "peak-precision", help="Audit BF16/FP32 peak and dense-score reachability"
    )
    precision.add_argument("--dataset-dir", required=True)
    precision.add_argument("--truth-manifest", required=True)
    precision.add_argument("--baseline-ladder-dir", required=True)
    precision.add_argument("--output-dir", required=True)
    precision.set_defaults(function=_run_peak_precision)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(args.function(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
