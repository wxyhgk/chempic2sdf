import copy

import pytest

from chemvector.audit_external_ocr_candidates import build_parser
from chemvector.external_ocr_r0_peak_precision import (
    BF16_DENSE_STAGES,
    FP32_DENSE_ANCHOR_STAGE,
    FP32_DENSE_STAGES,
    PEAK_STAGES,
    STAGE_NAMES,
    _dense_proposals,
    _ranked_dense_proposals,
    stable_dense_order,
    summarize_peak_precision,
)


def test_peak_precision_cli_is_available() -> None:
    args = build_parser().parse_args(
        [
            "peak-precision",
            "--dataset-dir",
            "dataset",
            "--truth-manifest",
            "truth.json",
            "--baseline-ladder-dir",
            "ladder",
            "--output-dir",
            "precision",
        ]
    )

    assert args.command == "peak-precision"


def test_stable_dense_order_uses_spatial_identity_for_ties() -> None:
    torch = pytest.importorskip("torch")
    scores = torch.tensor([[0.5, 0.9, 0.9, 0.1, 0.5]])

    order = stable_dense_order(scores)

    assert order.tolist() == [[1, 2, 0, 4, 3]]


def test_ranked_dense_proposals_are_stable_prefixes() -> None:
    torch = pytest.importorskip("torch")
    dense = {
        "atom_scores": torch.tensor([[0.2, 0.9, 0.9, 0.5]]),
        "atom_indices": torch.arange(4).reshape(1, 4),
        "atom_geometry": torch.arange(8, dtype=torch.float32).reshape(1, 4, 2),
    }

    ranked = _ranked_dense_proposals(dense, 4)

    assert ranked["atom_indices"].tolist() == [[1, 2, 3, 0]]
    assert ranked["atom_geometry"][0, :2].equal(
        dense["atom_geometry"][0, [1, 2]]
    )


def test_float_peak_ops_do_not_invent_precision_lost_in_bf16_logits() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]],
        dtype=torch.bfloat16,
    )
    atom_output = {
        "heatmap_logits": logits,
        "anchors": torch.zeros((1, 2, 3, 3)),
    }

    bf16_dense, bf16_peaks = _dense_proposals(
        atom_output, force_float_peak_ops=False
    )
    fp32_ops_dense, fp32_ops_peaks = _dense_proposals(
        atom_output, force_float_peak_ops=True
    )

    assert bf16_peaks.equal(fp32_ops_peaks)
    assert bf16_dense["atom_scores"].dtype == torch.bfloat16
    assert fp32_ops_dense["atom_scores"].dtype == torch.float32


@pytest.mark.parametrize(
    ("a1_recovers", "a2_recovers", "dense_recovers", "recommendation"),
    [
        (True, True, True, "bf16_peak_ops_quantization"),
        (False, True, True, "bf16_forward_quantization"),
        (False, False, True, "local_peak_operator_candidate"),
        (False, False, False, "heatmap_score_anchor_geometry_decoupling"),
    ],
)
def test_peak_precision_summary_routes_counterfactual(
    a1_recovers, a2_recovers, dense_recovers, recommendation
) -> None:
    target_rows = []
    for index in range(10):
        stages = {stage: {"matched": False} for stage in STAGE_NAMES}
        stages[FP32_DENSE_ANCHOR_STAGE]["matched"] = True
        stages["a1_bf16_fp32_peak_ops"]["matched"] = a1_recovers
        stages["a2_fp32_local_peaks"]["matched"] = a2_recovers
        stages[BF16_DENSE_STAGES[-1]]["matched"] = dense_recovers
        stages[FP32_DENSE_STAGES[-1]]["matched"] = dense_recovers
        target_rows.append(
            {
                "variant": "stress",
                "role": "text",
                "truth": "O",
                "baseline_deployed_miss": True,
                "baseline_representation_miss": True,
                "stages": stages,
            }
        )
    for index in range(10):
        stages = {stage: {"matched": True} for stage in STAGE_NAMES}
        target_rows.append(
            {
                "variant": "original",
                "role": "implicit",
                "truth": None,
                "baseline_deployed_miss": False,
                "baseline_representation_miss": False,
                "stages": stages,
            }
        )
    sample_stage = {
        stage: {
            "selected": 20,
            "matched": sum(row["stages"][stage]["matched"] for row in target_rows),
            "matched_text": sum(
                row["role"] == "text" and row["stages"][stage]["matched"]
                for row in target_rows
            ),
            "matched_implicit": sum(
                row["role"] == "implicit" and row["stages"][stage]["matched"]
                for row in target_rows
            ),
            "extra_proposals": 0,
        }
        for stage in STAGE_NAMES
    }
    sample_rows = [
        {
            "variant": "stress",
            "heatmap_unique_logits": {"a0_bf16": 10, "a2_fp32": 100},
            "stages": copy.deepcopy(sample_stage),
        }
    ]

    report = summarize_peak_precision(target_rows, sample_rows)

    assert report["gates"]["recommendation"] == recommendation
    assert report["baseline_failures"]["peak_inaccessible_text"] == 10
