import copy
import hashlib
import json

import numpy as np
import pytest

import chemvector.audit_external_ocr_candidates as cli
import chemvector.external_ocr_r0_contracts as contract
import chemvector.external_ocr_r0_recognizers as recognizers
from chemvector.external_ocr_r0_atom_reachability import (
    RAW_STAGE_NAMES,
    DIAGNOSTIC_STAGE_NAMES,
    SET_LOGIT_THRESHOLDS,
    STAGE_NAMES,
    _target_stage_row,
    legacy_match_target_anchors,
    match_target_anchors,
    summarize_reachability,
)
from chemvector.audit_external_ocr_candidates import build_parser
from chemvector.external_ocr_r0_contracts import (
    ARM_NAMES,
    ARM_SCHEMA_VERSION,
    CROP_SCHEMA_VERSION,
)
from chemvector.external_ocr_r0_report import summarize_records
from chemvector.external_ocr_r0_selection import (
    build_fresh_selection,
    validate_selection,
)
from chemvector.external_ocr_r0_truth import validate_truth_population
from chemvector.external_ocr_r0_truth import (
    rendered_truth_targets,
    validate_reveal_inputs,
    validate_reveal_sample,
)


def crop(key, truth, group):
    return {
        "object_key": key,
        "sample_key": key.split(":")[0],
        "molecule_id": "m1",
        "truth": truth,
        "group": group,
        "crop": {"sha256": f"sha-{key}", "path": f"crops/{key}.png", "width": 192, "height": 48},
    }


def arm_rows(name, crops, values):
    return [
        {
            "arm": name,
            "object_key": item["object_key"],
            "crop_sha256": item["crop"]["sha256"],
            "candidates": [
                {"rank": rank, "text": text, "score": 0.0, "score_kind": "test"}
                for rank, text in enumerate(values[index])
            ],
        }
        for index, item in enumerate(crops)
    ]


def test_raw_candidate_union_reports_incremental_rescue() -> None:
    crops = [
        crop("s1:p000", "N", "simple_element"),
        crop("s1:p001", "CH3", "modifier"),
        crop("s2:p000", "Ph", "open_string"),
        crop("s2:p001", None, None),
    ]
    values = {
        "ctc_v0109_greedy": [["N"], ["C"], ["P"], [""]],
        "ctc_v0109_top8": [["N"], ["C", "CH"], ["P"], [""]],
        "ppocrv5_mobile": [["N"], ["CH3"], ["P"], ["-"]],
        "ppocrv5_server": [["N"], ["CH3"], ["Ph"], [""]],
    }
    arms = {name: arm_rows(name, crops, values[name]) for name in ARM_NAMES}
    truth = [
        {
            "object_key": item["object_key"],
            "truth": item["truth"],
            "group": item["group"],
        }
        for item in crops
        if item["truth"] is not None
    ]
    for item in crops:
        item.pop("truth")
        item.pop("group")

    report, joined = summarize_records(crops, truth, arms)

    paired = report["paired"]["vs_incumbent_greedy_plus_top8"]
    assert paired["union_only"] == 2
    assert paired["baseline_only"] == 0
    assert report["union"]["external_only"]["exact"]["count"] == 3
    assert report["arms"]["ppocrv5_mobile"]["negative_nonempty"]["count"] == 1
    assert joined[1]["combined_union"] == ["C", "CH", "CH3"]
    assert not any(
        key.startswith("arm:") or key in {"incumbent", "external", "combined"}
        for row in joined
        for key in row
    )
    assert report["status"] == "INCONCLUSIVE"


def test_duplicate_candidates_are_counted_once() -> None:
    crops = [crop("s1:p000", "N", "simple_element")]
    truth = [{"object_key": "s1:p000", "truth": "N", "group": "simple_element"}]
    crops[0].pop("truth")
    crops[0].pop("group")
    values = {name: [["N", "N"]] for name in ARM_NAMES}
    arms = {name: arm_rows(name, crops, values[name]) for name in ARM_NAMES}
    report, joined = summarize_records(crops, truth, arms)
    assert joined[0]["external_union"] == ["N"]
    assert report["union"]["ctc_plus_external"]["exact"]["count"] == 1


def test_natural_population_does_not_require_absent_open_strings() -> None:
    crops = [
        crop(f"s{index:03d}:p000", "N", "simple_element")
        for index in range(contract.GROUP_MINIMUM)
    ] + [
        crop(f"m{index:03d}:p000", "NH", "modifier")
        for index in range(contract.GROUP_MINIMUM)
    ]
    truth = [
        {
            "object_key": item["object_key"],
            "truth": item.pop("truth"),
            "group": item.pop("group"),
        }
        for item in crops
    ]
    values = {name: [[row["truth"]] for row in truth] for name in ARM_NAMES}
    arms = {name: arm_rows(name, crops, values[name]) for name in ARM_NAMES}

    report, _ = summarize_records(crops, truth, arms)

    assert report["population"]["groups"]["open_string"]["objects"] == 0
    assert report["gates"]["adequate_groups"] is True
    assert report["status"] == "NO-GO"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("He", "simple_element"),
        ("Na", "simple_element"),
        ("NH2+", "modifier"),
        ("15N", "modifier"),
        ("Ph", "open_string"),
    ],
)
def test_label_groups_follow_atom_text_parser(text, expected) -> None:
    assert contract.label_group(text) == expected


def test_greedy_rescue_is_not_credited_to_external_ocr() -> None:
    crops = [crop("s1:p000", "N", "simple_element")]
    truth = [{"object_key": "s1:p000", "truth": "N", "group": "simple_element"}]
    crops[0].pop("truth")
    crops[0].pop("group")
    values = {
        "ctc_v0109_greedy": [["N"]],
        "ctc_v0109_top8": [["O"]],
        "ppocrv5_mobile": [["N"]],
        "ppocrv5_server": [["O"]],
    }
    arms = {name: arm_rows(name, crops, values[name]) for name in ARM_NAMES}
    report, _ = summarize_records(crops, truth, arms)
    paired = report["paired"]["vs_incumbent_greedy_plus_top8"]
    assert paired["both_correct"] == 1
    assert paired["union_only"] == 0
    assert report["gates"]["incremental_correct_count"] == 0


def test_crop_hash_mismatch_fails_closed() -> None:
    crops = [crop("s1:p000", "N", "simple_element")]
    crops[0].pop("truth")
    crops[0].pop("group")
    values = {name: [["N"]] for name in ARM_NAMES}
    arms = {name: arm_rows(name, crops, values[name]) for name in ARM_NAMES}
    arms["ppocrv5_server"][0]["crop_sha256"] = "wrong"
    with pytest.raises(ValueError, match="crop SHA256"):
        summarize_records(crops, [], arms)


def test_recognizer_arm_choices_are_closed() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "recognize",
                "--arm",
                "router",
                "--dataset-dir",
                "x",
                "--output",
                "y",
            ]
        )


def test_proposal_ladder_cli_is_available() -> None:
    args = build_parser().parse_args(
        [
            "proposal-ladder",
            "--dataset-dir",
            "dataset",
            "--truth-manifest",
            "truth.json",
            "--output-dir",
            "ladder",
        ]
    )
    assert args.command == "proposal-ladder"


def test_cli_does_not_reexport_domain_implementations() -> None:
    assert not any(
        hasattr(cli, name)
        for name in (
            "prepare_predicted_crops",
            "recognize_arm",
            "reveal_truth",
            "summarize",
            "summarize_records",
            "write_selection",
        )
    )


def test_reachability_matching_is_one_to_one_and_radius_bounded() -> None:
    matches = match_target_anchors(
        np.asarray([[0.1, 0.1], [0.9, 0.9], [0.11, 0.1]]),
        np.asarray([[102.3, 102.3], [920.7, 920.7], [500.0, 500.0]]),
        image_size=768.0,
        target_quantization_max=1023.0,
        match_radius_px=6.0,
    )
    assert set(matches) == {0, 1}
    assert len({candidate for candidate, _ in matches.values()}) == 2


def test_reachability_matching_maximizes_feasible_cardinality() -> None:
    candidates = np.asarray([[0.0, 0.0], [0.0, 1.0]])
    targets = np.asarray([[0.0, 0.0], [10.0, 0.0]])
    kwargs = {
        "image_size": 1.0,
        "target_quantization_max": 1.0,
        "match_radius_px": 10.0,
    }
    matches = match_target_anchors(
        candidates,
        targets,
        **kwargs,
    )
    legacy_matches = legacy_match_target_anchors(
        candidates,
        targets,
        **kwargs,
    )

    assert set(matches) == {0, 1}
    assert len({candidate for candidate, _ in matches.values()}) == 2
    assert all(distance <= 10.0 for _, distance in matches.values())
    assert len(legacy_matches) == 1


def test_raw_target_stage_never_indexes_set_head_outputs() -> None:
    torch = pytest.importorskip("torch")
    proposals = {
        "atom_scores": torch.ones((1, 384)),
        "atom_indices": torch.arange(384).reshape(1, 384),
    }

    row = _target_stage_row(
        match={"proposal_index": 300, "distance_px": 1.0},
        proposals=proposals,
        batch_index=0,
        presence_logits=None,
        decoded_mask=None,
        set_evaluated=False,
    )

    assert row["proposal_rank"] == 300
    assert row["spatial_index"] == 300
    assert row["presence_logit"] is None
    assert row["set_evaluated"] is False


def test_reachability_summary_routes_recoverable_misses_to_selector() -> None:
    stages = {
        name: {"matched": name != "deployed"} for name in STAGE_NAMES
    }
    stages.update(
        {
            f"threshold:{threshold:g}": {"matched": threshold < 0.25}
            for threshold in SET_LOGIT_THRESHOLDS
        }
    )
    target_rows = [
        {
            "variant": "stress",
            "role": "text",
            "stages": copy.deepcopy(stages),
        }
        for _ in range(10)
    ]
    target_rows.extend(
        {
            "variant": "original",
            "role": "implicit",
            "stages": copy.deepcopy(stages),
        }
        for _ in range(10)
    )
    sample_stage = {
        name: {
            "selected": 20,
            "matched": 20 if name != "deployed" else 0,
            "matched_text": 10 if name != "deployed" else 0,
            "matched_implicit": 10 if name != "deployed" else 0,
            "extra_proposals": 0 if name != "deployed" else 20,
            "count_exact": True,
            "object_exact": name != "deployed",
        }
        for name in STAGE_NAMES
    }
    sample_stage.update(
        {
            f"threshold:{threshold:g}": {
                "selected": 20,
                "matched": 20 if threshold < 0.25 else 0,
                "matched_text": 10 if threshold < 0.25 else 0,
                "matched_implicit": 10 if threshold < 0.25 else 0,
                "extra_proposals": 0 if threshold < 0.25 else 20,
                "count_exact": True,
                "object_exact": threshold < 0.25,
            }
            for threshold in SET_LOGIT_THRESHOLDS
        }
    )

    report = summarize_reachability(
        target_rows,
        [{"variant": "stress", "stages": sample_stage}],
        deployed_threshold=0.25,
    )

    assert report["deployment_text_misses"]["raw_recovery_rate"] == 1.0
    assert report["gates"]["stress_raw_text_recall"] == 1.0
    assert report["gates"]["recommendation"] == "selector_threshold_or_budget"
    assert "-0.25" in report["threshold_frontier"]


@pytest.mark.parametrize(
    ("top384_matched", "all_peaks_matched", "dense_matched", "recommendation"),
    [
        (True, True, True, "raw_capacity_or_peak_ranking"),
        (False, True, True, "deeper_raw_capacity_or_peak_ranking"),
        (False, False, True, "peak_precision_and_dense_score_counterfactual"),
        (False, False, False, "anchor_geometry_generator"),
    ],
)
def test_reachability_summary_separates_capacity_from_generator_failure(
    top384_matched, all_peaks_matched, dense_matched, recommendation
) -> None:
    target_rows = []
    for index in range(10):
        raw_matches = {
            name: index < (10 if name == "raw_top384" and top384_matched else 7)
            for name in RAW_STAGE_NAMES
        }
        stages = {
            "dense_anchors": {"matched": index < (10 if dense_matched else 7)},
            "all_local_peaks": {
                "matched": index < (10 if all_peaks_matched else 7)
            },
            **{name: {"matched": matched} for name, matched in raw_matches.items()},
            "decoded": {"matched": index < 7},
            "set_zero": {"matched": False},
            "deployed": {"matched": False},
        }
        stages.update(
            {
                f"threshold:{threshold:g}": {"matched": False}
                for threshold in SET_LOGIT_THRESHOLDS
            }
        )
        target_rows.append(
            {"variant": "stress", "role": "text", "stages": stages}
        )
    sample_stages = {
        name: {
            "selected": 10,
            "matched": sum(row["stages"][name]["matched"] for row in target_rows),
            "matched_text": sum(
                row["stages"][name]["matched"] for row in target_rows
            ),
            "matched_implicit": 0,
            "extra_proposals": 0,
            "count_exact": True,
            "object_exact": False,
        }
        for name in (
            *STAGE_NAMES,
            *(f"threshold:{value:g}" for value in SET_LOGIT_THRESHOLDS),
        )
    }

    report = summarize_reachability(
        target_rows,
        [{"variant": "stress", "stages": sample_stages}],
        deployed_threshold=-0.25,
    )

    assert report["gates"]["raw_capacities"]["raw_top128"]["passed"] is False
    assert (
        report["gates"]["raw_capacities"]["raw_top384"]["passed"]
        is top384_matched
    )
    assert report["gates"]["recommendation"] == recommendation


def test_union_does_not_mutate_input_records() -> None:
    crops = [crop("s1:p000", "N", "simple_element")]
    crops[0].pop("truth")
    crops[0].pop("group")
    original = copy.deepcopy(crops)
    values = {name: [["N"]] for name in ARM_NAMES}
    arms = {name: arm_rows(name, crops, values[name]) for name in ARM_NAMES}
    summarize_records(crops, [], arms)
    assert crops == original


def test_selection_is_deterministic_and_mutation_is_rejected() -> None:
    first = build_fresh_selection()
    second = build_fresh_selection()
    assert contract.canonical_json_bytes(first) == contract.canonical_json_bytes(second)
    changed = copy.deepcopy(first)
    changed["sample_keys"] = changed["sample_keys"][4:]
    changed["samples"] -= 4
    with pytest.raises(ValueError, match="2,000 samples"):
        validate_selection(changed)


def test_self_signed_forged_model_identity_is_rejected(tmp_path, monkeypatch) -> None:
    records_path = tmp_path / "greedy.jsonl"
    record = {
        "schema_version": ARM_SCHEMA_VERSION,
        "arm": "ctc_v0109_greedy",
        "object_key": "s1:p000",
        "crop_sha256": "crop",
        "model_id": "forged",
        "latency_ms": 1.0,
        "error": None,
        "candidates": [{"rank": 0, "text": "N", "score": 0.0, "score_kind": "test"}],
    }
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    record_sha = hashlib.sha256(records_path.read_bytes()).hexdigest()
    monkeypatch.setattr(contract, "FROZEN_CROP_MANIFEST_SHA256", "crop-manifest")
    metadata = {
        "schema_version": ARM_SCHEMA_VERSION,
        "arm": "ctc_v0109_greedy",
        "phase": "candidate_coverage_only",
        "crop_manifest": {"path": str(tmp_path / "manifest.json"), "sha256": "crop-manifest"},
        "records": {"path": str(records_path), "sha256": record_sha, "rows": 1},
        "model_files": [
            {
                "path": str(contract.CTC_CHECKPOINT),
                "sha256": contract.V0109_CTC_SHA256,
            }
        ],
        "implementation_contract_sha256": recognizers.recognizer_contract_sha256(),
        "training_performed": False,
        "routing_performed": False,
        "replacement_performed": False,
    }
    records_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    monkeypatch.setattr(
        contract,
        "FROZEN_ARM_BUNDLE_SHA256",
        {"ctc_v0109_greedy": contract.arm_bundle_sha256(records_path)},
    )
    with pytest.raises(ValueError, match="model identity"):
        recognizers.validate_arm_artifact(
            "ctc_v0109_greedy", records_path, tmp_path / "manifest.json"
        )


def test_truth_group_tampering_is_rejected() -> None:
    crops = [{"object_key": "s1:p000"}]
    truth = [
        {
            "schema_version": CROP_SCHEMA_VERSION,
            "gt_key": "s1:t000",
            "object_key": "s1:p000",
            "truth": "CH3",
            "group": "simple_element",
            "selected_as_text": True,
        }
    ]
    manifest = {"gt_text_objects": 1, "matched_gt_text_objects": 1}
    with pytest.raises(ValueError, match="group differs"):
        validate_truth_population(truth, manifest, crops)


def test_truth_reveal_revalidates_frozen_inputs(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        contract,
        "validate_frozen_file",
        lambda path, sha256, name: calls.append((path, sha256, name)),
    )
    monkeypatch.setattr(
        "chemvector.external_ocr_r0_truth.validate_fresh_shards",
        lambda shards: calls.append(tuple(shards)),
    )

    validate_reveal_inputs(["fresh-000.tar"])

    assert calls == [
        (
            contract.SOURCE_CHECKPOINT,
            contract.SOURCE_SHA256,
            "source proposal checkpoint",
        ),
        ("fresh-000.tar",),
    ]


def test_truth_reveal_revalidates_sample_identity() -> None:
    expected = {"molecule_id": "m1", "png_sha256": "png-sha"}
    sample = {
        "meta": {"molecule_id": "m1", "sha256": {"png": "png-sha"}},
        "layers": {"split": "fresh_dev"},
    }
    validate_reveal_sample(sample, expected)

    changed = copy.deepcopy(sample)
    changed["meta"]["sha256"]["png"] = "wrong"
    with pytest.raises(ValueError, match="PNG commitment"):
        validate_reveal_sample(changed, expected)


def test_truth_targets_allow_overlapping_render_coordinates() -> None:
    sample = {
        "render": {
            "atoms": [
                {"atom_id": 1, "quantized_x": 100, "quantized_y": 200},
                {"atom_id": 0, "quantized_x": 100, "quantized_y": 200},
            ]
        },
        "layers": {
            "atoms": [
                {"atom_id": 0, "visible_text": True, "text": "N"},
                {"atom_id": 1, "visible_text": True, "text": "O"},
            ]
        },
    }

    targets = rendered_truth_targets(sample)

    assert [row["source_atom_id"] for row in targets] == [0, 1]
    assert [row["truth"] for row in targets] == ["N", "O"]
    assert targets[0]["anchor"] == targets[1]["anchor"] == [100, 200]
