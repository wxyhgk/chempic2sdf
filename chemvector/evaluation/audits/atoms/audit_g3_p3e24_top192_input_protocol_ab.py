#!/usr/bin/env python3
"""Compare global and prefix-compatible top192 inputs without training."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import torch

from chemvector.components.atoms.objects.g3_deepsets_count import (
    g3_deepsets_count_features,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e23_top192_cycle_recovery as p3e23,
)
from chemvector.training.atoms import audit_g3_p3e7_base_expert_router as p3e7
from chemvector.training.atoms import audit_g3_p3e17_p3e16_count_router as p3e17
from chemvector.training.atoms import train_g3_p3e14_style_count_scale as p3e14
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e24-top192-input-protocol-ab"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
P3E23_AUDIT = p3e23.OUTPUT / "audit.json"
P3E23_AUDIT_SHA256 = "c562d6082775dd0c55d963ca24c6f56bee904aae1bf7a0d5271dda0653741b08"
SPLITS = ("train", "dev")


def prefix_normalized_scores(raw_scores: torch.Tensor) -> torch.Tensor:
    if raw_scores.shape != (p3e23.CANDIDATES,):
        raise ValueError("P3-E24 raw score width differs")
    logits = torch.logit(raw_scores.float().clamp(1e-6, 1.0 - 1e-6))
    reference = logits[:128]
    return (logits - reference.mean()) / reference.std(
        unbiased=False
    ).clamp_min(1e-4)


def prefix_scale_ranks() -> torch.Tensor:
    return torch.arange(p3e23.CANDIDATES).float() / 127.0


def build_prefix_records(
    top256: dict, old: dict[str, list[dict]]
) -> tuple[dict[str, list[dict]], dict]:
    result = {}
    checks = {
        "candidate_ids": True,
        "candidate_features": True,
        "candidate_anchors": True,
        "raw_scores": True,
        "normalized_scores": True,
        "scale_ranks": True,
    }
    digest = hashlib.sha256()
    ranks = prefix_scale_ranks()
    for split in SPLITS:
        old_by_key = {str(record["sample_key"]): record for record in old[split]}
        current = []
        for key in sorted(top256["records"][split]):
            record = p3e23.format_top192_record(key, top256["records"][split][key])
            record["normalized_scores"] = prefix_normalized_scores(record["raw_scores"])
            record["scale_ranks"] = ranks.clone()
            reference = old_by_key[key]
            for name in checks:
                checks[name] &= torch.equal(record[name][:128], reference[name])
            for name in (
                "candidate_ids",
                "candidate_features",
                "candidate_anchors",
                "raw_scores",
                "normalized_scores",
                "scale_ranks",
            ):
                digest.update(record[name].numpy().tobytes())
            current.append(record)
        result[split] = current
    return result, {
        "top128_prefix_equal": checks,
        "all_top128_prefix_equal": all(checks.values()),
        "candidate_tensor_sha256": digest.hexdigest(),
        "normalization": "all192 transformed by mean/std of first128 logits",
        "scale_rank": "rank/127 without clipping",
    }


def cached_outputs(records: list[dict]) -> tuple[dict, dict]:
    base = {}
    expert = {}
    for record in records:
        key = str(record["sample_key"])
        base[key] = {
            "existence_logits": record["base_existence_logits"].float(),
            "pointer_logits": record["base_pointer_logits"].float(),
        }
        expert[key] = {
            "existence_logits": record["expert_existence_logits"].float(),
            "pointer_logits": record["expert_pointer_logits"].float(),
        }
    return base, expert


@torch.no_grad()
def build_query_evidence(
    cache: dict[str, list[dict]], base_model, expert_model, device: torch.device
) -> tuple[dict[str, list[dict]], dict]:
    outputs = {}
    repeat_checks = []
    for split, records in cache.items():
        merged = []
        for start in range(0, len(records), p3e23.BATCH_SIZE):
            current = records[start : start + p3e23.BATCH_SIZE]
            batch = p3e23.collate(current, device)
            with _autocast_context(device, "bf16"):
                base = base_model(batch)
                expert = expert_model(batch)
                query, candidate, global_features = g3_deepsets_count_features(
                    base,
                    expert,
                    batch["candidate_features"],
                    batch["raw_scores"],
                    batch["normalized_scores"],
                    batch["scale_ranks"],
                )
            p3e23.validate_model_output(base, len(current))
            p3e23.validate_model_output(expert, len(current))
            if start == 0:
                with _autocast_context(device, "bf16"):
                    repeated_base = base_model(batch)
                    repeated_expert = expert_model(batch)
                repeat_checks.append(
                    p3e23.tensor_digest([
                        base["existence_logits"], base["pointer_logits"],
                        expert["existence_logits"], expert["pointer_logits"],
                    ])
                    == p3e23.tensor_digest([
                        repeated_base["existence_logits"], repeated_base["pointer_logits"],
                        repeated_expert["existence_logits"], repeated_expert["pointer_logits"],
                    ])
                )
            for index, record in enumerate(current):
                merged.append({
                    **record,
                    "query_features": query[index].cpu().to(torch.float16),
                    "count_candidate_features": candidate[index].cpu().to(torch.float16),
                    "global_features": global_features[index].cpu().to(torch.float16),
                    "label": int(record["full_atom_mask"].sum()),
                    "baseline_count": int(base["count_logits"][index].argmax()),
                    "base_existence_logits": base["existence_logits"][index].cpu().to(torch.float16),
                    "base_pointer_logits": base["pointer_logits"][index].cpu().to(torch.float16),
                    "expert_existence_logits": expert["existence_logits"][index].cpu().to(torch.float16),
                    "expert_pointer_logits": expert["pointer_logits"][index].cpu().to(torch.float16),
                })
        outputs[split] = merged
    return outputs, {
        "strict_load": True,
        "pointer_shape": [p3e23.QUERIES, p3e23.CANDIDATES],
        "repeat_runs_equal": all(repeat_checks),
    }


@torch.no_grad()
def attach_count_logits(records: list[dict], count_model, device: torch.device) -> None:
    for start in range(0, len(records), p3e23.BATCH_SIZE):
        current = records[start : start + p3e23.BATCH_SIZE]
        query = torch.stack([record["query_features"] for record in current]).to(device)
        candidate = torch.stack(
            [record["count_candidate_features"] for record in current]
        ).to(device)
        global_features = torch.stack(
            [record["global_features"] for record in current]
        ).to(device)
        with _autocast_context(device, "bf16"):
            logits = count_model(query, candidate, global_features).float()
        if logits.shape != (len(current), p3e23.CLASSES):
            raise ValueError("P3-E24 Count output shape differs")
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("P3-E24 Count output is not finite")
        for index, record in enumerate(current):
            record["p3e16_count_logits"] = logits[index].cpu().to(torch.float16)


def router_metrics(rows: list[dict], policy: dict) -> dict:
    routed = p3e7.apply_policy(rows, policy)
    matched = target = 0
    chosen_candidate_indices = {}
    for row in rows:
        value = row["features"][policy["feature"]]
        route = (
            value >= policy["threshold"]
            if policy["direction"] == "ge"
            else value <= policy["threshold"]
        )
        chosen = row["expert"] if route else row["base"]
        matched += int(chosen["matched_atoms"])
        target += int(chosen["target_atoms"])
        chosen_candidate_indices[str(row["sample_key"])] = chosen["candidate_indices"]
    return {
        **{name: value for name, value in routed.items() if name != "routed_sample_keys"},
        "matched_atoms": matched,
        "target_atoms": target,
        "atom_recall": matched / target,
        "candidate_indices": chosen_candidate_indices,
    }


def evaluate_protocol(records: dict[str, list[dict]]) -> tuple[dict, dict]:
    metrics = {}
    rows_by_split = {}
    for split, values in records.items():
        base, expert = cached_outputs(values)
        rows = p3e7.collect_rows(values, base, expert)
        rows_by_split[split] = rows
        summary = p3e7.summarize(rows)
        routed = router_metrics(rows, p3e17.POLICY)
        routed.pop("candidate_indices")
        metrics[split] = {
            "base": {
                "exact": summary["base_exact"],
                "matched_atoms": summary["base_matched_atoms"],
                "target_atoms": summary["target_atoms"],
                "atom_recall": summary["base_matched_atoms"] / summary["target_atoms"],
            },
            "expert": {
                "exact": summary["expert_exact"],
                "matched_atoms": summary["expert_matched_atoms"],
                "target_atoms": summary["target_atoms"],
                "atom_recall": summary["expert_matched_atoms"] / summary["target_atoms"],
            },
            "router": routed,
        }
    return metrics, rows_by_split


def drift_metrics(
    baseline_records: list[dict], baseline_rows: list[dict],
    current_records: list[dict], current_rows: list[dict],
) -> dict:
    baseline_by_key = {str(record["sample_key"]): record for record in baseline_records}
    current_by_key = {str(record["sample_key"]): record for record in current_records}
    baseline_base, baseline_expert = cached_outputs(baseline_records)
    current_base, current_expert = cached_outputs(current_records)
    baseline_router = router_metrics(baseline_rows, p3e17.POLICY)["candidate_indices"]
    current_router = router_metrics(current_rows, p3e17.POLICY)["candidate_indices"]
    overlaps = []
    tail = 0
    pointer_mae = {"base": [], "expert": []}
    for key in sorted(baseline_by_key):
        old_ids = {
            int(baseline_by_key[key]["candidate_ids"][index])
            for index in baseline_router[key]
        }
        new_ids = {
            int(current_by_key[key]["candidate_ids"][index])
            for index in current_router[key]
        }
        overlaps.append(len(old_ids & new_ids) / max(len(old_ids), 1))
        tail += sum(index >= 128 for index in current_router[key])
        for name, old_output, new_output in (
            ("base", baseline_base[key], current_base[key]),
            ("expert", baseline_expert[key], current_expert[key]),
        ):
            old_probability = old_output["pointer_logits"].softmax(dim=-1)
            new_probability = new_output["pointer_logits"][:, :128].softmax(dim=-1)
            pointer_mae[name].append(
                float((old_probability - new_probability).abs().mean())
            )
    return {
        "router_candidate_id_overlap_mean": sum(overlaps) / len(overlaps),
        "router_tail_selections": tail,
        "base_first128_conditional_pointer_mae": sum(pointer_mae["base"]) / len(overlaps),
        "expert_first128_conditional_pointer_mae": sum(pointer_mae["expert"]) / len(overlaps),
    }


def paired_utility_metrics(
    baseline_records: list[dict], baseline_rows: list[dict],
    current_records: list[dict], current_rows: list[dict],
) -> dict:
    baseline_router = router_metrics(baseline_rows, p3e17.POLICY)
    current_router = router_metrics(current_rows, p3e17.POLICY)
    baseline_correct = {}
    current_correct = {}
    for name, rows, destination in (
        ("baseline", baseline_rows, baseline_correct),
        ("current", current_rows, current_correct),
    ):
        del name
        for row in rows:
            value = row["features"][p3e17.POLICY["feature"]]
            route = (
                value >= p3e17.POLICY["threshold"]
                if p3e17.POLICY["direction"] == "ge"
                else value <= p3e17.POLICY["threshold"]
            )
            destination[str(row["sample_key"])] = bool(
                row["expert"]["correct"] if route else row["base"]["correct"]
            )
    baseline_by_key = {str(record["sample_key"]): record for record in baseline_records}
    current_by_key = {str(record["sample_key"]): record for record in current_records}
    rescues = breaks = protected_breaks = correct_tail_hits = newly_reachable_rescues = 0
    for key in sorted(baseline_correct):
        rescues += int(current_correct[key] and not baseline_correct[key])
        breaks += int(baseline_correct[key] and not current_correct[key])
        protected = torch.equal(
            baseline_by_key[key]["atom_mask"], baseline_by_key[key]["full_atom_mask"]
        )
        protected_breaks += int(protected and baseline_correct[key] and not current_correct[key])
        newly_reachable = (
            not protected
            and torch.equal(
                current_by_key[key]["atom_mask"], current_by_key[key]["full_atom_mask"]
            )
        )
        newly_reachable_rescues += int(newly_reachable and current_correct[key])
        selected = current_router["candidate_indices"][key]
        tail = [index for index in selected if index >= 128]
        if tail:
            expected = current_by_key[key]["atom_anchors"][
                current_by_key[key]["full_atom_mask"]
            ].float()
            anchors = current_by_key[key]["candidate_anchors"][tail].float()
            correct_tail_hits += int((torch.cdist(anchors, expected) <= 6.0 / 768.0).any(dim=1).sum())
    return {
        "matched_atom_delta": (
            current_router["matched_atoms"] - baseline_router["matched_atoms"]
        ),
        "exact_rescues": rescues,
        "exact_breaks": breaks,
        "top128_oracle_protected_breaks": protected_breaks,
        "correct_tail_hits": correct_tail_hits,
        "newly_reachable_exact_rescues": newly_reachable_rescues,
    }


def arm_checks(
    arm: dict, baseline: dict, reachability: dict, utility: dict
) -> dict[str, bool]:
    return {
        "router_exact_net_loss_max12": (
            arm["router"]["exact"] >= baseline["router"]["exact"] - 12
        ),
        "router_matched_atom_net_loss_max64": (
            arm["router"]["matched_atoms"]
            >= baseline["router"]["matched_atoms"] - 64
        ),
        "base_exact_net_loss_max19": (
            arm["base"]["exact"] >= baseline["base"]["exact"] - 19
        ),
        "expert_exact_net_loss_max19": (
            arm["expert"]["exact"] >= baseline["expert"]["exact"] - 19
        ),
        "candidate_recall_min99": reachability["atom_recall"] >= 0.99,
        "router_matched_atom_gain_min69": utility["matched_atom_delta"] >= 69,
        "paired_rescues_ge_breaks": utility["exact_rescues"] >= utility["exact_breaks"],
        "top128_oracle_protected_breaks_max6": (
            utility["top128_oracle_protected_breaks"] <= 6
        ),
        "correct_tail_hits_nonzero": utility["correct_tail_hits"] > 0,
        "newly_reachable_exact_rescues_nonzero": (
            utility["newly_reachable_exact_rescues"] > 0
        ),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E24 output is not empty: {OUTPUT}")
    if file_sha256(P3E23_AUDIT) != P3E23_AUDIT_SHA256:
        raise ValueError("P3-E24 P3-E23 audit SHA256 differs")
    if file_sha256(p3e23.TOP256_CACHE) != p3e23.TOP256_CACHE_SHA256:
        raise ValueError("P3-E24 top256 cache SHA256 differs")
    for split in SPLITS:
        expected = p3e23.OLD_CACHE_SHA256[split]
        if file_sha256(p3e23.OLD_CACHE_ROOT / f"{split}.pt") != expected:
            raise ValueError(f"P3-E24 old cache SHA256 differs: {split}")
    p3e23_payload = json.loads(P3E23_AUDIT.read_text())
    for split in SPLITS:
        if file_sha256(p3e23.OUTPUT / f"{split}.pt") != p3e23_payload["shards"][split]["sha256"]:
            raise ValueError(f"P3-E24 global192 shard SHA256 differs: {split}")

    started = time.perf_counter()
    old = {
        split: torch.load(
            p3e23.OLD_CACHE_ROOT / f"{split}.pt", map_location="cpu", weights_only=False
        )["records"]
        for split in SPLITS
    }
    global192 = {
        split: torch.load(
            p3e23.OUTPUT / f"{split}.pt", map_location="cpu", weights_only=False
        )["records"]
        for split in SPLITS
    }
    top256 = torch.load(p3e23.TOP256_CACHE, map_location="cpu", weights_only=False)
    prefix_records, prefix_audit = build_prefix_records(top256, old)
    if not prefix_audit["all_top128_prefix_equal"]:
        raise RuntimeError("P3-E24 prefix protocol does not preserve top128 inputs")

    device = torch.device("cuda")
    base_model = p3e23.load_dynamic_model(p3e7.BASE_CHECKPOINT, p3e7.BASE_SHA256, device)
    expert_model = p3e23.load_dynamic_model(p3e7.EXPERT_CHECKPOINT, p3e7.EXPERT_SHA256, device)
    prefix192, compatibility = build_query_evidence(
        prefix_records, base_model, expert_model, device
    )
    del base_model, expert_model

    protocols = {"top128_baseline": old, "global192": global192, "prefix192": prefix192}
    metrics = {}
    rows = {}
    for name, records in protocols.items():
        metrics[name], rows[name] = evaluate_protocol(records)
    drift = {
        arm: {
            split: drift_metrics(old[split], rows["top128_baseline"][split], protocols[arm][split], rows[arm][split])
            for split in SPLITS
        }
        for arm in ("global192", "prefix192")
    }
    utility = {
        arm: {
            split: paired_utility_metrics(
                old[split], rows["top128_baseline"][split],
                protocols[arm][split], rows[arm][split],
            )
            for split in SPLITS
        }
        for arm in ("global192", "prefix192")
    }
    arm_gate = {
        arm: arm_checks(
            metrics[arm]["dev"],
            metrics["top128_baseline"]["dev"],
            p3e23_payload["reachability"]["dev"],
            utility[arm]["dev"],
        )
        for arm in ("global192", "prefix192")
    }
    passed = {arm: all(checks.values()) for arm, checks in arm_gate.items()}
    selected = "prefix192" if passed["prefix192"] else (
        "global192" if passed["global192"] else None
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    compatibility_names = {
        "router_exact_net_loss_max12",
        "router_matched_atom_net_loss_max64",
        "base_exact_net_loss_max19",
        "expert_exact_net_loss_max19",
        "candidate_recall_min99",
    }
    compatibility_passed = {
        arm: all(value for name, value in checks.items() if name in compatibility_names)
        for arm, checks in arm_gate.items()
    }
    structural_decision = (
        "query_extension_selected"
        if selected is not None
        else (
            "top192_shape_compatible_but_not_useful"
            if any(compatibility_passed.values())
            else "top192_query_extension_no_go"
        )
    )
    commitment = {
        "schema_version": "g3-p3e24-query-protocol-commitment-0.1",
        "audit": AUDIT_NAME,
        "structural_decision": structural_decision,
        "selected_protocol": selected,
        "selection_order": ["prefix192", "global192"],
        "arm_gate": arm_gate,
        "arm_passed": passed,
        "dev_metrics": {name: value["dev"] for name, value in metrics.items()},
        "dev_utility": {name: value["dev"] for name, value in utility.items()},
        "candidate_cache_sha256": p3e23.TOP256_CACHE_SHA256,
        "base_checkpoint_sha256": p3e7.BASE_SHA256,
        "expert_checkpoint_sha256": p3e7.EXPERT_SHA256,
        "router_policy": p3e17.POLICY,
        "source_sha256": file_sha256(Path(__file__)),
        "count_evaluated_before_commitment": False,
    }
    _atomic_json(OUTPUT / "query_protocol.commitment.json", commitment)
    commitment_sha256 = file_sha256(OUTPUT / "query_protocol.commitment.json")

    count_metrics = None
    count_checks = None
    if selected is None:
        decision = structural_decision
    else:
        count_model = p3e14.load_count_model(p3e17.COUNT_CHECKPOINT, device)
        for split in SPLITS:
            attach_count_logits(protocols[selected][split], count_model, device)
        del count_model
        count_metrics = {
            "top128_baseline": {
                split: p3e23.count_metrics(old[split]) for split in SPLITS
            },
            selected: {
                split: p3e23.count_metrics(protocols[selected][split]) for split in SPLITS
            },
        }
        baseline_dev = count_metrics["top128_baseline"]["dev"]
        selected_dev = count_metrics[selected]["dev"]
        baseline_train = count_metrics["top128_baseline"]["train"]
        selected_train = count_metrics[selected]["train"]
        count_checks = {
            "dev_exact_regression_max2pp": (
                selected_dev["accuracy"] >= baseline_dev["accuracy"] - 0.02
            ),
            "dev_mae_increase_max0_10": (
                selected_dev["mae"] <= baseline_dev["mae"] + 0.10
            ),
            "train_exact_regression_max2pp": (
                selected_train["accuracy"] >= baseline_train["accuracy"] - 0.02
            ),
            "train_mae_increase_max0_10": (
                selected_train["mae"] <= baseline_train["mae"] + 0.10
            ),
        }
        dev_compatible = (
            count_checks["dev_exact_regression_max2pp"]
            and count_checks["dev_mae_increase_max0_10"]
        )
        train_abnormal = not (
            count_checks["train_exact_regression_max2pp"]
            and count_checks["train_mae_increase_max0_10"]
        )
        if dev_compatible and not train_abnormal:
            decision = "top192_input_protocol_go_count_compatible"
        elif not dev_compatible and train_abnormal:
            decision = "top192_query_extension_go_count_adaptation_required"
        else:
            decision = "top192_count_drift_inconclusive_no_go"

    prefix_shards = {}
    for split, records in prefix192.items():
        path = OUTPUT / f"prefix192-{split}.pt"
        _atomic_torch_save(path, {
            "schema_version": "g3-p3e24-prefix192-evidence-0.1",
            "audit": AUDIT_NAME,
            "split": split,
            "records": records,
        })
        prefix_shards[split] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    audit = {
        "schema_version": "g3-p3e24-top192-input-protocol-ab-audit-0.1",
        "decision": decision,
        "selected_protocol": selected,
        "arm_gate": arm_gate,
        "arm_passed": passed,
        "selected_count_checks": count_checks,
        "count_metrics": count_metrics,
        "query_protocol_commitment_sha256": commitment_sha256,
        "prefix_protocol": prefix_audit,
        "compatibility": compatibility,
        "metrics": metrics,
        "drift": drift,
        "utility": utility,
        "prefix_shards": prefix_shards,
        "lineage": {
            "p3e23_audit_sha256": P3E23_AUDIT_SHA256,
            "top256_cache_sha256": p3e23.TOP256_CACHE_SHA256,
            "base_checkpoint_sha256": p3e7.BASE_SHA256,
            "expert_checkpoint_sha256": p3e7.EXPERT_SHA256,
            "count_checkpoint_sha256": p3e17.COUNT_SHA256,
            "router_policy": p3e17.POLICY,
            "old_confirm_read": False,
            "fresh_dev_read": False,
            "sealed_test_read": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps({
        "decision": decision,
        "selected_protocol": selected,
        "arm_gate": arm_gate,
        "selected_count_checks": count_checks,
        "count_metrics": count_metrics,
        "dev_metrics": {name: value["dev"] for name, value in metrics.items()},
        "dev_drift": {name: value["dev"] for name, value in drift.items()},
        "dev_utility": {name: value["dev"] for name, value in utility.items()},
        "elapsed_seconds": audit["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
