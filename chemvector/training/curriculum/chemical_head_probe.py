"""Bounded chemical-output-only probe.

This is deliberately separate from the historical pair-scale launcher.  The
upstream structure path and the pair trunk are frozen; only the final chemical
linear layer is optimized.  The probe is a development experiment and never
promotes a checkpoint to the default inference profile.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import torch
from torch.nn import functional as F

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.training.curriculum.chemical_head_plan import CHECKPOINTS, draft_protocol
from chemvector.training.curriculum.chemical_head_stage import ChemicalHeadOnlyStage, chemical_head_loss
from chemvector.training.curriculum.chemical_head_stage import chemical_head_loss_with_aromatic_safety
from chemvector.training.curriculum.aromatic_ring_objective import AromaticRingObjectiveConfig
from chemvector.training.curriculum.checkpoints import file_digest
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import CurriculumCollator
from chemvector.training.curriculum.evaluation import FrozenOCRPredictor
from chemvector.training.curriculum.gradient_preflight import state_digest
from chemvector.training.curriculum.interface_diagnostic import tensor_tree_digest, write_json
from chemvector.training.curriculum.pair_cached_trial import frozen_digest
from chemvector.training.curriculum.pair_scale_plan import digest, sample_id
from chemvector.training.curriculum.pair_scale_training import source_checks
from chemvector.training.curriculum.release import RecordLoader
from chemvector.training.curriculum.runtime import ResourceBudget, run_budgeted
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.supervision_contract import (
    pair_targets_with_unknown,
    select_known_pair_rows,
)
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from chemvector.training.curriculum.validation_truth import canonical_molblock, expected_identity
from chemvector.training.curriculum.pair_scale_worker import INITIAL_STATE
from chemvector.train_direct_atom_pair_bond import match_atom_targets


ROOT = Path(__file__).resolve().parents[3]
PREPARATION = ROOT / "outputs/audits/chemvector-chemical-head-probe-preparation-20260925"
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_CHEMICAL_HEAD_OUTPUT",
    str(ROOT / "outputs/experiments/chemvector-chemical-head-probe-20261001-r3"),
))
AUTHORIZATION = Path(os.environ.get(
    "CHEMVECTOR_CHEMICAL_HEAD_AUTHORIZATION",
    str(ROOT / "outputs/experiments/chemvector-chemical-head-probe-20261001-r3.authorization.json"),
))
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
PREP_REPORT = PREPARATION / "report.json"
SAMPLES = PREPARATION / "training-samples.json"
PROTOCOL = PREPARATION / "protocol.json"
USE = os.environ.get("CHEMVECTOR_CHEMICAL_HEAD_USE", "local_chemical_head_probe_aromatic_safe_1200_once")
AUTH_USER_STATEMENT = os.environ.get(
    "CHEMVECTOR_CHEMICAL_HEAD_AUTH_STATEMENT",
    "开始训练：仅运行固定 chemical-head-only 芳香环安全试训",
)
AROMATIC_SAFETY_WEIGHT = 10.0
TRUST_REGION_WEIGHT = float(os.environ.get("CHEMVECTOR_CHEMICAL_TRUST_WEIGHT", "0.0"))
AROMATIC_SAFETY_CONFIG = AromaticRingObjectiveConfig()
AROMATIC_SAFETY_OBJECTIVE = {
    "enabled": True,
    "safety_weight": AROMATIC_SAFETY_WEIGHT,
    "mean_floor": AROMATIC_SAFETY_CONFIG.mean_floor,
    "minimum_floor": AROMATIC_SAFETY_CONFIG.minimum_floor,
    "mean_weight": AROMATIC_SAFETY_CONFIG.mean_weight,
    "minimum_weight": AROMATIC_SAFETY_CONFIG.minimum_weight,
}


def _objective_receipt() -> dict:
    value = dict(AROMATIC_SAFETY_OBJECTIVE)
    if TRUST_REGION_WEIGHT > 0.0:
        value["trust_region_weight"] = TRUST_REGION_WEIGHT
    return value


def _fresh_write(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return file_digest(path)


def _save_delta(path: Path, payload: dict) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite probe checkpoint: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".chemical-partial-", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return file_digest(path)


def _load_preparation() -> tuple[dict, dict, dict]:
    report = json.loads(PREP_REPORT.read_text(encoding="utf-8"))
    if not report.get("complete") or report.get("training_performed") or report.get("sealed_test_read"):
        raise ValueError("Complete non-authorizing chemical-head preparation is required")
    for name, expected in report["artifact_sha256"].items():
        path = PREPARATION / name
        if "/" in name or file_digest(path) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    plan = json.loads(SAMPLES.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if plan["samples_sha256"] != report["sample_sha256"] or digest(plan["samples"]) != plan["samples_sha256"]:
        raise ValueError("Fixed chemical-head sample list changed")
    expected_protocol = draft_protocol(
        sample_sha256=plan["samples_sha256"],
        preparation_sha256=report["source_preparation_sha256"],
        initial_sha256=report["initial_checkpoint_sha256"],
        manifest_sha256=report["release_manifest_sha256"],
    )
    if protocol != expected_protocol:
        raise ValueError("Chemical-head protocol changed")
    return report, plan, protocol


def authorize_probe() -> tuple[dict, dict, dict, dict]:
    """Bind the run to the immutable preparation and published training release.

    The exact authorization receipt is created outside this runner.  This
    function only verifies it; it never turns a conversational continuation
    message into training authority.
    """
    report, plan, protocol = _load_preparation()
    sources = source_checks()
    if file_digest(INITIAL) != report["initial_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint changed")
    if len(plan["samples"]) != 1200 or len({sample_id(r) for r in plan["samples"]}) != 1200:
        raise ValueError("Exactly 1,200 distinct training samples are required")
    manifest_rows = {sample_id(row): row for row in sources["manifest"]["samples"]}
    for row in plan["samples"]:
        key = sample_id(row)
        source = manifest_rows.get(key)
        if source is None or source["split"] != "train":
            raise ValueError("Probe sample is not an exact released train row")
        for field in ("canonical_smiles", "group_key", "molecule_id", "meta_sha256"):
            if row.get(field) != source.get(field):
                raise ValueError(f"Probe sample binding changed: {key}:{field}")
    if protocol["initial_checkpoint_sha256"] != report["initial_checkpoint_sha256"]:
        raise ValueError("Protocol/initial binding mismatch")
    authorization = {
        "schema_version": "chemical-head-probe-authorization-0.1",
        "use": USE,
        "user_statement": AUTH_USER_STATEMENT,
        "training_authorized": True,
        "deployment_authorized": False,
        "sealed_test_read": False,
        "new_identity_queries": 0,
        "preparation_report_sha256": file_digest(PREP_REPORT),
        "protocol_sha256": file_digest(PROTOCOL),
        "samples_sha256": plan["samples_sha256"],
        "initial_checkpoint_sha256": report["initial_checkpoint_sha256"],
        "release_manifest_sha256": report["release_manifest_sha256"],
        "budget": protocol["budget"],
        "trainable_parameters": protocol["trainable_parameters"],
        "frozen_parameters": protocol["frozen_parameters"],
        "aromatic_safety": {
            **_objective_receipt(),
            "objective_sha256": file_digest(ROOT / "chemvector/training/curriculum/aromatic_ring_objective.py"),
        },
    }
    if not AUTHORIZATION.is_file() or json.loads(AUTHORIZATION.read_text(encoding="utf-8")) != authorization:
        raise ValueError("Exact one-shot authorization receipt is absent or changed")
    return sources, report, plan, protocol


def _nonchemical_digest(model) -> str:
    return tensor_tree_digest({
        name: value for name, value in model.state_dict().items()
        if name not in {"pair.chemical.weight", "pair.chemical.bias"}
    })


def _chemical_digest(model) -> str:
    return tensor_tree_digest({
        name: value for name, value in model.state_dict().items()
        if name in {"pair.chemical.weight", "pair.chemical.bias"}
    })


def _presence_signature(output: dict[str, torch.Tensor], pairs: torch.Tensor, threshold: float) -> str:
    selected = output["presence_logits"].sigmoid() >= threshold
    return tensor_tree_digest({
        "pairs": pairs.detach(),
        "presence_logits": output["presence_logits"].detach(),
        "selected": selected.detach(),
    })


def _pair_output(stage, features, selected, pairs, rows):
    return stage(
        features.detach(),
        selected["atom_features"].detach(),
        selected["atom_geometry"][..., :2].detach(),
        selected["atom_scores"].detach(),
        pairs,
        rows,
    )


@torch.no_grad()
def _evaluate(model, config, manifest, loader, predictor, device, *, baseline_signatures=None, threshold=.85):
    model.eval()
    records = []
    parity = {}
    for row in manifest["samples"]:
        if row["split"] != "validation":
            continue
        sample = loader(manifest, row)
        if sample.get("purpose") != "diagnostic":
            raise ValueError("Validation sample must be diagnostic-only")
        pixels = image_tensor(sample["image"]).unsqueeze(0).to(device)
        structure = predictor.predict_structure(model, config, pixels)
        signature = _presence_signature(structure["output"], structure["pairs"], threshold)
        key = row["source"] + ":" + row["sample_key"]
        parity[key] = signature
        if baseline_signatures is not None and baseline_signatures.get(key) != signature:
            raise ValueError(f"Presence/selected-pair parity changed: {key}")
        predicted = predictor.from_structure(
            structure, model.role, model.bbox, pixels, sample["image"]
        )
        canonical = canonical_molblock(predicted["molblock"])
        expected = expected_identity(sample)
        records.append({
            "key": key,
            "domain": row["domain"],
            "valid": canonical is not None,
            "exact": canonical is not None and canonical == expected,
            "predicted_smiles": canonical,
            "expected_smiles": expected,
            "molblock": predicted["molblock"],
            "fallback": bool(predicted["fallback"]),
            "error": predicted["error"],
        })
    if len(records) != 944 or Counter(r["domain"] for r in records) != Counter({"full_molecule": 800, "curriculum": 144}):
        raise ValueError("Fixed 944-image development cohort changed")
    return records, parity


def _decision(before, after, *, complete: bool) -> dict:
    old = {r["key"]: r for r in before}
    new = {r["key"]: r for r in after}
    if old.keys() != new.keys():
        raise ValueError("Development identities changed")
    groups = {}
    for domain in ("full_molecule", "curriculum"):
        keys = [key for key in old if old[key]["domain"] == domain]
        groups[domain] = {
            "images": len(keys),
            "before": {k: sum(bool(old[key][k]) for key in keys) for k in ("exact", "valid", "fallback")},
            "after": {k: sum(bool(new[key][k]) for key in keys) for k in ("exact", "valid", "fallback")},
            "rescued": sum(not old[key]["exact"] and new[key]["exact"] for key in keys),
            "broken": sum(old[key]["exact"] and not new[key]["exact"] for key in keys),
        }
    full, course = groups["full_molecule"], groups["curriculum"]
    new_fallbacks = [key for key in old if not old[key]["fallback"] and new[key]["fallback"]]
    reasons = []
    if complete and full["after"]["exact"] <= full["before"]["exact"]:
        reasons.append("full_exact_not_strictly_higher")
    if full["after"]["valid"] < full["before"]["valid"]:
        reasons.append("full_valid_regression")
    if full["after"]["fallback"] > full["before"]["fallback"]:
        reasons.append("full_fallback_regression")
    if new_fallbacks:
        reasons.append("new_fallbacks")
    if complete and course["after"]["exact"] < course["before"]["exact"] - 2:
        reasons.append("curriculum_exact_drop_over_2")
    eligible = complete and not reasons
    return {
        "action": "complete_candidate_for_review" if eligible else ("complete_keep_initial" if complete else "continue"),
        "candidate_eligible": eligible,
        "reasons": reasons,
        "groups": groups,
        "new_fallback_keys": new_fallbacks,
        "broken_keys": [key for key in old if old[key]["exact"] and not new[key]["exact"]],
        "rescued_keys": [key for key in old if not old[key]["exact"] and new[key]["exact"]],
        "deployment_authorized": False,
        "independent_test_measured": False,
    }


def _training_step(
    model, stage, loader, manifest, row, collator, config, device,
    atom_threshold, pair_threshold, *, reference_chemical_state=None,
    trust_region_weight=0.0,
):
    sample = loader(manifest, row)
    batch = collator([sample])
    tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
    with torch.no_grad():
        forward = structure_forward(model, tensors["pixel_values"], config, atom_threshold)
        features, _, _, _, _, _, selected = forward
        mapping, _ = match_atom_targets(selected, tensors, config)
        contract = pair_targets_with_unknown(selected, mapping, tensors)
        chosen = select_known_pair_rows(contract, selected, negative_ratio=6, minimum_negatives=48)
        pairs, rows = contract["pair_indices"], contract["sample_indices"]
    if not len(chosen):
        return {"updated": False, "skip_reason": "no_known_pairs", "parity": None, "loss": None}
    output = _pair_output(stage, features, selected, pairs, rows)
    parity = _presence_signature(output, pairs, threshold=pair_threshold)
    chosen_output = {key: value[chosen] for key, value in output.items()}
    reference_logits = None
    if trust_region_weight > 0.0:
        if reference_chemical_state is None:
            raise ValueError("Trust-region training requires an initial chemical state")
        weight = reference_chemical_state["weight"].to(
            device=output["hidden"].device, dtype=output["hidden"].dtype,
        )
        bias = reference_chemical_state["bias"].to(
            device=output["hidden"].device, dtype=output["hidden"].dtype,
        )
        reference_logits = F.linear(output["hidden"][chosen], weight, bias)
    loss, safety = chemical_head_loss_with_aromatic_safety(
        chosen_output,
        contract["known_mask"][chosen],
        contract["legacy_targets"]["presence"][chosen],
        contract["legacy_targets"]["chemical"][chosen],
        pairs[chosen],
        rows[chosen],
        safety_weight=AROMATIC_SAFETY_WEIGHT,
        safety_config=AROMATIC_SAFETY_CONFIG,
        reference_chemical_logits=reference_logits,
        trust_region_weight=trust_region_weight,
    )
    if loss is None:
        return {"updated": False, "skip_reason": "no_positive_pairs", "parity": parity, "loss": None}
    if not bool(torch.isfinite(loss)):
        raise ValueError("Non-finite chemical loss")
    loss.backward()
    gradients = [parameter.grad for parameter in stage.trainable_parameters()]
    if any(gradient is None or not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise ValueError("Chemical head gradient missing or non-finite")
    norm = torch.nn.utils.clip_grad_norm_(stage.trainable_parameters(), 1.0, error_if_nonfinite=True)
    return {
        "updated": True,
        "skip_reason": None,
        "parity": parity,
        "loss": float(loss.detach()),
        "chemical_loss": safety["chemical_loss"],
        "aromatic_safety_loss": safety["aromatic_safety_loss"],
        "trust_region_loss": safety.get("trust_region_loss", 0.0),
        "trust_region_weight": float(trust_region_weight),
        "aromatic_rings": safety.get("rings", 0),
        "aromatic_reachable_edges": safety.get("reachable_aromatic_edges", 0),
        "gradient_norm": float(norm),
        "sampled_rows": len(chosen),
        "positive_rows": int(contract["legacy_targets"]["presence"][chosen].sum()),
    }


def worker() -> None:
    started = time.monotonic()
    sources, report, plan, protocol = authorize_probe()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Chemical-head probe requires explicitly selected GPU 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback for this probe")
    # ``run_budgeted`` creates the output directory and its two runtime
    # bookkeeping files before starting this worker.  They are launcher-owned
    # and do not count as prior probe artifacts; every other entry is refused.
    launcher_files = {"runtime.json", "worker.log"}
    if not OUTPUT.is_dir() or any(path.name not in launcher_files for path in OUTPUT.iterdir()):
        raise FileExistsError("Chemical-head probe output is not a fresh empty directory")
    auth_sha = file_digest(AUTHORIZATION)
    write_json(OUTPUT / "authorization.json", {
        "authorization_sha256": auth_sha,
        "user_statement": AUTH_USER_STATEMENT,
        "training_authorized": True,
        "deployment_authorized": False,
    })
    torch.set_num_threads(2)
    torch.manual_seed(protocol["seed"])
    torch.cuda.manual_seed_all(protocol["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    environment = {
        "device": str(device),
        "name": properties.name,
        "uuid": str(properties.uuid),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "dtype": "float32",
        "tf32": False,
        "autocast": False,
    }
    write_json(OUTPUT / "plan.json", {
        "schema_version": "chemical-head-probe-execution-0.1",
        "authorization_sha256": auth_sha,
        "protocol": protocol,
        "aromatic_safety": {
            **_objective_receipt(),
            "objective_sha256": file_digest(ROOT / "chemvector/training/curriculum/aromatic_ring_objective.py"),
        },
        "sample_sha256": plan["samples_sha256"],
        "initial_sha256": report["initial_checkpoint_sha256"],
        "environment": environment,
        "sealed_test_read": False,
        "deployment_authorized": False,
    })
    events_path = OUTPUT / "events.jsonl"
    counters = Counter(training_attempts=0, optimizer_steps=0, skipped_empty=0, evaluation_images=0)
    loader = RecordLoader(sources["manifest"])
    rows_by_key = {sample_id(row): row for row in sources["manifest"]["samples"]}
    sample_rows = [rows_by_key[sample_id(row)] for row in plan["samples"]]
    collator = CurriculumCollator(
        AtomBondSceneConfig(atom_queries=128, bond_queries=256), purpose="training"
    )
    model, config, payload = load_bundle_for_probe(INITIAL, report["initial_checkpoint_sha256"])
    model.to(device).eval()
    if state_digest(model) != INITIAL_STATE or payload["progress"]["optimizer_steps"] != 0:
        raise ValueError("Initial checkpoint/state differs")
    stage = ChemicalHeadOnlyStage(model.pair)
    initial_chemical_state = {
        name: value.detach().cpu().clone()
        for name, value in model.pair.chemical.state_dict().items()
    }
    predictor = FrozenOCRPredictor(payload["provenance"], device)
    frozen_before = _nonchemical_digest(model)
    upstream_before = frozen_digest(model)
    chemical_before = _chemical_digest(model)
    baseline, baseline_parity = _evaluate(
        model, config, sources["manifest"], loader, predictor, device,
        threshold=payload["provenance"]["pair_threshold"],
    )
    write_json(OUTPUT / "baseline-summary.json", {
        "records": baseline,
        "parity_count": len(baseline_parity),
        "chemical_sha256": chemical_before,
    })
    optimizer = torch.optim.AdamW(stage.trainable_parameters(), lr=protocol["optimizer"]["learning_rate"],
                                  weight_decay=protocol["optimizer"]["weight_decay"])
    seen = set()
    decisions = []
    parity_checks = 0
    checkpoint_set = set(CHECKPOINTS)
    with events_path.open("x", encoding="utf-8") as log:
        def emit(event, **fields):
            row = {"event": event, "elapsed_seconds": time.monotonic() - started, **fields}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(row, allow_nan=False), flush=True)

        for attempt, row in enumerate(sample_rows, 1):
            if time.monotonic() - started >= 1750:
                raise TimeoutError("Chemical-head probe hard budget reached")
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            key = sample_id(row)
            result = _training_step(
                model, stage, loader, sources["manifest"], row, collator, config, device,
                payload["provenance"]["atom_threshold"],
                payload["provenance"]["pair_threshold"],
                reference_chemical_state=initial_chemical_state,
                trust_region_weight=TRUST_REGION_WEIGHT,
            )
            counters["training_attempts"] += 1
            seen.add(key)
            if result["updated"]:
                optimizer.step()
                counters["optimizer_steps"] += 1
            else:
                counters["skipped_empty"] += 1
            optimizer.zero_grad(set_to_none=True)
            if _nonchemical_digest(model) != frozen_before or frozen_digest(model) != upstream_before:
                raise ValueError("Non-chemical model state changed")
            if result["parity"] is not None:
                # The parity signature was produced before the update. Re-run the
                # frozen inputs after the update and require exact presence bytes.
                sample = loader(sources["manifest"], row)
                batch = collator([sample])
                tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                with torch.no_grad():
                    forward = structure_forward(model, tensors["pixel_values"], config,
                                                payload["provenance"]["atom_threshold"])
                    features, _, _, _, _, _, selected = forward
                    pairs, pair_rows = enumerate_unordered_atom_pairs(selected["atom_mask"])
                    current = _pair_output(stage, features, selected, pairs, pair_rows)
                if _presence_signature(current, pairs, payload["provenance"]["pair_threshold"]) != result["parity"]:
                    raise ValueError(f"Presence/selected-pair parity changed after update: {key}")
                parity_checks += 1
            emit("train_attempt", attempt=attempt, key=key, **result)
            if attempt in checkpoint_set:
                current, current_parity = _evaluate(
                    model, config, sources["manifest"], loader, predictor, device,
                    baseline_signatures=baseline_parity,
                    threshold=payload["provenance"]["pair_threshold"],
                )
                counters["evaluation_images"] += len(current)
                complete = attempt == len(sample_rows) and len(seen) == len(sample_rows)
                decision = _decision(baseline, current, complete=complete)
                decisions.append({"attempt": attempt, **decision})
                write_json(OUTPUT / f"validation-{attempt:06d}.json", {"records": current})
                write_json(OUTPUT / f"decision-{attempt:06d}.json", decisions[-1])
                checkpoint = OUTPUT / f"chemical-{attempt:06d}.pt"
                checkpoint_sha = _save_delta(checkpoint, {
                    "schema_version": "chemical-head-probe-delta-0.1",
                    "initial_sha256": report["initial_checkpoint_sha256"],
                    "authorization_sha256": auth_sha,
                    "protocol_sha256": file_digest(PROTOCOL),
                    "chemical_state": {k: v.detach().cpu() for k, v in model.pair.chemical.state_dict().items()},
                    "optimizer_steps": counters["optimizer_steps"],
                    "decision": decisions[-1],
                    "deployment_authorized": False,
                    "resume_supported": False,
                })
                emit("checkpoint", attempt=attempt, checkpoint_sha256=checkpoint_sha, decision=decisions[-1])
    if len(seen) != 1200 or counters["training_attempts"] != 1200:
        raise ValueError("Fixed 1,200-image coverage incomplete")
    if _nonchemical_digest(model) != frozen_before:
        raise ValueError("Final non-chemical state changed")
    write_json(OUTPUT / "report.json", {
        "schema_version": "chemical-head-probe-report-0.1",
        "complete": True,
        "training_performed": counters["optimizer_steps"] > 0,
        "gpu_used": True,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "default_changed": False,
        "old_release_changed": False,
        "authorization_sha256": auth_sha,
        "initial_sha256": report["initial_checkpoint_sha256"],
        "sample_sha256": plan["samples_sha256"],
        "protocol_sha256": file_digest(PROTOCOL),
        "environment": environment,
        "counters": dict(counters),
        "distinct_training_images": len(seen),
        "parity_checks": parity_checks,
        "frozen_nonchemical_sha256": frozen_before,
        "upstream_sha256": upstream_before,
        "chemical_initial_sha256": chemical_before,
        "aromatic_safety": {
            **_objective_receipt(),
            "objective_sha256": file_digest(ROOT / "chemvector/training/curriculum/aromatic_ring_objective.py"),
        },
        "decisions": decisions,
        "elapsed_seconds": time.monotonic() - started,
    })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.check_only:
        report, plan, protocol = _load_preparation()
        print(json.dumps({"checks_passed": True, "images": len(plan["samples"]),
                          "training_performed": False, "sealed_test_read": False,
                          "protocol_sha256": file_digest(PROTOCOL)}, indent=2))
        return 0
    if args.worker:
        worker()
        return 0
    sources, report, plan, protocol = authorize_probe()
    if OUTPUT.exists():
        raise FileExistsError("Chemical-head probe output already exists; no retry")
    receipt = run_budgeted(
        [sys.executable, "-m", "chemvector.training.curriculum.chemical_head_probe", "--worker"],
        output=OUTPUT,
        budget=ResourceBudget(protocol["budget"]["wall_seconds"], ("0",)),
        training=True,
        admission=dict(sources["admission"], use=USE, training_run_authorized=True),
    )
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
