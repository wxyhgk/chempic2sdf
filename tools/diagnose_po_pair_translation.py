"""One fixed nine-view, target-free pair sampling diagnostic on CPU."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.predicted_incidence import PredictedProposalConfig
from chemvector.experiments import pair_sampling_ensemble as method
from chemvector.training.curriculum import bundle, continuation, data, pixels
from chemvector.training.curriculum.runtime import ResourceBudget, run_budgeted
from tools import audit_structure_bond_diagnostic as saved
from tools import diagnose_po_style_factorial as source
from tools import review_structure_expansion_identity as fixed

OUTPUT = fixed.REPO/"outputs/experiments/chemvector-po-pair-translation-20260924"
ATTRIBUTION = fixed.REPO/"outputs/experiments/chemvector-po-pair-attribution-20260924"
METRICS = source.prior.metrics
SOURCE_SHA = "9375cb81615222ce8bf00bf0b4b649998e2429be0adc8f918ec58881332035af"
ATTRIBUTION_SHA = "f6fce1d120a780349946c6cb8a0d7310f99169789733bf7e21f3290d9a461366"


def prepare():
    fixed.require(fixed.digest(source.OUTPUT/"report.json") == SOURCE_SHA
        and fixed.digest(ATTRIBUTION/"report.json") == ATTRIBUTION_SHA, "Fixed source results changed")
    old = json.loads(fixed.read(source.OUTPUT/"report.json"))
    old_plan = json.loads(fixed.read(source.OUTPUT/"plan.json"))
    audit = json.loads(fixed.read(ATTRIBUTION/"postcheck.json"))
    fixed.require(old["complete"] and fixed.digest(source.OUTPUT/"plan.json") == old["plan_sha256"]
        and audit["source_report_sha256"] == ATTRIBUTION_SHA
        and audit["status"] == "passed_saved_aggregate_postcheck", "Completed source diagnostics required")
    bindings = dict(old_plan["input_sha256"])
    for path in (source.OUTPUT/"report.json", source.OUTPUT/"plan.json", ATTRIBUTION/"report.json",
                 ATTRIBUTION/"postcheck.json", Path(__file__).absolute(), Path(method.__file__).absolute(),
                 fixed.REPO/"tests/data_sdf/test_pair_sampling_ensemble.py"):
        bindings[str(path)] = fixed.digest(path)
    for name, value in old["generated_manifest_sha256"].items(): bindings[str(source.DATA/name)] = value
    inventory = json.loads(fixed.read(source.DATA/"sample-sha256.json"))
    bindings.update({str(source.DATA/name): value for name, value in inventory.items()})
    rows = []
    for key, value in sorted(old["case_sha256"].items()):
        path = source.OUTPUT/"cases"/(key+".json"); bindings[str(path)] = value
        case = json.loads(fixed.read(path))
        rows.append({name: case[name] for name in ("sample_key", "molecule_id", "bits")})
    fixed.require(len(rows) == 64, "Entire existing factorial required")
    fixed.verify_bindings(bindings)
    return dict(schema_version="po-pair-translation-plan-1", input_sha256=bindings, samples=rows,
        thresholds=old_plan["thresholds"], provenance=old_plan["provenance"], offsets_pixels=method.OFFSETS,
        aggregation="uniform mean edge/type joint probabilities; no maximum, tuning, or type-specific routing",
        changed="ribbon sampling anchors translated jointly; original graph anchors/features/masks retained",
        planned_structure_forwards=64, planned_full_pair_forwards=576, max_cpu_seconds=300,
        training_authorized=False, gpu_used=False, identity_queries=0, automatic_retries=0,
        gate="large-font P-O improves, zero lost original correct bonds, no FP/unknown-present count increase",
        scope="Already observed four-structure diagnosis; panel pass does not authorize deployment")


def promoted(pairs, output, threshold):
    chemistry, _ = METRICS.promote_complete_aromatic_rings(pairs, output["presence_logits"].sigmoid(),
        output["chemical_logits"], presence_threshold=threshold, mean_threshold=.40, minimum_threshold=.05)
    return dict(pairs=pairs, output=output, chemistry=chemistry)


def pack(predicted):
    return dict(pairs=predicted["pairs"].tolist(),
        presence_logits=predicted["output"]["presence_logits"].tolist(),
        chemical_logits=predicted["output"]["chemical_logits"].tolist(),
        promoted_chemical_ids=predicted["chemistry"].tolist())


def predict_views(head, features, selected, baseline, *, on_forward=lambda: None):
    """No labels, sample IDs, target coordinates, or focus selection accepted."""
    pairs = baseline["pairs"]
    expected_pairs, rows = METRICS.enumerate_unordered_atom_pairs(selected["atom_mask"])
    fixed.require(torch.equal(pairs, expected_pairs), "All selected pairs in unchanged order required")
    presence = [baseline["output"]["presence_logits"]]
    chemistry = [baseline["output"]["chemical_logits"]]
    anchors = selected["atom_geometry"][..., :2]
    for offset in method.OFFSETS[1:]:
        shifted = method.translated_anchors(anchors, selected["atom_mask"], offset, head.config.image_size)
        on_forward()
        output = head(features, selected["atom_features"], shifted, selected["atom_scores"], pairs, rows)
        presence.append(output["presence_logits"]); chemistry.append(output["chemical_logits"])
    presence, chemistry = torch.stack(presence), torch.stack(chemistry)
    return method.mean_joint_predictions(presence, chemistry), dict(
        presence_logits=presence.tolist(), chemical_logits=chemistry.tolist())


def run():
    fixed.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "CPU-only diagnostic")
    fixed.require(not OUTPUT.exists() and OUTPUT.resolve() == OUTPUT, "Fresh output required")
    torch.set_num_threads(2); torch.manual_seed(20260924)
    started = time.monotonic(); plan = prepare()
    OUTPUT.mkdir(parents=True, exist_ok=False); (OUTPUT/"cases").mkdir()
    fixed.write_new(OUTPUT/"plan.json", plan)
    report = dict(status="incomplete", complete=False, plan_sha256=fixed.digest(OUTPUT/"plan.json"),
        model_loads=0, structure_forwards=0, full_pair_forwards=0, parity_passed=0,
        optimizer_steps=0, gpu_used=False, identity_queries=0, automatic_retries=0,
        training_performed=False, training_authorized=False, deployed=False,
        default_model_changed=False, default_thresholds_changed=False)
    cases = []
    def count_pair(): report["full_pair_forwards"] += 1
    try:
        model, config, payload = bundle.load_bundle(METRICS.RUN/"initial.pt", expected_sha256=METRICS.INITIAL_SHA)
        report["model_loads"] = 1
        fixed.require(payload["provenance"] == plan["provenance"] and payload["progress"]["optimizer_steps"] == 0
            and METRICS.state_digest(model) == METRICS.INITIAL_STATE
            and model.pair.config.image_size == 768, "Frozen initial required")
        model.eval().requires_grad_(False)
        report["config"] = asdict(config)
        collate = data.CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
            bond_queries=config.maximum_bonds), purpose="diagnostic")
        with torch.inference_mode():
            for row in plan["samples"]:
                key = row["sample_key"]
                old = json.loads(fixed.read(source.OUTPUT/"cases"/(key+".json")))
                sample = data.load_sample(source.DATA, key, purpose="diagnostic")
                report["structure_forwards"] += 1
                forward = continuation.structure_forward(model, pixels.image_tensor(sample["image"]).unsqueeze(0),
                    config, plan["thresholds"]["atom_threshold"])
                selected = forward[6]
                count_pair()
                baseline = METRICS.predict_pair(model, forward[0], selected, plan["thresholds"]["pair_threshold"])
                digest = METRICS.prediction_digest(forward[3], selected, baseline)
                fixed.require(digest == old["prediction_sha256"], "Exact baseline parity failed")
                report["parity_passed"] += 1
                output, views = predict_views(model.pair, forward[0], selected, baseline, on_forward=count_pair)
                mixed = promoted(baseline["pairs"], output, plan["thresholds"]["pair_threshold"])
                fixed.require(METRICS.prediction_digest(forward[3], selected, baseline) == digest,
                    "Experimental sampling mutated baseline/selected inputs")
                # Only now attach targets for measuring the completed predictions.
                batch = collate([sample])
                old_metrics = METRICS.measure(forward[3], selected, baseline, batch, config, plan["thresholds"]["pair_threshold"])
                fixed.require(old_metrics == old["metrics"], "Baseline metric parity failed")
                measured = METRICS.measure(forward[3], selected, mixed, batch, config, plan["thresholds"]["pair_threshold"])
                fixed.require(measured["coverage"] == old_metrics["coverage"]
                    and measured["pair_contract"] == old_metrics["pair_contract"], "Candidate/matching coverage changed")
                case = dict(**row, baseline_prediction_sha256=digest, candidates=old["candidates"],
                    baseline_predictions=pack(baseline), pair_predictions=pack(mixed), views=views,
                    baseline_metrics=old_metrics, metrics=measured,
                    po_bond_ids=[b["bond_id"] for b in old["po_bonds"]], diagnostic_only=True)
                fixed.write_new(OUTPUT/"cases"/(key+".json"), case); cases.append(case)
                if len(cases) % 8 == 0:
                    print(json.dumps(dict(images=len(cases), pair_forwards=report["full_pair_forwards"])), flush=True)
        fixed.require(report["full_pair_forwards"] == 576 and METRICS.state_digest(model) == METRICS.INITIAL_STATE
            and not torch.cuda.is_initialized() and all(not p.requires_grad and p.grad is None for p in model.parameters()),
            "Frozen model/device/budget guard failed")
        fixed.verify_bindings(plan["input_sha256"])
        report.update(status="completed_diagnostic_only", complete=True, state_after=METRICS.state_digest(model),
            summary=method.summarize(cases), all_bond_summary=METRICS.summarize(cases))
    except Exception as exc:
        report.update(status="failed_closed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
            case_sha256={c["sample_key"]: fixed.digest(OUTPUT/"cases"/(c["sample_key"]+".json")) for c in cases})
        fixed.write_new(OUTPUT/"report.json", report)
    return report


def audit():
    fixed.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not (OUTPUT/"postcheck.json").exists(),
        "Fresh CPU-only postcheck required")
    torch.set_num_threads(2)
    report = json.loads(fixed.read(OUTPUT/"report.json")); plan = json.loads(fixed.read(OUTPUT/"plan.json"))
    fixed.require(report["complete"] and fixed.digest(OUTPUT/"plan.json") == report["plan_sha256"], "Complete run required")
    fixed.verify_bindings(plan["input_sha256"])
    fixed.require(plan["offsets_pixels"] == [list(v) for v in method.OFFSETS], "Offsets changed")
    config = PredictedProposalConfig(**report["config"])
    collate = data.CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
        bond_queries=config.maximum_bonds), purpose="diagnostic")
    cases = []
    for key, value in report["case_sha256"].items():
        path = OUTPUT/"cases"/(key+".json"); fixed.require(fixed.digest(path) == value, "Saved case changed")
        case = json.loads(fixed.read(path))
        raw, selected, predicted = saved.restore_for_metrics(case, config,
            plan["thresholds"]["atom_threshold"], plan["thresholds"]["pair_threshold"])
        old_case = {**case, "pair_predictions": case["baseline_predictions"]}
        _, _, baseline = saved.restore_for_metrics(old_case, config,
            plan["thresholds"]["atom_threshold"], plan["thresholds"]["pair_threshold"])
        presence = torch.tensor(case["views"]["presence_logits"], dtype=torch.float32)
        chemistry = torch.tensor(case["views"]["chemical_logits"], dtype=torch.float32)
        fixed.require(torch.equal(presence[0], baseline["output"]["presence_logits"])
            and torch.equal(chemistry[0], baseline["output"]["chemical_logits"]), "Zero-shift control changed")
        mixed = method.mean_joint_predictions(presence, chemistry)
        fixed.require(all(torch.equal(mixed[k], predicted["output"][k]) for k in mixed), "Saved nine-view aggregation changed")
        sample = data.load_sample(source.DATA, key, purpose="diagnostic"); batch = collate([sample])
        for predicted_values, metric_key in ((baseline, "baseline_metrics"), (predicted, "metrics")):
            measured = METRICS.measure(raw, selected, predicted_values, batch, config, plan["thresholds"]["pair_threshold"])
            fixed.require(measured == case[metric_key], "Saved pair score reconstruction changed")
        cases.append(case)
    summary = method.summarize(cases)
    fixed.require(summary == report["summary"] and METRICS.summarize(cases) == report["all_bond_summary"], "Summary changed")
    result = dict(status="passed_saved_translation_postcheck", source_report_sha256=fixed.digest(OUTPUT/"report.json"),
        cases_rescored=len(cases), input_bindings_verified=len(plan["input_sha256"]),
        model_loads=0, model_forwards=0, training_authorized=False,
        scope="Recompute saved logits mixture, aromatic promotion, matching and metrics; no hidden-feature replay", summary=summary)
    fixed.write_new(OUTPUT/"postcheck.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "audit")); parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.action == "audit":
        print(json.dumps({k: v for k, v in audit().items() if k != "summary"})); return 0
    if args.worker: return 0 if run()["complete"] else 1
    receipt = run_budgeted([sys.executable, "-m", "tools.diagnose_po_pair_translation", "run", "--worker"],
        output=OUTPUT.with_name(OUTPUT.name+"-runtime"), budget=ResourceBudget(300))
    print(json.dumps(receipt)); return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__": raise SystemExit(main())
