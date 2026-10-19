"""One CPU-only frozen replay plus predeclared pair-input interventions."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import torch

from chemvector.atom_bond_scene import AtomBondSceneConfig
from chemvector.experiments import po_pair_attribution as probe
from chemvector.training.curriculum import bundle, continuation, data, pixels
from chemvector.training.curriculum.runtime import ResourceBudget, run_budgeted
from tools import diagnose_po_style_factorial as source
from tools import review_structure_expansion_identity as fixed

OUTPUT = fixed.REPO / "outputs/experiments/chemvector-po-pair-attribution-20260924"
SOURCE_SHA = "9375cb81615222ce8bf00bf0b4b649998e2429be0adc8f918ec58881332035af"
SOURCE_AUDIT_SHA = "b68f4a5e0f0dfb782ba75d965de85a307c6c8ece30ce56e8bde5fbe090561404"


def prepare():
    fixed.require(fixed.digest(source.OUTPUT/"report.json") == SOURCE_SHA
        and fixed.digest(source.AUDIT/"report.json") == SOURCE_AUDIT_SHA, "Audited factorial changed")
    old = json.loads(fixed.read(source.OUTPUT/"report.json"))
    plan = json.loads(fixed.read(source.OUTPUT/"plan.json"))
    fixed.require(old["complete"] and fixed.digest(source.OUTPUT/"plan.json") == old["plan_sha256"], "Source incomplete")
    bindings = dict(plan["input_sha256"])
    for path in (source.OUTPUT/"report.json", source.OUTPUT/"plan.json", source.AUDIT/"report.json",
                 Path(__file__).absolute(), Path(probe.__file__).absolute(),
                 fixed.REPO/"tests/data_sdf/test_po_pair_attribution.py"):
        bindings[str(path)] = fixed.digest(path)
    for name, expected in old["generated_manifest_sha256"].items():
        bindings[str(source.DATA/name)] = expected
    hashes = json.loads(fixed.read(source.DATA/"sample-sha256.json"))
    bindings.update({str(source.DATA/name): value for name, value in hashes.items()})
    cases = []
    for key, expected in sorted(old["case_sha256"].items()):
        path = source.OUTPUT/"cases"/(key+".json")
        bindings[str(path)] = expected
        cases.append(json.loads(fixed.read(path)))
    fixed.verify_bindings(bindings)
    fixed.require(len(cases) == 64 and all(len(c["po_bonds"]) == 3
        and all(b["presence_probability"] is not None for b in c["po_bonds"]) for c in cases), "Fixed reachable cohort required")
    return dict(schema_version="po-pair-attribution-plan-1", input_sha256=bindings,
        samples=[dict(sample_key=c["sample_key"], molecule_id=c["molecule_id"], bits=c["bits"]) for c in cases],
        thresholds=plan["thresholds"], provenance=plan["provenance"], planned_structure_forwards=64,
        planned_full_pair_forwards=64, planned_oracle_pair_forwards=64, planned_hybrid_readouts=32,
        branches=list(probe.BRANCHES), bits="0=12px source branch; 1=24px source branch",
        interventions="All eight encoded branch combinations; oracle changes only matched focus anchors to existing q1023 targets",
        scope="Diagnostic with target-based correspondence, not deployable correction or evaluation accuracy",
        logit_parity_atol=probe.ATOL, max_cpu_seconds=300, training_authorized=False,
        gpu_used=False, identity_queries=0, new_images=0, automatic_retries=0), cases


def assert_close(left, right, message):
    fixed.require(left.shape == right.shape and torch.allclose(left, right, atol=probe.ATOL, rtol=0), message)
    return float((left-right).abs().max()) if left.numel() else 0.0


def run():
    fixed.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "CPU-only probe")
    fixed.require(not OUTPUT.exists() and OUTPUT.resolve() == OUTPUT, "Fresh output required")
    torch.set_num_threads(2); torch.manual_seed(20260924)
    started = time.monotonic(); plan, old_cases = prepare()
    OUTPUT.mkdir(parents=True, exist_ok=False); (OUTPUT/"cases").mkdir()
    fixed.write_new(OUTPUT/"plan.json", plan)
    report = dict(status="incomplete", complete=False, plan_sha256=fixed.digest(OUTPUT/"plan.json"),
        model_loads=0, structure_forwards=0, full_pair_forwards=0, oracle_pair_forwards=0,
        hybrid_readouts=0, parity_passed=0, gpu_used=False, optimizer_steps=0, identity_queries=0,
        training_performed=False, training_authorized=False, default_model_changed=False,
        default_thresholds_changed=False, max_reconstruction_logit_error=0.)
    cases, comparisons, captures = [], [], {}
    try:
        metrics = source.prior.metrics
        model, config, payload = bundle.load_bundle(metrics.RUN/"initial.pt", expected_sha256=metrics.INITIAL_SHA)
        report["model_loads"] = 1
        fixed.require(payload["provenance"] == plan["provenance"] and payload["progress"]["optimizer_steps"] == 0
            and metrics.state_digest(model) == metrics.INITIAL_STATE, "Frozen initial required")
        model.eval().requires_grad_(False)
        widths = probe.branch_widths(model.pair)
        report.update(pair_config=asdict(model.pair.config), branch_widths=list(widths))
        collate = data.CurriculumCollator(AtomBondSceneConfig(atom_queries=config.maximum_atoms,
            bond_queries=config.maximum_bonds), purpose="diagnostic")
        with torch.inference_mode():
            for old in old_cases:
                sample = data.load_sample(source.DATA, old["sample_key"], purpose="diagnostic")
                report["structure_forwards"] += 1
                forward = continuation.structure_forward(model, pixels.image_tensor(sample["image"]).unsqueeze(0),
                    config, plan["thresholds"]["atom_threshold"])
                features, selected = forward[0], forward[6]
                report["full_pair_forwards"] += 1
                predicted, encoded = probe.capture_shared(model.pair, lambda: metrics.predict_pair(
                    model, features, selected, plan["thresholds"]["pair_threshold"]))
                fixed.require(metrics.prediction_digest(forward[3], selected, predicted) == old["prediction_sha256"],
                    "Exact unmodified baseline replay failed")
                report["parity_passed"] += 1
                # Baseline is complete before target-based correspondence or intervention.
                batch = collate([sample])
                fixed.require(metrics.measure(forward[3], selected, predicted, batch, config,
                    plan["thresholds"]["pair_threshold"]) == old["metrics"], "Baseline metrics changed")
                focus = sorted(old["po_bonds"], key=lambda b: b["bond_id"])
                indices = {tuple(p): i for i, p in enumerate(predicted["pairs"].tolist())}
                chosen = torch.tensor([indices[tuple(sorted(b["candidates"]))] for b in focus])
                pair_indices = predicted["pairs"][chosen]
                original_encoded = encoded[chosen].clone()
                rebuilt = probe.readout(model.pair, original_encoded)
                for name in ("presence_logits", "chemical_logits"):
                    error = assert_close(rebuilt[name], predicted["output"][name][chosen], "Shared reconstruction failed")
                    report["max_reconstruction_logit_error"] = max(report["max_reconstruction_logit_error"], error)
                anchors = probe.oracle_anchors(selected["atom_geometry"][..., :2], focus, batch["atom_anchors"])
                report["oracle_pair_forwards"] += 1
                oracle, oracle_encoded = probe.capture_shared(model.pair, lambda: model.pair(features,
                    selected["atom_features"], anchors, selected["atom_scores"], pair_indices,
                    torch.zeros(len(chosen), dtype=torch.long)))
                assert_close(original_encoded[:, widths[0]:], oracle_encoded[:, widths[0]:], "Oracle changed nonvisual branch")
                baseline_rows = probe.result_rows({k: predicted["output"][k][chosen]
                    for k in ("presence_logits", "chemical_logits")})
                oracle_rows = probe.result_rows(oracle)
                bonds = [dict(bond_id=b["bond_id"], expected_type=b["expected_type"], candidates=b["candidates"],
                    target_slots=b["target_slots"], endpoint_error_px=b["predicted_endpoint_error_px"],
                    baseline=base, oracle=ora) for b, base, ora in zip(focus, baseline_rows, oracle_rows)]
                case = dict(sample_key=old["sample_key"], molecule_id=old["molecule_id"], bits=old["bits"],
                    baseline_prediction_sha256=old["prediction_sha256"], bonds=bonds,
                    encoded_inputs=original_encoded.tolist(), oracle_encoded_inputs=oracle_encoded.tolist(),
                    source_feature_shape=list(features.shape), diagnostic_only=True)
                fixed.write_new(OUTPUT/"cases"/(case["sample_key"]+".json"), case)
                cases.append(case); captures[(case["molecule_id"], tuple(case["bits"]))] = original_encoded
                if len(cases) % 8 == 0:
                    print(json.dumps(dict(replayed=len(cases), parity=report["parity_passed"])), flush=True)
            lookup = {(c["molecule_id"], tuple(c["bits"])): c for c in cases}
            for (mid, bits), low in sorted(lookup.items()):
                if bits[1]: continue
                high_bits = list(bits); high_bits[1] = 1; high_key = (mid, tuple(high_bits))
                high = lookup[high_key]
                fixed.require([(b["bond_id"], b["expected_type"]) for b in low["bonds"]] ==
                    [(b["bond_id"], b["expected_type"]) for b in high["bonds"]], "Cross-style bond lineage mismatch")
                left, right = captures[(mid, bits)], captures[high_key]
                mixed = torch.cat([probe.mix_inputs(left, right, widths, mask) for mask in probe.MASKS])
                report["hybrid_readouts"] += 1
                output = probe.readout(model.pair, mixed)
                rows = probe.result_rows(output)
                # All-low/all-high must reconstruct the real-image raw predictions.
                for start, original in ((0, low), (21, high)):
                    assert_close(output["presence_logits"][start:start+3],
                        torch.tensor([r["baseline"]["presence_logit"] for r in original["bonds"]]), "Hybrid endpoint presence failed")
                    assert_close(output["chemical_logits"][start:start+3],
                        torch.tensor([r["baseline"]["chemical_logits"] for r in original["bonds"]]), "Hybrid endpoint chemistry failed")
                for i, bond in enumerate(low["bonds"]):
                    comparisons.append(dict(low_key=low["sample_key"], high_key=high["sample_key"],
                        bond_id=bond["bond_id"], expected_type=bond["expected_type"],
                        hybrids={"".join(map(str, mask)): rows[j*3+i] for j, mask in enumerate(probe.MASKS)}))
        fixed.require(metrics.state_digest(model) == metrics.INITIAL_STATE and not torch.cuda.is_initialized()
            and all(not p.requires_grad and p.grad is None for p in model.parameters()), "Frozen state/device failed")
        fixed.verify_bindings(plan["input_sha256"])
        fixed.write_new(OUTPUT/"comparisons.json", comparisons)
        report.update(status="completed_privileged_diagnostic", complete=True,
            summary=probe.summarize(cases, comparisons, plan["thresholds"]["pair_threshold"]),
            state_after=metrics.state_digest(model), comparisons_sha256=fixed.digest(OUTPUT/"comparisons.json"))
    except Exception as exc:
        report.update(status="failed_closed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
            case_sha256={c["sample_key"]: fixed.digest(OUTPUT/"cases"/(c["sample_key"]+".json")) for c in cases})
        fixed.write_new(OUTPUT/"report.json", report)
    return report


def audit():
    path = OUTPUT/"postcheck.json"
    fixed.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not path.exists(), "Fresh CPU-only postcheck required")
    report = json.loads(fixed.read(OUTPUT/"report.json")); plan = json.loads(fixed.read(OUTPUT/"plan.json"))
    fixed.require(report["complete"] and fixed.digest(OUTPUT/"plan.json") == report["plan_sha256"], "Incomplete probe")
    fixed.verify_bindings(plan["input_sha256"])
    cases = []
    for key, expected in report["case_sha256"].items():
        p = OUTPUT/"cases"/(key+".json"); fixed.require(fixed.digest(p) == expected, "Saved capture changed")
        cases.append(json.loads(fixed.read(p)))
    fixed.require(fixed.digest(OUTPUT/"comparisons.json") == report["comparisons_sha256"], "Saved hybrids changed")
    comparisons = json.loads(fixed.read(OUTPUT/"comparisons.json"))
    summary = probe.summarize(cases, comparisons, plan["thresholds"]["pair_threshold"])
    fixed.require(summary == report["summary"], "Aggregate changed")
    result = dict(status="passed_saved_aggregate_postcheck", source_report_sha256=fixed.digest(OUTPUT/"report.json"),
        cases=64, comparisons=96, model_loads=0, model_forwards=0, input_bindings_verified=len(plan["input_sha256"]),
        scope="Hashes and scalar aggregate replay only; nonlinear head activations are not re-executed", summary=summary)
    fixed.write_new(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "audit")); parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.action == "audit":
        print(json.dumps({k: v for k, v in audit().items() if k != "summary"})); return 0
    if args.worker: return 0 if run()["complete"] else 1
    receipt = run_budgeted([sys.executable, "-m", "tools.diagnose_po_pair_attribution", "run", "--worker"],
        output=OUTPUT.with_name(OUTPUT.name+"-runtime"), budget=ResourceBudget(300))
    print(json.dumps(receipt)); return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__": raise SystemExit(main())
