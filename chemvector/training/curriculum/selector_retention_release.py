"""Fixed 58-image fitting release; data admission is NOT GPU-run authorization.

Copy sixteen new witnesses through the existing lifecycle transition. Reuse
ten exact admitted aliases and thirty-two old training controls. New64 is
diagnostic/validation-only. Old releases and quarantine sources are immutable.
This standalone schema intentionally cannot be used by the old mixed launcher.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import torch

from . import data, publish as lifecycle, runtime
from .admission import identities, require_training_admission
from .release_evidence import read
from tools import audit_curriculum_v02_selector_matching as evidence

ROOT = evidence.REPO
PREP = ROOT / "outputs/audits/curriculum-v02-release-preparation-20260919"
OLD = ROOT / "outputs/datasets/chemvector-expanded-release-20260915"
OUTPUT = ROOT / "outputs/datasets/chemvector-selector-retention-fit-20260919"
REVIEW = ROOT / "docs/model/training/releases/20260919-selector-retention-data-review.json"
PREP_SHA = "39e66a3a4b0ba17651a5834fe3ac65fa8661fef247df98f58195f9e6fe496310"
TRIAL_SHA = "4111f716f2833c444a356550ad5332929c7b5bcbd74333fa349d201965b5afcf"
RIGHTS_SHA = "571302ab4b49efb6169a83f5460bf45e51262bff9560d21fd14e321d2e787238"
SCHEMA = "chemvector-selector-retention-fit-release-0.1"
USE = "local_selector_retention_fit_only"
WALL_SECONDS = 60


def validate_review(review):
    expected = dict(schema_version="chemvector-selector-retention-data-review-0.1",
        decision="approve_fixed_subset_local_data_publication_only",
        approved_preparation_report_sha256=PREP_SHA, approved_trial_proposal_sha256=TRIAL_SHA,
        approved_source_use_assessment_sha256=RIGHTS_SHA, copied_new_training_images=16,
        inherited_training_images=42, validation_only_references=64, use=USE,
        redistribution_authorized=False, training_run_authorized=False,
        full_curriculum_publication_authorized=False, source_mutation_authorized=False,
        private_identity_queries_authorized=0)
    if any(review.get(k) != v for k, v in expected.items()) or not review.get("reviewer") or not review.get("rationale"):
        raise ValueError("Exact bounded data-only review required")


def validate_manifest(manifest):
    if (manifest.get("schema_version") != SCHEMA or manifest.get("use") != USE
            or manifest.get("training_run_authorized") is not False):
        raise ValueError("Not the bounded selector data release")
    train, validation = manifest.get("training_samples", []), manifest.get("validation_only_samples", [])
    if not train or not validation:
        raise ValueError("Both training and separate validation references required")
    refs = [(r["source"], r["sample_key"]) for r in train + validation]
    if len(set(refs)) != len(refs):
        raise ValueError("Duplicate training/validation record")
    cache = {}
    for row in train + validation:
        if row["source"] not in manifest["sources"] or not Path(manifest["sources"][row["source"]]).is_absolute():
            raise ValueError("Explicit source root required")
        if row["canonical_smiles"] not in cache:
            cache[row["canonical_smiles"]] = identities(row["canonical_smiles"])
        if row["group_key"] != cache[row["canonical_smiles"]]["group_key"]:
            raise ValueError("Record grouping differs from chemistry")
    if any(r.get("split") != "train" for r in train) or any(r.get("purpose") != "diagnostic" for r in validation):
        raise ValueError("Invalid train/diagnostic split")
    for field in ("inchikey", "connectivity_prefix", "group_key"):
        if {cache[r["canonical_smiles"]][field] for r in train} & {cache[r["canonical_smiles"]][field] for r in validation}:
            raise ValueError("Fitting/validation identity overlap")


def load_row(manifest, row, *, purpose):
    if purpose not in {"training", "diagnostic"}:
        raise ValueError("Unsupported row purpose")
    allowed = manifest["training_samples"] if purpose == "training" else manifest["validation_only_samples"]
    if row not in allowed:
        raise ValueError("Row is outside exact authorized purpose; validation cannot be fitted")
    root = Path(manifest["sources"][row["source"]])
    path = root / "samples" / (row["sample_key"] + ".meta.json")
    if evidence.digest(path) != row["meta_sha256"]:
        raise ValueError("Released metadata changed")
    sample = data.load_sample(root, row["sample_key"], purpose=purpose)
    if (sample["graph"]["canonical_smiles"] != row["canonical_smiles"]
            or sample["meta"]["molecule_id"] != row["molecule_id"]):
        raise ValueError("Released record identity changed")
    return sample


def verify_release(root):
    root = Path(root)
    admission = json.loads(read(root / "admission.json"))
    require_training_admission(admission)
    raw = read(root / "manifest.json")
    if (sha256(raw).hexdigest() != admission.get("approved_manifest_sha256")
            or admission.get("use") != USE or admission.get("training_run_authorized") is not False):
        raise ValueError("Admission is not bound to this data-only release")
    manifest = json.loads(raw)
    validate_manifest(manifest)
    if (admission["checked_training_images"] != len(manifest["training_samples"])
            or admission["checked_validation_references"] != len(manifest["validation_only_samples"])):
        raise ValueError("Admission sample denominator mismatch")
    return manifest, admission


def publish(output=OUTPUT):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only data publication required")
    output = Path(output)
    if (not output.is_absolute() or output.resolve() != output or not output.is_dir()
            or any((output / name).exists() for name in ("samples", "manifest.json", "admission.json"))):
        raise ValueError("Fresh independent budgeted release directory required")
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started >= WALL_SECONDS:
            raise TimeoutError("Fixed publication deadline exceeded")
    review = json.loads(read(REVIEW))
    validate_review(review)
    prep = evidence.read_bound(PREP / "report.json", PREP_SHA)
    trial = evidence.read_bound(PREP / "trial-proposal.json", TRIAL_SHA)
    rights = evidence.read_bound(PREP / "source-use-assessment.json", RIGHTS_SHA)
    plan = evidence.read_bound(PREP / "execution-plan.json", prep["execution_plan_sha256"])
    inputs = dict(plan["input_sha256"])
    for path in (PREP / "report.json", PREP / "trial-proposal.json", PREP / "execution-plan.json",
                 PREP / "source-use-assessment.json", REVIEW, Path(__file__), Path(lifecycle.__file__),
                 ROOT / "tests/chemvector/test_selector_retention_release.py"):
        inputs[str(path)] = evidence.digest(path)
    evidence.verify_bindings(inputs, deadline)
    if (not prep["complete"] or prep["alias_target_parity_checks"] != 10
            or trial["fitting_origins"] != {"identical_admitted_training_alias": 10, "new_quarantine_needs_publication": 16}
            or rights["use"] != "local_research_only"):
        raise ValueError("Incomplete preparation or source-use assessment")
    old = json.loads(read(OLD / "manifest.json"))
    source_data = evidence.DATA
    for root in [source_data, *(Path(r) for r in old["sources"].values())]:
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Release output must be independent of original sources")
    corpus = json.loads(read(source_data / "sample-sha256.json"))
    source_rows = {(r["source"], r["sample_key"]): r for r in old["samples"]}
    new_source = "selector_retention_new16"
    sources = dict(old["sources"], **{new_source: str(output), "quarantine_validation_only": str(source_data)})
    manifest = dict(schema_version=SCHEMA, use=USE, sources=sources, training_samples=[], validation_only_samples=[],
        trial_proposal_sha256=TRIAL_SHA, preparation_report_sha256=PREP_SHA,
        training_run_authorized=False, old_development_release=dict(path=str(OLD / "manifest.json"),
            sha256=trial["existing_release"]["manifest_sha256"], validation_images=944))
    lifecycle.write(output / "publication-review.json", read(REVIEW))
    lifecycle.write(output / "trial-proposal.json", read(PREP / "trial-proposal.json"))
    (output / "samples").mkdir()
    report = dict(complete=False, optimizer_steps=0, gpu_used=False, training_performed=False,
        training_run_authorized=False, new_identity_queries=0, source_files_modified=0,
        input_sha256=inputs, copied_bundles=0, inherited_training_images=0,
        verified_training_images=0, verified_validation_references=0)
    try:
        for witness in trial["fitting_witnesses"]:
            deadline()
            key = witness["sample_key"]
            if witness["origin"] == "identical_admitted_training_alias":
                alias = witness["admitted_alias"]
                row = dict(source_rows[(alias["source"], alias["sample_key"])])
                if row["split"] != "train" or row["meta_sha256"] != alias["meta_sha256"]:
                    raise ValueError("Admitted fitting alias changed")
                report["inherited_training_images"] += 1
            else:
                payloads = {s: read(source_data / "samples" / (key + s)) for s in lifecycle.SUFFIXES}
                if any(sha256(raw).hexdigest() != corpus.get("samples/" + key + s) for s, raw in payloads.items()):
                    raise ValueError("New fitting bundle changed since full source audit")
                row = dict(sample_key=key, molecule_id=witness["molecule_id"], canonical_smiles=witness["canonical_smiles"],
                    group_key=identities(witness["canonical_smiles"])["group_key"], domain="curriculum", source=new_source,
                    split="train", meta_sha256=sha256(payloads[".meta.json"]).hexdigest())
                transformed = lifecycle.transition(payloads, row, plan_sha256=TRIAL_SHA, original_root=str(source_data))
                for suffix, raw in transformed.items():
                    lifecycle.write(output / "samples" / (key + suffix), raw)
                # Original bytes must remain unchanged; unchanged images/SDF/text
                # are independent copies, never symlinks or hardlinks.
                for suffix in lifecycle.SUFFIXES:
                    original = source_data / "samples" / (key + suffix)
                    dest = output / "samples" / (key + suffix)
                    if read(original) != payloads[suffix] or original.stat().st_ino == dest.stat().st_ino:
                        raise ValueError("Publication modified or aliased an original file")
                    if suffix not in {".meta.json", ".graph.json", ".layers.json"} and read(dest) != payloads[suffix]:
                        raise ValueError("Publication changed non-lifecycle artifact bytes")
                row["meta_sha256"] = sha256(transformed[".meta.json"]).hexdigest()
                report["copied_bundles"] += 1
            manifest["training_samples"].append({**row, "fit_domain": "witness", "diagnostic_sample_key": key})
        for row in trial["old_full_molecule_replay"]:
            if source_rows[(row["source"], row["sample_key"])] != row or row["split"] != "train":
                raise ValueError("Fixed old replay changed")
            manifest["training_samples"].append({**row, "fit_domain": "old_full_molecule"})
            report["inherited_training_images"] += 1
        for row in trial["new_group_validation"]:
            manifest["validation_only_samples"].append({k: row[k] for k in (
                "sample_key", "molecule_id", "canonical_smiles", "group_key", "meta_sha256")})
            manifest["validation_only_samples"][-1].update(source="quarantine_validation_only", purpose="diagnostic",
                domain="curriculum", use="new_group_development_only_not_training")
        validate_manifest(manifest)
        for purpose, rows in (("training", manifest["training_samples"]), ("diagnostic", manifest["validation_only_samples"])):
            collator = data.CurriculumCollator(purpose=purpose)
            for row in rows:
                deadline()
                sample = load_row(manifest, row, purpose=purpose)
                batch = collator([sample])
                if batch["training_authorized"] != (purpose == "training"):
                    raise ValueError("Target adapter authorization changed")
                field = "verified_training_images" if purpose == "training" else "verified_validation_references"
                report[field] += 1
        evidence.verify_bindings(inputs, deadline)
        if (report["copied_bundles"], report["inherited_training_images"], report["verified_training_images"],
                report["verified_validation_references"]) != (16, 42, 58, 64):
            raise ValueError("Fixed local release denominator mismatch")
        lifecycle.write(output / "manifest.json", manifest)
        admission = dict(schema_version="chemvector-curriculum-admission-0.1", training_authorized=True,
            use=USE, external_overlap_status="checked", coverage_gaps=[], training_run_authorized=False,
            approved_manifest_sha256=evidence.digest(output / "manifest.json"), approved_trial_proposal_sha256=TRIAL_SHA,
            publication_review_sha256=evidence.digest(REVIEW), checked_training_images=58, checked_validation_references=64,
            limitations=["Data-only admission, not authorization to execute the proposed GPU/CPU experiment",
                "Six old-validation diagnostic witnesses remain outside training; old944 remains unchanged",
                "58 fitting/replay images do not establish generalization; only 16 new copies published",
                "Original full curriculum remains quarantine; no general mixed training or deployment"])
        report.update(complete=True, approved_manifest_sha256=admission["approved_manifest_sha256"],
            admission_sha256=sha256(lifecycle.encoded(admission)).hexdigest(), elapsed_seconds=time.monotonic() - started)
        # Admission is the FINAL commit marker after all copies and tests pass.
        lifecycle.write(output / "report.json", report)
        lifecycle.write(output / "admission.json", admission)
    except Exception as exc:
        if not (output / "report.json").exists():
            report.update(error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.monotonic() - started)
            lifecycle.write(output / "report.json", report)
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        result = publish()
        print(json.dumps(dict(complete=result["complete"], copied=result["copied_bundles"])), flush=True)
        return 0
    result = runtime.run_budgeted([sys.executable, "-m", "chemvector.training.curriculum.selector_retention_release", "--worker"],
        output=OUTPUT, budget=runtime.ResourceBudget(WALL_SECONDS))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
