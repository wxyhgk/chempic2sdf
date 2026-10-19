"""CPU-only release preparation from saved evidence; no publication or training.

Verify existing identity receipts, target journal and every retained bundle.
Partition by unioned identities and old holdouts, deduplicate old train PNGs,
and fix a single selector-retention fitting/validation panel and budget proposal.
No private worker, external corpus scan, model load or original-file writes.
"""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

from rdkit import rdBase
import torch

from chemvector.training.curriculum import data, release, runtime
from chemvector.training.curriculum.admission import identities
from chemvector.training.curriculum.publish import SUFFIXES
from chemvector.training.curriculum.release_evidence import read, verify_reviews
from tools import curriculum_v02_release_policy as policy
from tools import prepare_curriculum_v02_identity_request as request_check
from tools import audit_curriculum_v02_selector_matching as saved

ROOT = saved.REPO
DATA = saved.DATA
AUDITS = ROOT / "outputs/audits"
OLD = ROOT / "outputs/datasets/chemvector-expanded-release-20260915"
PUBLIC = AUDITS / "curriculum-v02-public-identity-20260919"
REQUEST = AUDITS / "curriculum-v02-identity-review-request-20260919"
TARGETS = AUDITS / "curriculum-v02-target-preflight-20260919"
HEADROOM = AUDITS / "chemvector-initial-headroom32-20260918"
OUTPUT = AUDITS / "curriculum-v02-release-preparation-20260919"
WALL_SECONDS = 300
BOUND = {
    str(PUBLIC / "report.json"): "4a9cb9a874b3ef1ac5f7862e23be42ddd639bb372cda1f760b4cfbd70f7ea8e1",
    str(PUBLIC / "remaining-candidates.json"): "3c91baa8137e4720761c8f9e6847dbeeb096dd48669a958c9ba1ac42b80a18ca",
    str(TARGETS / "report.json"): "3b599c7d2165c60295d928ea7fdbbe72a40a3d76c37eaf289c35c3696136003c",
    str(OLD / "manifest.json"): "67ccb1307f7af3ca19f16ee6fd900d1a2e8c3c62dd0b9c0ec28704ddd93aa05e",
    str(OLD / "admission.json"): "da6d4f2639c5294768a9d186edc060d104998a261eecd45471c12ed7b9a2e076",
    str(saved.SOURCE / "plan.json"): saved.PLAN_SHA,
    str(HEADROOM / "plan.json"): "2b5aedf18a4ab18613c19ca33a658a3471aaec2dff4cda73fa913ff5ce4d2aac",
    str(HEADROOM / "report.json"): "80787a19a7fbfb526a568de228e722f492068fd946cb77e5ef0942a80febaa68",
    str(AUDITS / "curriculum-v02-selector-cold-adamw6-20260919/report.json"):
        "0a71c29942c2f07572f8f3525b67be34930ca90add1b0bb13a15a5c107f0451a",
    str(ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"):
        "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1",
}


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def verify_bundle(root, key, corpus, quality, candidate, deadline):
    """Hash all eleven assets; reuse prior target verdict only for exact bytes."""
    payloads = {}
    for suffix in sorted(SUFFIXES):
        deadline()
        name = "samples/" + key + suffix
        raw = read(root / name)
        if sha256(raw).hexdigest() != corpus.get(name):
            raise ValueError("Retained source asset changed: " + name)
        payloads[suffix] = raw
    meta = json.loads(payloads[".meta.json"])
    if (meta["sample_key"] != key or meta["molecule_id"] != candidate["molecule_id"]
            or meta["canonical_smiles"] != candidate["canonical_smiles"]
            or meta["split"] != "quarantine" or meta["training_authorized"] is not False
            or meta["external_overlap_status"] != "unchecked"
            or meta["sha256"][".png"] != quality["png_sha256"]
            or any(sha256(raw).hexdigest() != meta["sha256"].get(suffix)
                   for suffix, raw in payloads.items() if suffix != ".meta.json")):
        raise ValueError("Retained metadata/lifecycle/identity mismatch")
    return {**quality, "meta_sha256": sha256(payloads[".meta.json"]).hexdigest()}, len(payloads)


def check_alias(new_sample, old_sample):
    """A byte-identical old image is reusable only with exactly equal targets."""
    if (new_sample["graph"]["canonical_smiles"] != old_sample["graph"]["canonical_smiles"]
            or new_sample["meta"]["sha256"][".png"] != old_sample["meta"]["sha256"][".png"]):
        raise ValueError("Admitted alias pixels/chemistry differ")
    collator = data.CurriculumCollator(purpose="diagnostic")
    a, b = collator([new_sample]), collator([old_sample])
    fields = [key for key in a if isinstance(a[key], torch.Tensor)]
    if (not fields or fields != [key for key in b if isinstance(b[key], torch.Tensor)]
            or any(not torch.equal(a[key], b[key]) for key in fields)):
        raise ValueError("Admitted alias target tensors differ; cannot silently substitute")
    return fields


def prepare(output=OUTPUT):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only release preparation required")
    output = Path(output)
    if not output.is_dir() or any((output / n).exists() for n in ("execution-plan.json", "report.json")):
        raise ValueError("Fresh budgeted output required")
    torch.set_num_threads(2)
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started >= WALL_SECONDS:
            raise TimeoutError("Release preparation CPU budget exhausted")
    inputs = dict(BOUND)
    def bound(path, expected=None):
        path = str(Path(path).absolute())
        raw = read(Path(path))
        actual = sha256(raw).hexdigest()
        if (expected is not None and actual != expected) or (path in inputs and inputs[path] != actual):
            raise ValueError("Release evidence binding changed: " + path)
        inputs[path] = actual
        return raw
    def document(path, expected=None):
        return json.loads(bound(path, expected))
    for path in BOUND:
        bound(path)
    paths = dict(request=REQUEST / "request.json", request_candidates=REQUEST / "candidate-identities.json",
        public_report=PUBLIC / "report.json", remaining=PUBLIC / "remaining-candidates.json",
        private_receipt=AUDITS / "curriculum-v02-private-identity-20260919/receipt/receipt.json")
    raw = {name: bound(path) for name, path in paths.items()}
    docs = {name: json.loads(value) for name, value in raw.items()}
    receipt = verify_reviews(raw, docs)  # Verifies saved allowlisted receipt; never calls the worker.
    candidates = docs["remaining"]
    if (len(candidates), sum(r["image_count"] for r in candidates)) != (3384, 19824):
        raise ValueError("Fixed cleared candidate set changed")
    target = document(TARGETS / "report.json")
    for path, digest in target["input_sha256"].items():
        bound(path, digest)
    handoff = document(target["handoff_path"])
    journal = bound(TARGETS / "samples.jsonl", target["journal_sha256"])
    request_check.validate_preflight(target, handoff, journal)
    quality = {r["sample_key"]: r for r in handoff["retained_candidates"]}
    target_rows = {r["sample_key"]: r for r in map(json.loads, journal.splitlines())}
    molecules = {r["molecule_id"]: r for r in document(DATA / "molecules.json")}
    corpus = document(DATA / "sample-sha256.json")
    generation = document(DATA / "plan.json")
    for path, digest in generation["input_sha256"].items():
        bound(ROOT / path, digest)
    for r in candidates:
        m = molecules[r["molecule_id"]]
        if (m["canonical_smiles"] != r["canonical_smiles"]
                or m.get("provenance", {}).get("kind") != "deterministic_hand_defined_chemical_templates"):
            raise ValueError("Unreviewed non-template source in new curriculum")
        for key in r["sample_keys"]:
            if quality[key]["molecule_id"] != r["molecule_id"] or not target_rows[key]["compatible"]:
                raise ValueError("Cleared candidate is not in passed target/quality inventory")
    manifest, admission, old_binding = release.verify_release(OLD / "manifest.json", OLD / "admission.json")
    # Persist every metadata input used by verify_release/FreshDevSource. No tar,
    # image, SDF, sealed path or private membership index is followed.
    for name, binding in manifest.get("validation_sources", {}).items():
        bound(binding["access_plan"], binding["access_plan_sha256"])
        for file in ("commitment.json", "validation.json", "samples.csv"):
            bound(Path(manifest["sources"][name]) / file)
    originals = release._validation_sources(manifest)
    cache, old_molecules, old_pixels = {}, {}, defaultdict(list)
    for row in manifest["samples"]:
        deadline()
        smiles = row["canonical_smiles"]
        if smiles not in cache:
            with rdBase.BlockLogs():
                cache[smiles] = identities(smiles)
        old_molecules[(row["molecule_id"], row["split"])] = {**row, **cache[smiles]}
        if row["source"] in originals:
            png = originals[row["source"]].rows[row["sample_key"]]["png_sha256"]
        else:
            path = Path(manifest["sources"][row["source"]]) / "samples" / (row["sample_key"] + ".meta.json")
            meta = document(path, row["meta_sha256"])
            if meta["split"] != row["split"] or meta["canonical_smiles"] != smiles:
                raise ValueError("Admitted old sample metadata differs from release")
            png = meta["sha256"][".png"]
        old_pixels[png].append(row)
    diagnostic = document(saved.SOURCE / "plan.json")
    decisions = policy.grouped_proposal(candidates, list(old_molecules.values()),
                                       {r["molecule_id"] for r in diagnostic["samples"]})
    headroom = document(HEADROOM / "plan.json")
    old_by_key = {(r["source"], r["sample_key"]): r for r in manifest["samples"]}
    replay_rows = headroom["samples"]
    if len(replay_rows) != 32 or any(old_by_key.get((r["source"], r["sample_key"])) != r for r in replay_rows):
        raise ValueError("Fixed old replay panel no longer matches admitted train release")
    copyright = Path("/usr/share/doc/fonts-dejavu-core/copyright")
    bound(copyright, "63d3ba759d12804c5b31a9d5940d855c1820d1f5999e6b0872eb1c7ff045fbc9")
    for path in (Path(__file__), Path(policy.__file__), Path(release.__file__), Path(runtime.__file__),
                 ROOT / "tools/selector_retention_prototype.py", ROOT / "tests/data_sdf/test_curriculum_v2_release_plan.py"):
        bound(path)
    execution = dict(schema_version="chemvector-v02-release-preparation-plan-0.1", input_sha256=inputs,
        seed=policy.SEED, cleared_molecules=3384, cleared_images=19824,
        split_policy="Union molecule_id/core/connectivity; exclude old holdout components; pin old train/diagnosed; untouched hash mod5 holdout",
        dedup_policy="Byte-identical old train image excluded from new count, reusable only as admitted exact-target alias",
        cpu_wall_seconds=WALL_SECONDS, gpu_used=False, automatic_retries=0, training_authorized=False,
        model_forwards=0, optimizer_steps=0, private_identity_queries=0, original_files_modified=False)
    write(output / "execution-plan.json", execution)
    report = dict(complete=False, training_authorized=False, training_performed=False, dataset_published=False,
        model_forwards=0, gpu_used=False, optimizer_steps=0, new_identity_queries=0, sealed_sample_access=False,
        evaluation_media_opened=False, copied_sample_files=0, original_files_modified=False,
        execution_plan_sha256=saved.digest(output / "execution-plan.json"))
    verified_assets, inventory = 0, {}
    artifacts = []
    try:
        for candidate in candidates:
            for key in candidate["sample_keys"]:
                inventory[key], count = verify_bundle(DATA, key, corpus, quality[key], candidate, deadline)
                verified_assets += count
            if len(inventory) // 1000 > (len(inventory) - candidate["image_count"]) // 1000:
                print(json.dumps(dict(event="verified_source_bundles", images=len(inventory))), flush=True)
        partitions = policy.partition_samples(candidates, decisions, inventory, old_pixels)
        trial = policy.select_trial(diagnostic["samples"], partitions, replay_rows)
        alias_checks = []
        for row in trial["fitting_witnesses"]:
            if row["origin"] != "identical_admitted_training_alias":
                continue
            deadline()
            alias = row["admitted_alias"]
            new_sample = data.load_sample(DATA, row["sample_key"], purpose="diagnostic")
            old_sample = data.load_sample(manifest["sources"][alias["source"]], alias["sample_key"], purpose="diagnostic")
            fields = check_alias(new_sample, old_sample)
            alias_checks.append(dict(sample_key=row["sample_key"], admitted_alias=alias, target_tensor_fields=fields,
                                     png_identical=True, tensors_identical=True))
        trial.update(schema_version="chemvector-selector-retention-trial-proposal-0.1",
            existing_release=old_binding, initial_checkpoint=ROOT.as_posix() + "/outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt",
            initial_sha256="09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1",
            model_mode="eval", parameter_scope="set_head only", learning_rate=5e-6,
            eps=1e-8, weight_decay=0, clip_norm=1., batch_size=1,
            training_cycle=["old_full_molecule", "old_full_molecule", "new_witness"],
            objective="original_full_set + 1.0 * per_image_positive_mean(softplus(-0.25 - atom_logit))",
            objective_source_sha256=inputs[str(ROOT / "tools/selector_retention_prototype.py")],
            thresholds=diagnostic["thresholds"],
            proposed_budget=dict(gpu_count=1, maximum_wall_seconds=600, maximum_optimizer_steps=400,
                automatic_retries=0, local_checkpoints_steps=[0, 100, 200, 400],
                early_stop="Stop at a local check if old32 loses a baseline strict-correct graph or reachable target bond; stop at step200 if fitting complete-endpoint decisions have not improved",
                new_validation_policy="Fixed 64 untouched groups, baseline and final only; no minibatch fitting or outcome reselection",
                full_regression_cpu_wall_seconds=2400, full_regression_gpu_count=0,
                full_regression_condition="Only if local finite-update gate passes; requires explicit run authorization"),
            launch_authorization="not_granted_for_this_proposal",
            alias_checks=alias_checks,
            fixed_acceptance=["Both C#C endpoints retained and its known wrong candidate rejected at fixed threshold",
                "No lost baseline-retained correct raw assignments or new false selections on the four core witnesses",
                "No lost baseline-reachable target bonds on old32 replay; no baseline strict-correct SDF becomes wrong",
                "New64 untouched-group endpoint coverage and false-selection count must not worsen; report full-image denominators",
                "Only after local pass, run old fixed944 development panel with paired rescued/broken and strict SDF metrics",
                "No automatic model replacement, threshold tuning, retry, or extending the wall/step ceiling"],
            limitations=["26 diagnostic witnesses and old32 are fitting/replay, not generalization accuracy",
                "Positive retention cannot repair an unproposed atom or guarantee safe negative scores",
                "Current raw matching contract remains unchanged; known assignment ambiguity is reported, not relabeled",
                "Proposal is not a runnable launcher or training admission; no new GPU run authorized"])
        source_use = dict(schema_version="chemvector-v02-source-use-assessment-0.1",
            reviewer="Codex local engineering assessment, not legal advice or independent certification",
            use="local_research_only", redistribution_authorized=False, training_launch_authorized=False,
            remaining_candidates_sha256=inputs[str(PUBLIC / "remaining-candidates.json")],
            sources=dict(hand_authored="Exact cleared project-authored deterministic chemical templates; no external images imported",
                renderer="Existing bound same-SVG rendering pipeline and generation provenance; chemistry/plausibility is not experimentally guaranteed",
                dejavu="Existing local DejaVu Sans/Serif font use; Bitstream Vera notice and DejaVu public-domain changes; no font redistribution",
                old_replay="Unchanged admitted old training rows and source-use scope; no new third-party data acquisition"),
            font_copyright_sha256=inputs[str(copyright)], generation_plan_sha256=inputs[str(DATA / "plan.json")])
        for name, doc in (("molecule-split-proposal.json", decisions), ("sample-split-proposal.json", partitions),
                          ("trial-proposal.json", trial), ("source-use-assessment.json", source_use)):
            write(output / name, doc)
            artifacts.append(name)
        saved.verify_bindings(inputs, deadline)
        counts = Counter(r["proposed_split"] for r in partitions)
        if sum(counts.values()) != 19824 or len(inventory) != 19824 or verified_assets != 19824 * len(SUFFIXES):
            raise ValueError("Release preparation accounting failed")
        new_train = [r for r in partitions if r["proposed_split"] == "train"]
        new_val = [r for r in partitions if r["proposed_split"] == "validation"]
        for field in policy.IDENTITY_FIELDS:
            if {r[field] for r in new_train} & {r[field] for r in new_val}:
                raise ValueError("New train/validation identity leakage")
        report.update(complete=True, status="passed_release_preparation_not_published",
            image_partition=dict(counts), molecule_partition=dict(Counter(r["proposed_split"] for r in decisions)),
            proposed_new_train_distinct_molecules=len({r["molecule_id"] for r in new_train}),
            proposed_new_validation_distinct_molecules=len({r["molecule_id"] for r in new_val}),
            old_validation_preserved_images=sum(r["split"] == "validation" for r in manifest["samples"]),
            new_train_validation_component_overlap=0, old_private_receipt_status=receipt["status"],
            fitting_witnesses=len(trial["fitting_witnesses"]), fitting_origins=trial["fitting_origins"],
            excluded_diagnostic_witnesses=len(trial["excluded_diagnostic_witnesses"]), old_replay_images=len(replay_rows),
            new_validation_probe_images=len(trial["new_group_validation"]), alias_target_parity_checks=len(alias_checks),
            readiness=dict(identity_receipts="verified_without_queries", target_compatibility="inherited_exact_byte_bound_full_pass",
                retained_payload_integrity="all_eleven_assets_per_image_rehashed", grouped_split="proposal_checked",
                source_use="bounded_local_research_assessed", actual_publication="not_done", training_launcher="not_implemented",
                training_budget="proposal_only_requires_user_approval"),
            limitations=["Original 19824 source images remain quarantine; this report cannot grant training admission",
                "Old-validation group exclusions do not mean the earlier full-identity screen was incorrect",
                "Cross-release dedup uses PNG bytes from admitted metadata; decoded RGB audit is inherited within new corpus only",
                "64 new validation groups are development data, not a sealed or external benchmark",
                "No old944 inference, new model prediction, gradients or checkpoint updates occurred"])
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report.update(verified_candidate_images=len(inventory), verified_source_assets=verified_assets,
            elapsed_seconds=time.monotonic() - started, input_bindings=len(inputs),
            artifact_sha256={name: saved.digest(output / name) for name in artifacts})
        write(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        report = prepare()
        print(json.dumps(dict(complete=report["complete"], partition=report["image_partition"])), flush=True)
        return 0
    result = runtime.run_budgeted([sys.executable, "-m", "tools.prepare_curriculum_v02_release_plan", "--worker"],
        output=OUTPUT, budget=runtime.ResourceBudget(WALL_SECONDS))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
