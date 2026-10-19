# Next bounded training protocol (release-gated)

This is a protocol draft, not a training authorization.  It is the only
training shape to use for the next experiment after a new endpoint-focused
dataset is admitted.  It deliberately separates release admission from
execution authority; neither quarantine data nor a release-preparation report
may be passed to the launcher.

## Current admission decision

No new run is admissible at the time of this note.

| artifact | evidence | decision |
|---|---|---|
| `curriculum-v02-release-preparation-20260919/report.json` | 19,824 candidates; `complete=true`, but `actual_publication=not_done`, `training_authorized=false` | no launch |
| `curriculum-v02-target-preflight-20260919/report.json` | 20,289 compatibility rows; `split_created=false`, `predicted_candidate_reachability=not_measured` | no launch |
| `chemvector-selector-retention-fit-20260919/admission.json` | data `training_authorized=true`, explicitly `training_run_authorized=false` | no rerun |
| `chemvector-selector-retention-trial-20260920/report.json` | 400 steps; `local_pass=false`, `development_nonregression=false` | route closed |
| `chemvector-expanded-release-20260915/admission.json` | authorized historical release, 14,095 images, hash-bound 1,200-step protocol | historical control only |

* `outputs/audits/curriculum-v02-release-preparation-20260919/report.json`
  is `passed_release_preparation_not_published`, with
  `training_authorized=false` and `dataset_published=false`.
* `outputs/audits/curriculum-v02-target-preflight-20260919/report.json` is
  target-compatibility only (`split_created=false`, `training_authorized=false`)
  and did not measure predicted reachability.
* `outputs/datasets/curriculum-v0.2-expanded-20260919/summary.json` and the
  targeted structure-expansion batches remain quarantine/diagnostic data.
* The 20260919 selector-retention bundle has data admission only:
  `training_authorized=true` but `training_run_authorized=false`; its 400-step
  retention trial already failed its local and development non-regression gates
  and must not be repeated.
* The 20260915 expanded release has a valid historical admission and a
  hash-bound 1200-step protocol, but its freeze-role and mixed continuation
  runs produced no selected improvement.  It is not a new endpoint experiment
  and should not be silently reused as evidence for one.

The missing artifacts are therefore a newly published, exact-byte-bound
endpoint-focused release and an explicit run authorization for that release.
Publication must include source-use, public/fresh identity, grouped split, and
model-target checks; execution additionally needs the fixed budget below.
The existing continuation launcher is not sufficient by itself for this run:
it restores the registry's current components, whereas this protocol must load
and verify the immutable `initial.pt` bundle before constructing the isolated
sidecar. A separate run-authorization receipt (bound to the manifest digest,
initial SHA, sidecar source digest, step ceiling, wall ceiling, and GPU index)
is required; adding a boolean to a quarantine report is not authorization.
For the group-selector route, the selector implementation, candidate-group
cache, and a forward-only audit/label manifest are also not present yet; those
must be produced and reviewed before a run receipt can be issued.

## Required release before execution

The release must pass `chemvector.training.curriculum.release.verify_release`
and satisfy all of the following before loading the model or CUDA:

1. `admission.json` has `training_authorized=true`,
   `external_overlap_status="checked"`, `coverage_gaps=[]`, and a manifest
   digest equal to `approved_manifest_sha256`.
2. Every train/holdout identity is disjoint by `group_key`, molecule ID,
   connectivity prefix, and canonical isomeric SMILES.  The holdout is fixed
   before any model forward and is never used for threshold or checkpoint
   selection.
3. The release manifest binds `initial_provenance` to the production sources;
   each source checkpoint is rehashed at worker startup.  The baseline file is
   `outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt`
   with SHA-256
   `09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`.
4. A fresh forward-only replay on the exact train/holdout rows records raw and
   selected endpoint coverage by style, element role, and bond type.  For the
   currently recommended group-selector route it must establish at least 50
   strict train positives and 14 holdout positives, all of which are existing
   raw candidates that the frozen selector dropped.  Unknown labels remain
   excluded rather than counted as negatives.  If raw candidates are absent,
   this route is NO-GO and data/rendering work resumes.
5. The release contains only the intended endpoint/bond objective.  OCR,
   external text checkpoints, sealed identities, and global/oracle matching
   are not inputs to training.
6. A run-authorization receipt, reviewed after the above checks, binds the
   exact manifest/admission hashes and immutable initial SHA to one run. It
   states `automatic_resume=false`, `max_gpus=1`, `max_steps=100`,
   `wall_seconds=1800`, and the fixed evaluation checkpoints below. Without
   this receipt the worker exits before model/CUDA setup.

## Single run (once all gates pass)

* Start from the immutable `initial.pt`; copy it to the fresh run directory and
  verify the SHA and its internal `progress.optimizer_steps == 0` before the
  first forward.  Record before/after digests of the source file and the
  frozen modules.  Never resume an older run and never overwrite `initial.pt`
  or the production registry.
* Keep the production vision encoder, atom proposal path, bond detector, OCR,
  and old selected prefix frozen.  Optimize only a newly admitted **group
  selector** (one predeclared small MLP/ranker per bond-junction/glyph group;
  no pointwise text sidecar, architecture sweep, or threshold sweep).  The
  old selected prefix is copied into the output first and is never rescored.
* Use a fixed group-disjoint 480/720 image split when the admitted pool has
  1200 rows (otherwise bind the exact counts in the manifest).  Labels require
  strict one-to-one selected atom identity and target `bond_atom_refs`; unknown
  pairs are omitted from the loss and all metrics.
* One GPU only, at most **100 optimizer steps or 1,800 seconds**, whichever
  comes first.  Batch size 1, deterministic seed from the manifest, AdamW with
  the predeclared learning rate, no automatic resume, no retry, and no
  threshold/K/capacity search.  Checkpoints are in-memory or a fresh run
  directory only; no deployment promotion.  The 1,200-step budget in the old
  20260915 protocol is historical and does not authorize this route.
* Evaluate at steps 0, 25, 50, 75, and 100 (or at the wall cutoff).
  Select any threshold from train rows only, then apply it once to holdout.
  Stop immediately on a failed invariant; do not continue to consume budget.

## Mandatory gates

All must pass for the run to be considered informative; otherwise classify
`NO-GO` and retain the immutable production baseline:

* baseline prefix retention is exactly 100% at every checkpoint;
* frozen module digests are byte-identical before/after;
* train/holdout identity overlap is zero and sealed-test access is false;
* strict residual precision on holdout is at least 0.85 (target 0.95), with
  reported unknown rows in the denominator separately and no oracle metrics;
* holdout strict residual recall improves over step 0 without increasing false
  selections or losing any reachable baseline bond/endpoint;
* per-stratum results are reported for `font_large/serif × visible N/O`,
  O-aromatic, N-double, and implicit-C around DOUBLE/AROMATIC; no aggregate
  gain may hide a stratum regression;
* a fixed old-capability replay (the 32 diagnostic witnesses and the 944-row
  development panel, both read-only) has no broken strict graph, no lost
  reachable bond, and no loss of prior exact predictions.  The 944-row panel
  remains development evidence, never a sealed-test claim; neither panel may
  supply gradients, threshold choices, or checkpoint-selection scores.

Only after all gates pass may a separate, explicitly authorized promotion
decision be considered.  A failed run cannot be repaired by lowering a
threshold, adding steps, changing K, or reusing its checkpoint in production.
