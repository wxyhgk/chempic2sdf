# r17 pair-incidence CPU training-entry review (2026-10-18)

This review is planning-only.  No training or GPU work was started.

## Evidence

The v7 source is the first admissible full-profile source artifact:

- admission: `outputs/audits/chemvector-r17-new-source-admission-20261008-v7/report.json`;
- source: `outputs/datasets/chemvector-r17-new-source-20261008-v7/manifest.json`;
- 128 train / 64 holdout molecules, 8 profiles, 1,024 / 512 images;
- molecule/canonical/connection/scaffold overlap, ledger, image-hash, and
  split gates pass.

The already completed 600-step CPU sidecar is:

`outputs/audits/chemvector-r17-new-source-pair-incidence-cpu-v7/report.json`

It froze the checkpoint, atom set, unordered pair set, and type heads.  The
calibrated v8 route and safe graph integration improved holdout edge recall
`.95425 → .95767`, typed recall `.75180 → .75594`, and exact typed topology
`.36523 → .37891`, with no precision/sanitize regression.  The target
perturbation digest is identical.

The t1 targeted source is not a valid next training source despite passing
metadata admission: its 600-step sidecar improved difficult-profile recall,
but oa33 typed precision fell 2.39 percentage points before calibration, and
the train rare-stratum quotas were insufficient.  Its train-only calibration
produced no oa13/oa33 holdout typed-recall gain.  Do not rerun t1 or reuse its
state as a release candidate.

## Minimal CPU training entry (only after a repaired source)

The next source must be a new append-only batch (v8 or later), with train and
holdout quotas enforced after rendering, not merely planned.  Require, per
profile and on both splits, at least eight molecule identities for each of:

- visible O aromatic;
- visible N aromatic;
- visible O double;
- visible N triple;
- any graph triple.

The exact quota may be raised if the rendered inventory permits it; no old
source identity or public train row may be relabeled as holdout.

After metadata admission and the strengthened strata audit pass, the one
bounded CPU run is:

```bash
R17_PAIR_COHORT=outputs/datasets/<new-source>/manifest.json \
R17_PAIR_SOURCE=outputs/datasets/<new-source>/samples \
R17_PAIR_OUT=outputs/audits/<new-source>-pair-incidence-cpu \
R17_PAIR_TRAIN_MOLECULES=<train-count> \
R17_PAIR_HOLDOUT_MOLECULES=<holdout-count> \
R17_PAIR_STEPS=600 \
python tools/train_r17_pair_incidence_known_clean_cpu.py
```

Then run calibration once, with explicit molecule counts (never the trainer
defaults):

```bash
R17_PAIR_CALIB_COHORT=outputs/datasets/<new-source>/manifest.json \
R17_PAIR_CALIB_STATE=outputs/audits/<new-source>-pair-incidence-cpu/sidecar-state.pt \
R17_PAIR_CALIB_OUT=outputs/audits/<new-source>-pair-calibration-cpu \
R17_PAIR_TRAIN_MOLECULES=<train-count> \
R17_PAIR_HOLDOUT_MOLECULES=<holdout-count> \
python tools/audit_r17_pair_incidence_calibration_cpu.py
```

Finally run safe graph integration and its target-perturbation sentinel with
the same cohort/state/calibration report.  These are already implemented in
`tools/audit_r17_pair_incidence_graph_integration_cpu.py`; no new model or
production checkpoint is needed.

## Hard CPU gates

Every gate below is required; otherwise stop without GPU:

1. checkpoint SHA, production state, and frozen pair-set digest unchanged;
2. source admission, ledger, train/holdout identity disjointness, and the
   strengthened per-profile train/holdout strata quotas pass;
3. sidecar uses only frozen pair features; unknown rows are not converted to
   negatives; exactly 600 CPU steps and one state artifact are recorded;
4. train-only calibration uses explicit full-cohort counts and incurs zero
   train precision degradation;
5. independent holdout edge and typed precision drops are at most 1 pp per
   profile, while oa13 and oa33 typed recall each improve by at least 0.5 pp;
6. safe graph integration has no sanitize drop, no exact-topology regression,
   and retains all baseline edges;
7. target-perturbation routing/compile digest is byte-identical to normal;
8. no sealed data is read, no GPU is used, and `production_go=false` remains
   explicit.

The existing v7 artifacts satisfy the integration shape but not the rare
strata release requirement, so they remain shadow evidence.  There is no
current authorization for another pair sidecar run until a repaired source
passes the data gates.

