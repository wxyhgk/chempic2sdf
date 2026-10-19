# Rescue scorer readiness (2026-10-06)

## Scope and contract

This was a bounded, audit-only readiness check for the frozen high-resolution
atom sidecar.  The production checkpoint was not changed:

`outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt`

with SHA-256
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`.

The fixed 1,200-image cohort was split deterministically into 120 train-sidecar
images and 1,080 holdout images.  The frozen production atom threshold was used
for baseline assignment.  The existing high-resolution branch was reproduced
for 200 in-memory virtual steps; no RescueScorer optimizer was run and no
checkpoint or optimizer state was persisted.  The sealed test was not read.

Machine report:

`outputs/audits/chemvector-rescue-scorer-readiness-20261006-r1/report.json`

The contract-aligned recheck is recorded at
`outputs/audits/chemvector-rescue-scorer-readiness-20261006-r3/report.json`.

## Result

| split | candidates | positives | AUPRC |
|---|---:|---:|---:|
| train-sidecar | 524 | 19 | 0.1998 |
| holdout | 6,441 | 93 | 0.0945 |

Thresholds were pre-registered as `{0.25, 0.30, ..., 0.60}` and selected only
from train-sidecar labels.  No train threshold reached the required precision
floor of 0.85.  The best train point was 8 accepted / 3 matched (37.5%) at
`τ=0.50`; the best holdout point was 2 / 9 (22.2%) at `τ=0.60`.

The readiness gate is **NO-GO**:

```text
bounded_training_admissible = false
```

As a contract-aligned recheck, the candidate pool was then restricted to the
actual assignment-safe policy (top 16 peaks and at least 12 px from every
baseline-selected point).  This reduced the pool to 59 train candidates / 7
positives and 605 holdout candidates / 39 positives.  AUPRC was 0.2262 / 0.2627;
the best train threshold was only 2/7 (28.6%), while holdout reached 2/3
(66.7%) at `τ=0.60` but the train-side calibration had no positive at that
threshold.  The pre-registered train-only gate therefore remains NO-GO; the
holdout result is not a license to tune on holdout.

The previous bounded scorer trial independently confirms the same failure:
holdout precision was 88 / 2,309 = 3.81% after scorer fitting.  A learned
3→16→1 scorer cannot recover signal that is absent from the sidecar candidate
features.

## Decision

Do not train or integrate the RescueScorer, do not append sidecar candidates to
the selector, and do not alter the production checkpoint.  The next experiment
must improve candidate separability upstream, with a new train-only audit before
any scorer training.  Useful directions are denser/target-aware visual
supervision and explicitly stratified small-font, thick-line, and implicit
carbon examples; changing scorer capacity or threshold alone is not justified
by this evidence.
