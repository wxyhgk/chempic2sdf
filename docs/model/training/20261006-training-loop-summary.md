# Current-initial exploration loop (2026-10-06)

This note records the discussion → forward audit → bounded trial loop on the
fixed 1,200-image public cohort. Every trial started from
`initial.pt` (SHA-256
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`), used no
sealed identities, and kept production deployment unchanged.

## Evidence before training

- 232 raw-gap targets all have a dense-cell anchor within 6px, but only 45 are
  in heatmap top-128 and none of their target cells are local peaks.
- Anchor-aware replacement raises raw shadow matching, but the frozen set head
  removes many old selected IDs.
- Existing heatmap/role/offset/prominence ranking features do not make a
  residual sidecar precise; the corrected feature audit remains around 1%.
- Predicted-count decoding is unusable (count exact accuracy 1.9%, MAE 12.44).

## Bounded trials

### 1. Heatmap global-rank + raw retention

Only `atom.heatmap` (129 parameters) was updated for at most 200 steps. It
stopped at step 100:

- train raw recovered 3, lost 1;
- holdout raw recovered 24, lost 3;
- train selected lost 1;
- holdout selected lost 17.

The ranking loss can move missing cells upward, but raw-peak retention does not
protect selected ownership. Gate: **NO-GO**.

### 2. Visible-text gap ranking + selected-logit distillation

Only visible-text raw-gap cells were positive targets. Selected and raw cell
logits were distilled from initial. It stopped at step 100:

- train raw recovered 1, selected lost 3;
- holdout raw recovered 3, lost 10;
- holdout selected lost 15;
- holdout recovery floor (14) failed.

The extra text-specific supervision still changes downstream ownership. Gate:
**NO-GO**.

### 3. Frozen-model text objectness sidecar

A new 1x1 text-objectness head was trained in memory for 100 steps. The main
heatmap, selector and set head stayed frozen. Residual-only candidate precision
was:

| K | train | holdout |
|---:|---:|---:|
| 16 | 0.21% | 0.31% |
| 32 | 0.21% | 0.23% |

This is far below the 85% candidate gate, so no scorer or integration is
allowed.

### 4. Text-center objectness + center-to-anchor offset sidecar

A forward audit showed rendered text-center cells rank substantially better
than atom-anchor cells (median rank 203 vs 665), so a separate 1x1 score plus
offset head was trained for 100 steps. It still produced only:

| K | train | holdout |
|---:|---:|---:|
| 16 | 0.26% | 0.27% |
| 32 | 0.21% | 0.18% |

The target-location hypothesis is useful diagnostically but is not yet a
deployable residual candidate source.

## Decision

Freeze all three routes and do not spend GPU on more learning-rate/objective
sweeps. The remaining bottleneck is the representation/target contract before
candidate generation: visible text and implicit carbon need an endpoint
objectness target that is both spatially separable and ownership-compatible.

The next useful work is CPU/forward-only design of that contract (cell
assignment, multi-slot collision handling, selected-baseline replay, and
candidate labels). A new training trial is only justified after that contract
produces at least 50 train positives, precision ≥0.85, and 100% old selected
retention in a current-initial readiness audit.

Artifacts:

- `outputs/audits/chemvector-heatmap-rank-retention-trial-20261006/report.json`
- `outputs/audits/chemvector-heatmap-text-gap-trial-20261006/report.json`
- `outputs/audits/chemvector-text-endpoint-objectness-trial-20261006/report.json`
- `outputs/audits/chemvector-text-center-offset-sidecar-20261006-r2/report.json`
- `outputs/audits/chemvector-anchor-rank-features-20261006-r2/report.json`
