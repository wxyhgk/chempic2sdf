# r17 targeted visible O/N source t1 (2026-10-08)

## Data admission

The independent targeted builder rendered the candidate pool first and then
assigned molecule-disjoint splits by measured graph/layer strata.  It did not
reuse the existing r17 source identities.

- dataset: `outputs/datasets/chemvector-r17-targeted-source-20261008-t1/`
- admission: `outputs/audits/chemvector-r17-targeted-source-admission-20261008-t1/report.json`
- strata: `outputs/audits/chemvector-r17-targeted-source-strata-20261008-t1/report.json`
- 48 train / 24 holdout molecules, 8 profiles, 576 images
- identity, ledger, image hash, and split gates pass
- holdout molecule quotas: O-aromatic 8, N-triple 9, N-aromatic 8,
  O-double 10, O-single 8, graph-triple 9

The holdout quotas were deliberately enforced after rendering.  The initial
t1 implementation did not require equivalent train-side quotas, so the train
split contains too few N-aromatic examples; this is recorded as a limitation,
not hidden by re-splitting after evaluation.

## Pair-incidence CPU trial

`outputs/audits/chemvector-r17-targeted-source-pair-incidence-cpu-t1/report.json`

The 600-step frozen-pair residual improved aggregate edge recall from
`0.98464` to `0.99829` and typed recall from `0.86812` to `0.87537`.
The difficult profiles improved typed recall (`oa13 +1.00pp`, `oa33 +4.10pp`),
but oa33 typed precision fell `2.39pp`.  The raw sidecar therefore fails the
profile precision gate and is not integrated.

## Train-only calibration

Calibration was rerun on the complete 48/24 cohort, not the default 32/16
prefix.  Precision gates pass, but neither difficult profile has a holdout
typed-recall gain after the conservative train-only route:

- `profile_oa13_typed_recall_gain=false`
- `profile_oa33_typed_recall_gain=false`

No graph integration or GPU run is authorized from t1.

## Decision

The targeted holdout construction works, but t1 does not provide enough
rare-stratum training support.  A second source attempt using the same template
pool was rejected by the identity/coverage contract (only 52 complete new
molecules remained, and no new O-aromatic quota could be met).  Do not relax
identity admission or reuse old scaffolds.  The next model experiment should
address selector retention on the existing public pilot, where frozen
reachability shows raw candidates are present but stress and implicit-C
degree-2 candidates are dropped by selection.
