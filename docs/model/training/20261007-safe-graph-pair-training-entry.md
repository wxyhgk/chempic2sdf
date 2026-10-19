# Safe graph/pair scorer entry (2026-10-07)

## Scope

This is a quarantine-only CPU experiment.  The reviewed scored vertex cache
was audited before any labels were opened by the scorer.  The production
checkpoint remained fixed at
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`.

Audit: `outputs/audits/chemvector-safe-graph-pair-training-entry-20261007/report.json`.
Pair run: `outputs/audits/chemvector-safe-graph-pair-cpu-20261007/report.json`.

## Isolation audit

- scored candidate artifact: 1,170 rows; SHA256
  `dde80da4c97551ba06848e4c74584949c415d6a1f6ef5cee86b1b5158b3dec01`;
- every row joins exactly one accepted render row, with 696 train and 474
  holdout images;
- candidate rows contain only `sample_key`, `source`, and inference-only
  `x/y/score` values; no target-like fields were present;
- train/holdout overlap is zero for `molecule_id`, `canonical_smiles`, and
  `connection_identity`;
- sealed test was not read, no production state was written, and no GPU was
  used.

## CPU pair scorer

The existing target-independent endpoint-pair feature builder was run for 300
CPU steps (751 parameters, 160/80 fixed image slice).  Features are formed
before post-hoc labels; each accepted residual edge requires two distinct
candidate/owner endpoints.  Train-only threshold calibration gave:

| split | selected | precision | recall | AUPRC |
|---|---:|---:|---:|---:|
| train | 1,147 | 0.8500 | 0.6925 | 0.8558 |
| holdout | 587 | 0.8688 | 0.7846 | 0.8835 |

The pair-level CPU gate therefore passes its minimum (`P>=0.80`, `R>=0.50`)
and demonstrates a reproducible, bounded training entry.  This does **not**
authorize graph insertion or GPU training: complete-edge precision/recall,
dangling rate, baseline-prefix retention, and per-profile graph metrics must
still be measured by an independent graph shadow.  The report explicitly
keeps `go=false` and the sidecar remains isolated.

## Reproducibility

The preflight now accepts `CHEMVECTOR_ENDPOINT_PAIR_CANDIDATES` and
`CHEMVECTOR_ENDPOINT_PAIR_STEPS` so the reviewed scored export can be selected
without editing protocol constants.  CPU runs are capped at 500 steps; this
run used 300.  No checkpoint or sidecar weights were promoted.
