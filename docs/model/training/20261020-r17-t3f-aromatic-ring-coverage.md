# r17 t3f aromatic/ring coverage quarantine

The t3e residual-density source had aromatic/ring molecules only in holdout.
That made the observed type correction result confounded: a frozen chemical
head could be asked to correct aromatic edges without seeing an independent
aromatic/ring training context.  t3f therefore reserves whole, identity-safe
molecules on both sides of the split before any model use.

Source:

`outputs/datasets/chemvector-r17-residual-density-source-20261020-t3f`

- 24 train and 12 holdout molecules; profiles `oa13` and `oa33` only.
- Train has 5 independent `GRAPH:AROMATIC` and 7 `RING` molecules.
- Holdout has 3 independent `GRAPH:AROMATIC` and 3 `RING` molecules.
- Historical molecule/scaffold/connection identities are disjoint, as are
  train and holdout molecules and rendered image bytes.

Receipts:

- `outputs/audits/chemvector-r17-residual-density-admission-20261020-t3f/report.json`
- `outputs/audits/chemvector-r17-residual-density-endpoint-tags-20261020-t3f/report.json`
- `outputs/audits/chemvector-r17-residual-density-strata-20261020-t3f/report.json`

This remains quarantine-only.  The source has only two profiles, so the
full-profile admission gate is intentionally false.  No model, checkpoint,
sealed test payload, or GPU was accessed.  Next action is a bounded CPU
feature/replay or graph-level type-correction shadow, followed by an
independent holdout gate; this source must not be promoted directly.
