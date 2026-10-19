# r17 endpoint ranking + graph/valence constraint shadow

## Scope

This is a bounded CPU attribution probe.  It starts from the frozen r17
endpoint-conditioned ranker and frozen candidate/type logits.  It does not
create candidates, alter the checkpoint, read sealed test data, or produce a
release decoder.  A target-free greedy selector sorts candidates by ranker
probability and accepts an edge only when the accumulated incident bond-order
budget at both endpoints remains at most `4.0` (`SINGLE=1`, `DOUBLE=2`,
`TRIPLE=3`, `AROMATIC=1.5`).

## Result

Report: `outputs/audits/chemvector-endpoint-rank-graph-constraint-cpu-r17/report.json`.

The train-only threshold selection picked `0.06` for the constrained selector.
On the fixed fresh holdout, rank-only was edge P/R `0.75517/0.99896` and typed
P/R `0.72401/0.95774`.  The valence-greedy selector was edge P/R
`0.75596/0.90024` and typed P/R `0.72135/0.85902`: recall fell by about
`9.87` percentage points, while typed precision also fell slightly.  Budgets
of `5` or higher made no change to rank-only, so the effect is not a useful
precision/recall tradeoff in this candidate distribution.

## Decision

**NO-GO.**  A generic endpoint valence budget cannot repair the current
candidate/type interface.  It rejects true crowded endpoints before the
chemical type problem is corrected.  Keep the ranker as an analysis/post-
processing candidate only; do not route it into production or start GPU work
for this variant.  Prioritize graph-level type correction and train-side
aromatic/ring coverage.
