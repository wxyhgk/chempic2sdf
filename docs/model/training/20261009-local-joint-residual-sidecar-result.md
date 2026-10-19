# Local pair residual sidecar result

## Protocol

This was the single bounded trial admitted by the forward-only local-pair
audit. The current `initial.pt`, vision/atom/bond/set path, OCR, and existing
pair head were frozen. Only a new 481-parameter residual MLP was optimized for
800 steps. Its cache contained known strict local pairs with frozen presence
probability `< 0.85`; unknown pairs were excluded and existing baseline edges
were never rescored or replaced. The split was 480/720 group-, molecule-, and
canonical-SMILES-disjoint. No sealed-test data or production checkpoint was
read/written.

## Result

| split | residual candidates | strict positives | AUPRC | best train-selected threshold |
|---|---:|---:|---:|---|
| train | 91,683 | 345 | 0.838 | no threshold reached 0.95 precision (best 0.8425 at 0.98) |
| holdout | 125,404 | 534 | 0.883 | no admissible threshold |

At the train-selected grid, the best point was threshold `0.98`, selecting 254
rows with 214 matches (`0.8425` precision, `0.6203` recall). Because the
pre-registered residual precision floor was `0.95`, no threshold was selected;
the holdout deployment-like gate therefore has zero admissible claims by
construction. A high AUPRC alone is not sufficient for graph insertion.

Frozen model state SHA256 was identical before and after:

`f85ff136932a0f356828cbb2a03b9f9052f75408c8834e620291f69976eb350f`

## Decision

**NO-GO.** The residual sidecar does not meet the precision gate. Do not lower
the threshold, increase MLP size, or extend training steps. The existing pair
head is already about 98.5% conservative precision including unknown-selected
rows on the holdout threshold replay, so this residual scorer does not provide
a safe graph gain.

The remaining major limitation is endpoint coverage: 793 target bonds are
unreachable because at least one atom was not selected by the frozen atom
selector. A pair sidecar cannot recover those endpoints. The next investigation
should therefore return to raw-vs-selector endpoint coverage (especially
font-large/serif visible O/N and implicit C), with the current initial kept as
the production baseline.

Artifact:

`outputs/audits/chemvector-local-joint-residual-sidecar-20261009-r2/report.json`
