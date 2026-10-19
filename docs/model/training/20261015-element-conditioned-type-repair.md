# Element-conditioned type repair (2026-10-15)

The r10 direct AtomPair presence sidecar improved edge P/R but left chemistry
incorrect because the frozen chemical type logits were the bottleneck.  A
small shared type residual plus a target-free aromatic-cycle rule was helpful,
but the carbon-like valence cap was rejected: post-hoc audit found it
downgraded real O=P, O=S and C≡C bonds.

## Element probe

An isolated 1,419-parameter head (`LayerNorm(128) -> Linear(128, 11)`) was
trained on the frozen AtomObject features from the 384-image balanced train
subset.  It was never connected to production.

- independent dev element accuracy: `.98526`
- O accuracy: `1.0`
- N accuracy: `1.0`
- fresh holdout element accuracy: `.98190`

## Predicted-element repair audit

The element-conditioned order cap uses only predicted element labels, pair
geometry and frozen logits; target elements are joined after inference only for
the audit.  The shared type residual, aromatic-cycle veto and this
element-conditioned cap were evaluated without profile metadata.

| split | element acc. | edge P/R | type acc. | oracle sanitize |
|---|---:|---:|---:|---:|
| fresh holdout (95) | .98190 | 1.000/.966 | .97744 | 1.000 |
| dev (96) | .98526 | >.99/.98 | .9642 | .9583 |

All four dev profiles reached `.9583` oracle sanitize; the lowest profile type
accuracy was `.9567`.  These are post-hoc chemistry audits, not production
claims.  The production checkpoint SHA is unchanged and sealed test was not
read.

## Decision

The element-conditioned CPU path is the first candidate that passes both the
fresh holdout and independent dev chemistry gates.  Before any GPU shadow, the
element head contract, target-free sentinel, profile-independent inference and
per-profile edge/type metrics must be frozen in a single receipt.  No
carbon-like valence cap or profile-specific metadata may enter production.
