# r17 t2 type-aware routing and type residual shadows

The t2 presence sidecar improved edge reachability but reduced typed precision.
Two CPU-only follow-ups were run on the same frozen t2 feature cache; neither
changed the production checkpoint or pair set.

## Frozen type-confidence gate

`outputs/audits/chemvector-r17-targeted-source-pair-type-aware-gate-cpu-20261018-t2/report.json`

Train-only confidence/margin selection chose confidence `0.60` and margin
`0.30`.  It admitted only two residual additions in the train split and no
useful oa13/oa33 additions on holdout.  Both difficult-profile typed-recall
gates were false.

## Type residual shadow

`outputs/audits/chemvector-r17-targeted-source-pair-type-residual-cpu-20261018-t2-r2/report.json`

A 600-step four-class type residual was trained from typed positive rows.  The
corrected evaluation keeps frozen chemical predictions on all baseline edges
and applies the residual only to newly admitted additions.  With the strict
train precision floor, it still admitted only two additions; holdout oa13 and
oa33 typed-recall gains were both zero.  This is a NO-GO, not a production
candidate.

## Interpretation

The t2 source has enough rare-strata coverage and the pair presence route can
reach the missing edges.  The remaining failure is a joint routing/type
objective problem: a presence-only residual cannot select additions that are
both structurally correct and typed correctly.  Do not expand data or start
GPU training yet.  The next bounded experiment should train a joint presence
plus type-aware residual (with hard non-edge/type negatives) on the same cache,
then apply the same train-only precision and independent oa13/oa33 gates.

## Joint presence+type probe

That joint 600-step CPU probe was also run:

`outputs/audits/chemvector-r17-targeted-source-pair-joint-residual-cpu-20261018-t2/report.json`

It did not pass the same gates.  The train-only precision-constrained route
again retained only two additions, and both oa13/oa33 holdout typed-recall
gains were zero.  This closes the current family of cheap sidecar/routing
fixes: more threshold tuning or another confidence gate is unlikely to recover
the missing edges without a real change to the upstream visual/type interface.
