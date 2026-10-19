# Raw objectness label review

The raw-objectness CPU gate uses `protected + strict_positive` as the
positive class, `hard_negative` as negative, and masks `near_unknown` and
`selected_unmatched`. This label choice is correct only for an **atomness**
head: protected candidates are true atoms already retained by the frozen
baseline, so treating them as negatives would teach the head to reject valid
atoms.

It is not the deployment rescue objective. The rescue head must use strict
positive versus hard negative only, with protected and unknown rows masked.
At inference, protected baseline rows pass through unchanged; the rescue head
is called only for dropped rows. A single head trained on protected+strict
would mostly learn the baseline-kept distribution (143,091 protected versus
1,244 strict train rows) and can suppress the rare dropped atoms.

The raw gate report is therefore evidence for atomness only (core conservative
precision about 0.975 and raw recall about 0.994; stress precision about
0.911), not evidence for strict rescue or a production selector.

The train-core and dev-core molecule identities are disjoint. Stress shares
all 400 dev-core molecules and is a style-pressure test, not an independent
identity-generalization set. Before any GPU shadow, require strict-only rescue
metrics per style/profile, baseline-retention invariants, and no protected-row
regression. No GPU authorization follows from the raw-objectness gate alone.
