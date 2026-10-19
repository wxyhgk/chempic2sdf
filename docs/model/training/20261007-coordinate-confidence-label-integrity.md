# Coordinate confidence label integrity (2026-10-07)

The coordinate shadow has two different tasks that must not be conflated:

1. **Refinability:** a frozen proposal is within 10 px of some rendered atom
   center.  This target includes strict selector drops and near-miss
   candidates.  It is useful for deciding whether to apply a coordinate
   refiner, but it is not a selector-positive target.
2. **Strict selector recovery:** the proposal is a raw match within the
   existing 6 px matcher radius and was dropped by the selector.  This is the
   only strict recovery label.

On the released cache, candidate rows (protected groups excluded) are:

| split | strict | near-unknown | far | group-attached unknown |
|---|---:|---:|---:|---:|
| train | 359 | 1,121 | 27,305 | 13 |
| holdout | 286 | 655 | 13,985 | 13 |

The 13+13 attached rows are raw-unmatched members inside a strict-positive
group. They are neither confirmed negatives nor strict positives. The old
`collect()` path includes them, and the candidate head labels them solely by
distance (`<=10` positive, `>10` negative). That is a protocol bug: they must
be neutral and excluded from the candidate confidence loss and evaluation.

The GPU shadow's holdout assignment precision of 0.933 is conditional on 269
already-selected rows (251 refinable and 18 far). It is not end-to-end
candidate precision; strict recovery is only 44/286 (0.154), and refinable
precision/recall are 0.808/0.328. The CPU candidate preflight's 0.892/0.576
uses a different objective and 300 steps, so it cannot authorize the 100-step
joint GPU shadow.

Protocol correction: create a new release/receipt that excludes group-attached
unknown members, keeps refinability and strict recovery as separate metrics,
and reports assignment precision only as a conditional downstream metric.
Do not modify the previous release or retry the failed GPU shadow.
