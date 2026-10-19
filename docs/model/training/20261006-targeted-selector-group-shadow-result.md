# Targeted group-selector shadow result (2026-10-06)

The fixed 130-image slice was admitted independently and one run receipt was
issued for a single 100-step, one-GPU, 1,800-second shadow.  The run optimized
only a new 225-parameter scorer over the frozen group cache.  It did not load
or write the production model, did not read sealed data, and did not alter the
immutable `initial.pt`.

## Result

| metric | step 0 | step 100 | gate |
|---|---:|---:|---|
| holdout strict recall | 0.625 | 0.875 | improved |
| holdout strict precision | 0.0078 | 0.0126 | **fail; required >= 0.85** |
| holdout selected groups | 1,287 | 1,110 | still too many |
| holdout confirmed-negative selections | 1,277 | 1,096 | not acceptable |
| baseline prefix retention | 100% | 100% | pass |

The run therefore has `pass=false`.  Its recall gain is not deployable: the
scorer activates nearly every negative group, so the precision failure is more
than a calibration edge case.

## Decision

This run is terminal under the fixed receipt.  No retry, threshold sweep,
extra steps, or checkpoint reuse is allowed.  The production baseline remains
unchanged and the shadow scorer is not promoted.

The evidence rejects the first implementation of a pointwise sigmoid over
group members.  A next attempt would need a predeclared abstaining group-level
objective/inference contract (with confirmed-negative groups not forced to
emit a winner), plus a fresh cache/readiness audit.  It must be a new bounded
experiment, not a continuation of this failed run.
