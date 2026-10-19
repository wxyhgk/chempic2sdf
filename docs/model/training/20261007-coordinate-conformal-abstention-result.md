# Coordinate sidecar conformal abstention diagnostic (2026-10-07)

## Scope

This was a CPU-only, train-only diagnostic for the frozen coordinate candidate
gate and the local ridge coordinate sidecar.  It is a post-gate veto: it may
abstain from an already selected candidate, but it never proposes candidates
and cannot replace the selector.  The sealed test and production checkpoint
were not read or modified.

The train images were split deterministically by image identity (251 fit,
65 calibration).  A ridge coordinate refiner was fit on the fit images' near
rows (`raw_distance <= 10 px`).  The 90% split-conformal residual quantile was
calibrated only on the calibration rows that the frozen gate itself selected
(183 rows), with a ridge-leverage adjustment.

## Result

The calibrated radius was `q = 5.5578 px`.  On holdout, every one of the 610
frozen-gate selections had an estimated radius below 6 px, so the veto accepted
all 610 and abstained from none:

| metric | frozen gate | gate + conformal veto |
|---|---:|---:|
| selected/accepted | 610 | 610 |
| near precision | 0.8934 | 0.8934 |
| near recall | 0.5792 | 0.5792 |
| strict-only precision | 0.3508 | 0.3508 |

For comparison, calibrating over all near candidates (not conditioned on the
post-gate domain) gave `q = 7.4892 px` and rejected every selected candidate.
The two outcomes show that this residual/leverage uncertainty signal has no
useful per-candidate separation: it degenerates to all-accept or all-reject
under the strict 6 px criterion.

## Decision

**NO-GO for production integration.**  Keep the conformal result as a
diagnostic only.  Do not tune the quantile or call it a selector improvement;
the underlying strict selector precision remains unsolved.  A future
uncertainty route would need a separately trained, image-only error/ambiguity
model with a pre-registered calibration split and must still be evaluated as a
post-gate veto.

Authoritative report:

`outputs/audits/chemvector-targeted-selector-coordinate-conformal-abstention-20261007-r2/report.json`
