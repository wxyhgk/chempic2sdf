# Coordinate-refinement shadow result (2026-10-07)

## Protocol

The coordinate release contained the same 480 byte-identical image bundles as
the visual pool, plus independent train/holdout residual targets.  Candidates
within 10px of a rendered atom were refinement cases; they were never used as
selector negatives.  A new candidate head and coordinate refiner were trained
in memory for one authorized 100-step, single-GPU shadow.  The initial
production checkpoint and selected prefix were frozen.

## Result

Holdout at step 100:

| metric | value |
|---|---:|
| refinable candidates selected | 311 / 949 |
| refinable precision | 0.808 |
| refinable recall | 0.328 |
| strict selector recovery | 44 / 286 |
| one-to-one assignment precision | 0.933 |

The assignment figure is conditional on an oracle nearest-atom distance used
only for audit; it is not a deployable selector metric.  The candidate-head
precision/recall gates and strict-recovery gate failed, so the shadow is
**NO-GO**.  No production checkpoint was written and the initial checkpoint
hash is unchanged.

## Interpretation

The coordinate branch is useful: the CPU preflight and the conditional
assignment audit show that continuous refinement can bring near-miss points
closer to atom centers.  The failure is the learned refinable-candidate gate:
the 100-step MLP underfits/calibrates worse than the closed-form CPU ridge
preflight (holdout 0.892 precision / 0.576 recall).  This is not permission to
retry, extend, or sweep the failed run.

The next route must either freeze a validated CPU-linear refinement sidecar or
add a separately audited uncertainty/calibration head.  It must pass a new
identity-bound CPU preflight before any GPU authorization.

