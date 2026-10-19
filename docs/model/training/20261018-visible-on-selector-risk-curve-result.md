# Visible O/N selector retention risk curve (2026-10-18)

This is a read-only replay of the frozen initial model on the 1,000-image
visible O/N pilot.  Raw proposals and set-head presence logits are frozen;
threshold offsets and top-K policies are fixed before inference.  Targets are
joined only after each target-free mask is formed.  No optimizer, checkpoint,
sealed split, or training path is used.

Artifact:

`outputs/audits/chemvector-visible-on-selector-risk-curve-20261017-r5/report.json`

## Aggregate curve

The baseline presence threshold is `-0.25`.  On the holdout, threshold offsets
move monotonically along a precision/coverage tradeoff:

| policy | strict coverage | candidate precision |
|---|---:|---:|
| offset -0.50 | 0.9361 | 0.8925 |
| offset 0.00 (baseline) | 0.9046 | 0.9071 |
| offset +0.50 | 0.8987 | 0.9158 |
| offset +1.00 | 0.8867 | 0.9217 |

Top-K is more damaging than a threshold offset: holdout strict coverage is
`.8504/.8954/.9088/.9096` at K `64/80/96/112`, while candidate precision is
`.7613/.6413/.5424/.4653`.  The current selected mask is safer than these
arbitrary top-K masks because it retains about `.9854` candidate precision in
the paired reachability audit.

## Focused strata

At the baseline offset, holdout stress strict coverage is `.9232` versus
`.9451` 6 px reachability; offset `+1.00` drops these to `.8725/.8858`.
Implicit-C degree-2 baseline strict coverage is `.9742` (train `.8363`).
Visible O/N baseline holdout strict coverage is `.9764/.9787`.

## Decision

The curve does not justify selector sidecar training: threshold changes cannot
recover the train stress/implicit-C gap without sacrificing coverage, and
top-K policies create a direct precision collapse.  Keep the current selector
frozen.  If a future CPU sidecar is considered, it must use additional
stress/implicit-C evidence and a fixed ownership objective; no GPU or
production merge follows this diagnostic.
