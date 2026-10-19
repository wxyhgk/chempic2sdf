# Collision-only candidate score + offset shadow (2026-10-17)

## Why this is a new route

The cap/stroke replay has many raw proposals within the 16 px audit radius, but
only 76.7--77.9% of visible-text targets have a proposal inside the deployed
6 px radius.  Existing attempts already closed the following routes: global or
local heatmap readout, peak-margin loss, offset-only training, raster-only local
encoder, rescue scorer, selector residual, and all candidate appends through the
set head.  Repeating any of those would not be an informative experiment.

The proposed shadow is narrower: it only ranks and refines *existing* raw
proposal slots.  It cannot create a candidate, enter the production selector,
change the baseline prefix, or claim graph recovery.  The frozen panel split is
molecule-disjoint (128 train molecules / 32 holdout molecules, six profiles per
molecule); the split must be re-verified by the preflight.

## CPU preflight first

Use the frozen initial checkpoint and the candidate-level manifest.  Fit a small
candidate score head and a 2D offset residual only on unprotected raw slots.
Refinable positives are raw proposals at 6--16 px from a target; strict
recovery is measured only after per-image one-to-one assignment.  Confirmed far
candidates are negatives; ambiguous many-to-one, near-unknown, and
collision-attached rows are masked.  Target IDs and distances are labels for
the audit, never inference features.

The train-only threshold must reach 0.90 precision, and the same threshold must
reach at least 0.85 precision on holdout.  Holdout must contain at least five
strict collision recoveries and improve over step zero.  Protected baseline
selected retention must remain exactly 1.0 and baseline raw targets lost must be
zero.  Any failure stops the route without GPU.

If the CPU gate passes, one separately receipted GPU shadow may run for at most
100 steps or 300 seconds.  It remains in-memory and writes only an audit report.
The pass/fail conditions are unchanged on the locked holdout; no threshold or
capacity sweep is permitted.  A failed shadow is terminal NO-GO.

Machine-readable protocol:
`outputs/audits/chemvector-collision-offset-shadow-20261017-r1/protocol.json`.
