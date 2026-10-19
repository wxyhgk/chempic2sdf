# Selector route decision after the 480-image experiments

## Evidence collected

The fixed 480-image release is identity-disjoint (320 train / 160 holdout),
with 359 / 286 strict-positive groups.  The bounded group contract is valid,
but 346/359 train positives and 273/286 holdout positives are singletons.  The
problem is therefore candidate existence under a 1–2% positive prior, not
within-group ranking.

Four distinct trials were run without changing production:

1. **Pooled existence MLP, one GPU, 100 steps.** ROC-AUC was 0.935 train and
   0.929 holdout, but holdout conservative precision was 0.0817.  Train-only
   calibration had no threshold reaching 0.85 precision.
2. **CPU same-image hard-negative pairwise linear scorer.** It reversed global
   ordering (AUC 0.488 train / 0.510 holdout), demonstrating that hard-pair
   loss without absolute calibration is unsafe.
3. **CPU CVaR/top-tail scorer.** Fixed top-32 negative-tail penalty emitted
   only false positives at the score gate; precision and recall were both 0.
4. **CPU monotonic four-feature scorer.** Restricting the scorer to positive
   contributions from atom score, presence, component fraction, and ink center
   still produced only 0.109 train / 0.116 holdout precision at the fixed gate.

The independent read-only rank preflight agrees: holdout atom-score AUC is
0.871, but top-128 precision is 0.0625 and positive top-8 hit rate is 0.563.
Thus ROC-AUC is not the deployment bottleneck; the extreme negative tail is.

## Decision

Do **not** authorize another GPU selector run on this release.  Changing BCE
weights, scanning thresholds, changing the bounded grouping radius, or adding
more steps cannot satisfy the current gates.  The production checkpoint and
selected prefix remain unchanged.

## Next admissible route

The next work item is candidate-generation/visual evidence, not selector
capacity:

- keep the frozen initial model and protected prefix;
- add a genuine local crop encoder or better endpoint/glyph proposal source,
  rather than another shallow scalar head;
- construct a new exact-byte, identity-disjoint release containing reviewed
  hard-negative tails and enough positive rescue cases per style;
- run a CPU-only PR/tail preflight first, requiring train and holdout
  precision ≥0.85 and recall floor ≥0.50 at one predeclared gate;
- only then issue a new one-shot 100-step/30-minute GPU authorization.

The follow-up label-strata audit found an additional contract problem.  Among
confirmed negatives, 303 train members and 199 holdout members are within
10px of a rendered atom center.  These near-misses concentrate in the high
score tail and are not equivalent to background.  They must be treated as
unknown/soft coordinate cases, never silently relabeled as negatives.

A fixed raw-PNG centroid shift was then audited as a coordinate-refinement
readiness signal.  On near-target candidates it improved holdout mean distance
from 6.39px to 5.91px and the fraction within 6px from 0.349 to 0.565.  This
is promising evidence for a continuous coordinate-refinement branch, but it is
not a trained result and does not authorize a run.  The next release should
keep near-target bands out of the negative loss and evaluate refined-distance
gates before training any existence scorer.

No current report meets that preflight, so no additional GPU run is justified.
