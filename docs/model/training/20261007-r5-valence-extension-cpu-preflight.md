# r5 valence/topology extension CPU preflight

## Scope

The extension contains 72 new catalog identities and 216 train-only images
(three weak cov13 profiles).  The identities are disjoint from the fixed r5
train and holdout sets.  Rendering, loader round-trip, and target-free line
reachability all passed; no sealed identity, production checkpoint, or GPU was
used.

## Result

The combined fixed-train plus extension edge-feature preflight selected every
candidate at its train-calibrated threshold.  On the fixed full holdout it
obtained candidate precision `0.922355`, versus `0.924724` for the frozen r5
shadow (`-0.002369`).  Candidate-positive recall was `1.0`, but this is not
graph-bond recall and therefore is not a release criterion.

Decision: **NO-GO for GPU shadow from this data-only extension**.  Keep the
extension quarantined and do not admit it to production training.  Continue
with the target-free Hungarian ownership and deterministic valence/topology
guards, whose value is measurable without another training run.

