# Vertex-to-bond NO-GO diagnosis and next route (2026-10-17)

The first vertex-to-bond integration shadow is `NO-GO`.  The failure must not
be interpreted as proof that the vertex detector is unusable:

* the r5 vertex field has high isolated recall;
* the new candidate path has better label precision than the legacy path, but
  still below the integration gate;
* many attached candidates are dangling because the opposite endpoint has no
  compatible candidate;
* the frozen endpoint geometry is itself a possible ceiling.

The holdout report makes the last point material.  With target anchors as an
oracle, only a small fraction of frozen endpoints are within the fixed 12 px
attachment radius.  A candidate score cannot move an endpoint, and an
endpoint-compatibility classifier cannot recover an endpoint that is outside
its geometric support.  Therefore a score-only trial is not the next action.

## Decision tree before any GPU budget

Run a CPU/read-only endpoint reachability panel on the fixed r5 holdout.  For
the frozen bond endpoints, report the fraction of target endpoints reachable at
6, 12, 16, 20 and 24 px, p50/p90 distances, and the same values by profile,
degree, and O/N/implicit-C incidence.  Also report complete two-endpoint bond
reachability.  Do not use these distances to generate candidates; they are an
upper-bound diagnostic only.

Do not use the fraction of *all active endpoint proposals* within the radius as
the sole ceiling: the frozen bond head emits many low-quality proposals, so
that denominator measures contamination as well as geometry.  The decisive
upper bound is the fraction of **target bonds** for which at least one frozen
proposal has both endpoints within the radius.  If target-bond recall at 12 px
is below 0.70 (and remains below 0.70 at 24 px), stop this route: the frozen
bond endpoint head must be repaired or retrained separately.  If target-bond
recall is high while all-proposal endpoint reachability is low, the problem is
proposal compatibility/filtering, which is exactly what the sidecar can test.

The r5 holdout panel currently shows 3,178/3,234 target bonds reachable at 12
px (0.9827) and 3,234/3,234 at 24 px, while only 13.46% of all active
endpoints lie within 12 px.  This supports one bounded **endpoint
compatibility + optional offset sidecar** shadow; it does not support a scalar
candidate score-only head.

The first sidecar should be endpoint-conditioned compatibility only. For each
frozen bond proposal and each nearby vertex candidate, a small MLP scores the
candidate/endpoint pair. A deterministic complete-edge reducer then keeps a
pair of distinct candidates only when both endpoint scores pass the single
train-selected threshold; the edge score is the minimum of the two endpoint
scores plus the frozen bond score. This directly attacks the dangling
attachments. A scalar candidate score independent of the endpoint cannot
distinguish a candidate that is correct for one bond from the same-looking
candidate attached to a nearby false bond.

Do not add a coordinate offset head in the first trial: the fixed panel already
has 98.27% target-bond reachability at 12 px. Add a bounded offset branch only
if a CPU label-only separability audit shows that correct endpoint pairs are
reachable but fail because of a systematic 2--6 px displacement. This keeps
the 800-step shadow attributable to compatibility rather than adding two new
degrees of freedom at once.

## Sidecar isolation

The sidecar consumes only target-independent values at inference:

* r5 vertex candidate coordinates, score, role/aromatic flags and sampled
  frozen vision feature;
* frozen bond endpoint coordinates, bond score/type probabilities, and frozen
  bond feature;
* relative displacement, distance, and local bond direction.

It outputs an endpoint compatibility logit (and only in a separately justified
follow-up, a bounded 2-D endpoint offset).
The r5 vertex head, frozen bond head, production selector, set head, graph
assembler and checkpoint remain unchanged.  The sidecar is evaluated only in
memory; no candidate target IDs or GT nearest assignments are available during
candidate generation or inference.  Training labels are used only inside the
train split loss, after fixed target-independent proposals have been produced.

The endpoint assignment remains deterministic: one candidate per endpoint,
same candidate may be reused across different bonds at a junction, never on
both endpoints of one bond.  A complete bond requires two distinct accepted
endpoints after the bounded offset.

## Fixed budget and holdout gates

Use one run only: at most 800 steps, at most 300 seconds GPU wall time, fixed
batch size 8, no sweep, no restart, and no checkpoint promotion. Freeze all
inputs; train only the small compatibility MLP. An offset branch requires a
newly reviewed protocol and is not part of this budget.

The holdout must satisfy every condition:

* correct attached-endpoint precision >= 0.85;
* complete two-endpoint bond precision >= 0.80;
* bond recovery recall >= 0.50;
* visible O and N incident-bond recovery >= 0.50 each;
* implicit-C incident-bond recovery >= 0.70;
* conditional bond-type accuracy >= 0.75;
* dangling endpoint rate <= 0.05 and same-bond candidate reuse conflicts <= 0.05;
* target-bond recovery does not regress by more than 0.05 from the
  new-candidate frozen baseline (the current holdout baseline is 0.944, so the
  sidecar must retain at least 0.894, in addition to the absolute 0.50 floor);
* the sidecar may reduce raw attachment volume, but it may not pass by keeping
  only a tiny high-precision subset; report accepted endpoints, complete edges,
  and recovered target bonds with fixed denominators;
* no production state/checkpoint mutation, zero identity overlap, and
  `training_admissible=false`.

If the reachability panel fails, or if the sidecar reaches the gates only with
oracle anchors, the route is `NO-GO`.  A candidate-score-only head may be
considered later solely as a precision calibration layer after endpoint
geometry is proven reachable; it is not a repair for dangling geometry.
