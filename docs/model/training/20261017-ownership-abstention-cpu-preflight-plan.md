# Ownership/geometry abstention: CPU-only preflight

## Decision

Do not train an ownership or image-only abstention head yet.  Abstention can
only remove candidates from the frozen selected set; it cannot recover a
strict-positive candidate that the baseline did not select.  In the existing
candidate cache, non-protected selected strict recall is only 5.6%, 5.5%, and
6.0% on the three core train variants, and 3.4%, 7.6%, and 6.9% on the three
core dev variants.  A policy that does not add candidates therefore has an
absolute recall ceiling below 0.08 on this protocol.  Any apparent precision
gain from abstaining everything is a degenerate result.

## Strict CPU preflight (if the route is audited once)

1. Consume only the frozen initial candidate cache and public train-core,
   dev-core, and stress manifests.  Never read sealed identities, checkpoint
   gradients, or target coordinates as features.
2. Treat the existing selected baseline as an immutable prefix.  Protected
   candidates must be retained exactly (`protected_retention = 1.0`).  No new
   candidate may be added.
3. For each non-protected selected candidate, compute deterministic image-only
   evidence: local ink occupancy/connected components, directional line
   continuation around the anchor, nearest competing candidate distance, and
   a unique ownership flag.  Text element, target atom distance, target bbox,
   and graph labels are evaluation-only and must not enter the feature vector.
4. Abstain if ownership is ambiguous, line/text evidence is contradictory, or
   the locked calibration score is below threshold.  Abstention is not a
   confirmed negative and must not be used to train a selector.
5. Lock thresholds once from a fixed scaffold-disjoint `train_core`
   calibration slice.  No threshold sweep or feature choice may inspect
   `dev_core` or stress.  Replay each dev-core variant independently and report
   stress separately.

## Gates

The preflight is **PASS** only if every dev-core variant satisfies all of:

* protected retention = 1.0;
* strict-positive precision >= 0.85;
* strict-positive recall >= 0.50 (measured against all strict positives in the
  raw candidate cache, preventing all-abstain solutions);
* hard-negative selected = 0;
* selected count is nonzero and at least 10% of the frozen baseline selected
  non-protected candidates.

Stress is reported but cannot rescue a failed core gate.  A failure is **NO-GO**
and forbids GPU training or production integration.

## Current evidence

The earlier group-abstention CPU preflight already failed the core gates:
dev precision was 0.59/0.59/0.73 and strict recall 0.19/0.24/0.33.  The raw
objectness preflight did not establish strict rescue.  The current cap/stroke
panel has only ten difficult identities and its complete cap52 comparison
still misses visible O/N atoms; it is not a training corpus or evidence of a
learnable ownership signal.

Therefore the recommended action is to stop at this CPU contract audit and
investigate candidate generation/text glyph representation.  Do not launch
an ownership/abstention training run.

The ceiling audit is reproducible at
`outputs/audits/chemvector-ownership-abstention-ceiling-20261017/report.json`:
dev-core ceilings are 3.41%, 7.62%, and 6.86% for coordgen, original, and
rdkit respectively, so the route is NO-GO before any feature fitting.
