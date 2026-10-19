# Identity-aligned integration shadow protocol (2026-10-17)

This is a quarantine-only protocol. It does not promote a checkpoint, change
the production model, or read sealed evaluation data.

## Why the existing r5 shadow is not an integration result

The r5 vertex shadow trains a new `ChemVertexDetector` over a frozen feature
trunk. Its high holdout recall is useful evidence for an isolated head, but the
head is not wired through the production atom proposal decoder,
`structure_forward`, bond/set/OCR modules, or graph construction. The r5
baseline and shadow also use greedy one-to-one matching; that is a comparison
diagnostic, not the strict identity-ownership contract.

The frozen-initial raw replay provides the relevant production reachability
context: only 22 strict and 23 refinable O/N candidates were admitted among
3,222 visible O/N targets. The isolated-head result must not be described as a
production proposal improvement.

## Fixed run

Run exactly one in-memory branch on the reviewed r5 rows:

* fixed seed and fixed accepted train/holdout rows;
* maximum 800 optimizer steps or 300 seconds, whichever comes first;
* candidate vertex/atom head learning rate at most `2e-4`, final vision block
  at most `1e-6`;
* start from a deep-copied in-memory branch; production atom, bond, set, OCR,
  graph and decoder modules remain frozen;
* no selector, graph rescue, threshold sweep, checkpoint write, or sealed-test
  read.

The candidate must be evaluated through the same production
`structure_forward`/decoder path immediately after the shadow update. A new
standalone decoder is not an integration test.

## Required metrics

Report train and holdout separately, with fixed denominators, for all atoms,
visible O, visible N, implicit C, and every profile. For every cell report:

1. legacy greedy ≤6 px matching, for comparability with the r5 baseline;
2. strict ownership: a target is matched only when exactly one candidate is
   within the radius and that candidate is not within 20 px of another target;
3. ambiguous/unknown counts, candidate precision, recall and protected
   retention.

Strict ownership is the primary metric. Hungarian or nearest-target rescue is
forbidden.

## Promotion gate

The shadow remains `NO-GO` unless every condition holds on the fixed holdout:

* strict O and N recall improves over the frozen-initial baseline by at least
  3 percentage points in aggregate and in each nonempty profile cell;
* implicit-C recall decreases by no more than 0.5 percentage points;
* all-atom recall and precision do not regress against the same-decoder
  baseline;
* ambiguous strict-ownership rows are zero; protected baseline retention is
  exactly `1.0`;
* train/holdout molecule identity overlap is zero;
* initial checkpoint SHA and a **full** production-state digest (including the
  branch's originally trainable vision block) are unchanged after the branch is
  discarded; a digest that excludes updated parameters is insufficient;
* `training_admissible=false`, `deployment_authorized=false`, and
  `sealed_test_read=false`.

Any failed cell stops the route. No extra steps, threshold changes, data
resampling, or production checkpoint are allowed after failure.
