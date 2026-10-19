# Identity-aligned O/N residual data route (2026-10-17)

The current cap/stroke raw-gap manifest is not a safe residual-training set:
its 6--16 px rows are all unassigned under the frozen Hungarian match.  Treating
the nearest target as an offset label would be oracle assignment and can train
the sidecar toward the wrong atom.  The next data step therefore precedes any
model step.

## Generation contract

Render controlled O/N visible-text cases with a target anchor at least 48 px
from every other atom anchor.  Keep all six cap/stroke profiles and split by
molecule scaffold/canonical identity (style variants stay together).  During a
frozen initial replay, admit a candidate/target pair only if:

- candidate distance to its target is at most 16 px;
- every other target is more than 20 px away from the candidate;
- the target has exactly one raw candidate within 16 px;
- no other candidate/target ownership conflict exists.

Everything else is unknown and masked.  This is a bijective geometric filter,
not a forced Hungarian assignment.  Target IDs and distances are used only to
admit labels and are never model inputs.

## Minimum scale and CPU gate

The minimum useful release is 20 unique molecules per element/profile for train
and 10 for holdout: 240 and 120 positive pairs respectively, with at least ten
train and five holdout positives in each of the 12 O/N × profile strata.  Keep at
least 4 far hard negatives per train positive and 2 per holdout positive.

Before any GPU, a CPU-only candidate score + 2D offset preflight must verify:

- zero molecule overlap and zero ambiguous admitted pairs;
- train-selected threshold precision ≥0.90 and locked-holdout precision ≥0.85;
- at least 10 strict holdout recoveries, improving over step zero;
- protected baseline retention exactly 1.0 and baseline raw loss zero.

Failure of any item is terminal; no threshold or capacity sweep is allowed.

If all gates pass, one independent receipt may authorize a 100-step/300-second
in-memory shadow over existing raw slots only.  It cannot create proposals,
enter the selector, alter the production checkpoint, or claim graph recovery.

Machine-readable protocol:
`outputs/audits/chemvector-identity-aligned-visible-text-20261017-r1/protocol.json`.
