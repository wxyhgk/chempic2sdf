# r17 pair-incidence calibration v3 (superseded)

**Superseded: do not use this report as evidence.** A provenance audit found
that the report's cohort and sidecar state were from different identity
cohorts (`known-clean` versus `slice80-disjoint`). The metrics below are kept
only as a record of the invalid run. A replacement must bind cohort, state,
feature-cache, and checkpoint SHA before it can be considered.

The uncalibrated pair residual improved recall but violated per-profile
precision.  v3 keeps the same frozen candidate set and sidecar transform, then
selects only profile thresholds on the train split under strict **zero** train
precision degradation.  Holdout is evaluated once.

On the identity-clean 128-image holdout:

- edge precision/recall: `.981144/.943202 -> .981250/.948640`
- typed precision/recall: `.952231/.915408 -> .951875/.920242`
- oa13 typed recall: `+0.96pp`
- oa33 typed recall: `+0.49pp`
- every profile edge and typed precision drop is within 1pp

Checkpoint SHA is unchanged, GPU is false, sealed test is unread, and
`production_go` remains false.  This is the first admissible CPU routing
candidate, but it still requires downstream graph compiler/sanitize/exact
topology integration before any GPU or production decision.
