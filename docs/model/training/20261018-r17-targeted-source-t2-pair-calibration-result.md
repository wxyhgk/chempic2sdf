# r17 targeted-source t2 pair calibration result (2026-10-18)

The t2 train-only calibration diagnostic completed on CPU from the cached
frozen features.  It did not read sealed test data, use GPU, modify the
production checkpoint, or change the frozen unordered pair set.

Artifacts:

- sidecar: `outputs/audits/chemvector-r17-targeted-source-pair-incidence-cpu-20261018-t2/sidecar-state.pt`
- feature cache: `outputs/audits/chemvector-r17-targeted-source-pair-calibration-cache-20261018-t2.pt`
- report: `outputs/audits/chemvector-r17-targeted-source-pair-calibration-cpu-20261018-t2/report.json`

The global calibrated route selected no additions.  The metadata-only
profile route retained the baseline in most profiles and used thresholds
`oa02=.73`, `oa11=.73`, `oa13=.74`; it produced only a `+0.243 pp` oa33 typed
recall gain and no oa13 typed-recall gain.  Its oa33 typed precision change
was `-0.452 pp`, within the 1 pp safety limit.

Therefore the calibration safety checks pass, but the release objective does
not: both oa13 and oa33 must gain at least 0.5 pp typed recall.  The t2 route
is **NO-GO** and remains a shadow artifact.  Do not run safe graph integration
for release, start GPU training, or tune thresholds on holdout data.

The next valid experiment is a new append-only source with the strengthened
rare-stratum quotas on both train and holdout, followed by one bounded 600-step
CPU sidecar and this calibration protocol.
