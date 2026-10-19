# Known-clean selector retention sidecar (2026-10-08)

After the readiness review, one bounded CPU training shadow was run on a
quarantine cohort that excludes every molecule and known
canonical/connection/scaffold identity in the r17 specialist plan and all
materialized r17 feature caches. The cohort has 128 train molecules and 64
holdout molecules, eight profiles each (1,536 images). The historical prior-use
ledger is still incomplete, so this is not a formal production-training or
benchmark authorization.

Artifacts:

- cohort: `outputs/audits/chemvector-r17-selector-known-clean-shadow-cohort-v1/report.json`
- training: `outputs/audits/chemvector-r17-selector-retention-known-clean-cpu-v1/report.json`
- graph replay: `outputs/audits/chemvector-r17-selector-graph-replay-known-clean-v1/report.json`
- perturbation: `outputs/audits/chemvector-r17-selector-graph-replay-known-clean-v1-perturb/report.json`

## Training result

Only the selector-retention MLP was optimized for 1,200 CPU steps. Vision,
AtomObject, set head, AtomPair/bond, OCR, chemistry, and the production
checkpoint remained frozen.

- holdout candidate precision: `.86567`
- selected-prefix retention: `1.0`
- potential bond reachability: `7131 → 7225` (`+94`)
- production checkpoint SHA unchanged

## Frozen graph replay

| holdout metric | baseline | replay |
|---|---:|---:|
| edge recall | .93978 | .94959 |
| strict typed-edge recall | .90456 | .91423 |
| strict typed-edge precision | .95397 | .95292 |
| sanitize rate | .86719 | .86719 |

The holdout strata show why this remains shadow-only: `oa13` did not improve
(`.83094` typed recall in both arms), while `oa33` improved only
`.72155 → .73702` and its typed precision moved `.86951 → .86736`. The
aggregate gain therefore must not be read as a repair of the difficult serif
profiles.

The target-perturbation sentinel produced the same decision digest as the
normal run:

`5fc865e343cdbd40093f7bb167eccbd18aebe2095818f3988737ba567598643e`

Thus candidate selection, pair enumeration, and safe compiler acceptance do
not depend on target labels. The strict replay gates pass, but the decision is
still **shadow-only**: historical prior-use is incomplete, stress-profile
coverage remains unresolved, and no production graph artifact has been
authorized. The sidecar is not merged and no GPU checkpoint training follows
from this result.

The consolidated readiness receipt is
`outputs/audits/chemvector-training-readiness-20261008-r5/report.json`.
It records that this bounded CPU shadow completed its gates, while full GPU
training remains `NO-GO` because the stress-profile and historical-ledger
requirements are not satisfied.
