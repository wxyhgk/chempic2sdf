# v0.2 frozen-initial validation replay (2026-10-06)

This was a forward-only replay of the fixed 2,619-row validation proposal from
`curriculum-v02-release-preparation-20260919`. The source bundles remained
quarantine records and were loaded through the diagnostic path. The immutable
production checkpoint was checked against SHA-256
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`.

Report: `outputs/audits/chemvector-curriculum-v02-validation-endpoint-replay-20261006/report.json`.

| quantity | count |
| --- | ---: |
| validation images | 2,619 |
| target atoms | 19,926 |
| raw matched | 19,724 (98.985%) |
| raw gaps | 202 |
| selected matched | 19,292 (96.817%) |
| selector drops after raw match | 432 |

The raw field is therefore mostly reachable on this panel; the dominant loss
is downstream selector retention. The largest style strata are `font_large`
(117 drops) and `serif` (113). By incident bond type, `TRIPLE` has 259 drops
and `SINGLE` 334; `AROMATIC` has only 2. Visible iodine is also a distinct raw
gap (60 of 156 endpoints).

Decision: no full v0.2 training run is admitted from this replay. The result
supports a small, identity-disjoint diagnostic/rendering slice focused on
`font_large`/`serif`, visible I/N/O, iodine, and triple-bond incident endpoints.
It does not justify another generic image expansion, selector threshold sweep,
or a model-side sidecar. The replay performed zero optimizer steps, did not
read sealed data, and left the initial model state unchanged.
