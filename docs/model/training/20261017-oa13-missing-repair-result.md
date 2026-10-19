# oa13 missing-row repair result

The existing quarantine batch has 1,446 accepted `oa13` rows and 88 quality
rejections.  A deterministic, read-only retry was run with the exact original
profile tuple (DejaVu Serif, 32 px, bond cap 28 px, stroke 3.2 px, rotation
180 degrees).

Result:

| retry status | count |
|---|---:|
| accepted | 0 |
| quality rejected | 88 |

All 88 retries returned the same quality reason as the original rejection:

* 83 would erase a bond segment during typography clearance;
* 3 have a degenerate auxiliary junction;
* 2 have text/geometry overlap outside the renderer contract.

This is deterministic renderer/profile behavior, not a missing-file problem.
The fixed-profile repair gate is therefore **NO-GO**.  No new rows were
persisted, no profile parameters were changed, no sealed/evaluation identities
were read, and no training was authorized.

The machine-readable audit is
`outputs/audits/chemvector-oa13-missing-repair-20261017/report.json`.

The next experiment must diagnose visual candidate reachability or isolate the
renderer factor causing the large-serif failure.  It must not claim that extra
oa13/oa33 samples repair the original 0.934/0.919 raw-reachability result.
