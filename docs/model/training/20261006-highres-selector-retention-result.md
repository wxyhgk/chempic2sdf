# High-resolution sidecar selector retention (2026-10-06)

The K=16 sidecar was appended to the existing 128-primary proposal set and
sent through the unchanged `ProposalObjectSetHead` as a read-only shadow.  No
checkpoint was written.  The sidecar was trained only in memory on 120 fixed
raw-gap images for 200 virtual steps; the remaining 1,080 images were holdout.

## Result

On holdout, appending 16 candidates changed the incidence softmax denominator
for the old candidates:

| metric | primary only | primary + sidecar |
|---|---:|---:|
| selected matched targets | 31,894 | 31,885 |
| old selected targets lost | 0 | **10** |
| newly selected targets | 0 | 1 |

The gate is **NO-GO**.  A raw candidate sidecar cannot be passed through the
existing selector unchanged; its global incidence normalization moves old
presence logits and causes regressions.

## Consequence

Do not increase the selector query count or append rescue candidates to the
existing set head.  The next viable policy must preserve the baseline selected
set as immutable and score/fill only unused capacity with an independent
sidecar selector.  That policy needs its own holdout audit before any training
or integration.

Machine report:
`outputs/audits/chemvector-highres-selector-retention-20261006-r1/report.json`.
