# High-resolution independent-fill shadow (2026-10-06)

After the direct append experiment failed, a second read-only shadow copied the
baseline selected candidates and used only unused capacity (up to 16 sidecar
peaks) for independent filling.  The sidecar was trained in memory on 120
images for 200 steps; 1,080 fixed images were holdout.  No checkpoint was
written and no sealed data was read.

## Holdout result

| metric | baseline selected | independent fill |
|---|---:|---:|
| matched selected targets | 31,894 | 31,893 |
| old selected targets lost | 0 | **3** |
| new selected targets | 0 | 2 |

The fill policy is therefore also **NO-GO**.  Although it avoids changing the
set-head incidence denominator, adding a rescue candidate can still alter the
Hungarian target assignment and does not produce a net gain under the strict
old-target retention gate.

The high-resolution branch remains a capacity signal only.  It must not be
connected to the deployed graph path until a dedicated capacity-aware selector
is trained and passes old selected/valid/exact retention.

Machine report:
`outputs/audits/chemvector-highres-selector-retention-20261006-r2/report.json`.
