# Existing expanded-style coverage audit (2026-10-06)

The CPU-only audit inspected layer/metadata contracts in the existing
`curriculum-v0.2-expanded-20260919` quarantine preview.  It did not load the
model, open sealed identities, or read raster pixels.

The bundle contains 20,904 valid preview images from 3,484 molecules, with
3,484 images each for `font_large` and `serif`.  Per style, target atoms and
incident chemical bonds are:

| target | atoms | SINGLE | DOUBLE | TRIPLE | AROMATIC |
|---|---:|---:|---:|---:|---:|
| visible O | 3,447 | 2,796 | 1,386 | 0 | 66 |
| visible N | 1,841 | 2,011 | 60 | 264 | 552 |
| implicit C | 21,411 | 30,032 | 1,782 | 995 | 10,882 |

The same counts occur in both target styles.  This is enough broad coverage
for a frozen replay; the sparse cells are O-aromatic and N-double.  O-triple
is absent because it is not represented by the current valid chemistry
catalog, so it should not be fabricated as a training target.

Decision: do not create another generic 3,600-image batch now.  The expanded
bundle remains quarantine-only (`training_authorized=false`,
`external_overlap_status=unchecked`), so the next gate is identity/release
admission followed by a read-only model replay.  Only if that replay shows a
real sparse endpoint cell should a small targeted render pilot be generated.

Machine report:
`outputs/audits/chemvector-targeted-style-coverage-20261006/report.json`.
