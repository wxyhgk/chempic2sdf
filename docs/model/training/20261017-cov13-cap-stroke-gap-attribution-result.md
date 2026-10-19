# cov13 cap/stroke gap attribution

A read-only frozen-initial probe compared the ten `cov13_rot0` failure
identities.  Geometry was bucketed by the nearest incident bond length.
All target atoms in this small panel were in the `>=40 px` bucket, so there
is no evidence here for a short-bond-only failure.

| variant | accepted renders | target atoms | raw gap | raw reachability |
|---|---:|---:|---:|---:|
| `cov13_base` (rotation 180°) | 10/10 | 88 | 11 | 87.5% |
| cap52, rotation 0° | 10/10 | 88 | 9 | 89.8% |
| stroke1.6, rotation 0° | 2/10 | 16 | 0 | not comparable (incomplete denominator) |

For the two complete variants, implicit C was fully matched (45/45 in both).
The remaining gaps were predominantly visible O/N labels:

* `cov13_base`: 10 visible O/N gaps, plus 1 other visible gap;
* cap52: 7 visible O/N gaps, plus 2 other visible gaps.

Thus increasing the bond cap repairs renderer completeness, but does not solve
the visual candidate gap.  The model-side failure remains concentrated in
visible O/N text even when all ten cap52 images render successfully.  Stroke
1.6 cannot be used as a comparison because eight of ten images are rejected by
the renderer.

Machine-readable output:
`outputs/audits/chemvector-cov13-cap-stroke-gap-attribution-20261017/report.json`.
