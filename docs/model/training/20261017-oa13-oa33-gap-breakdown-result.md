# oa13/oa33 raw-gap breakdown

The frozen initial model was replayed read-only on the existing 1,260-image
slice for each difficult profile.  No gradients, selector updates, checkpoint
writes, or sealed reads occurred.

| profile | target atoms | raw gap | raw reachability |
|---|---:|---:|---:|
| oa13 | 15,362 | 1,012 | 93.412% |
| oa33 | 15,362 | 1,241 | 91.922% |

The gap is overwhelmingly visible text, not implicit-carbon localization:

| profile | visible-text gaps | implicit-C gaps |
|---|---:|---:|
| oa13 | 972 (96.0%) | 40 (4.0%) |
| oa33 | 1,237 (99.7%) | 4 (0.3%) |

The largest buckets are:

* oa13: visible O degree-1 `344/1,107` gaps (31.1%), visible N degree-2
  `476/1,678` (28.4%), and visible O degree-2 `58/431` (13.5%).
* oa33: visible O degree-1 `587/1,107` (53.0%), visible O degree-2
  `187/431` (43.4%), and visible N degree-2 `336/1,678` (20.0%).

Implicit carbon degree-2/3/4 remains nearly fully reachable in both profiles.
This rules out “the whole atom detector cannot localize the large-font
rendering” as the primary explanation.  The first diagnosis target is the
large serif visible-glyph appearance and its text/geometry context, followed
by candidate decoding around O/N labels.  Extra oa13/oa33 images and selector
training are not justified by this result.

Machine-readable output:
`outputs/audits/chemvector-oa13-oa33-gap-breakdown-20261017/report.json`.
