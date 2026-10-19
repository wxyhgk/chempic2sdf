# cap/stroke pixel-geometry diagnostic (2026-10-17)

This is a read-only replay of the six-factor cap/stroke quarantine panel.  It
uses the frozen initial checkpoint (`09d166...88dbe1`), with no optimizer,
checkpoint write, OCR, or sealed-test read.  The 960-row panel rendered 959
usable images; one `cov33_cap28` row was rejected by the renderer's typography
clearance rule and was not silently substituted.

Machine-readable output:
`outputs/audits/chemvector-oa13-oa33-cap-stroke-pixel-geometry-diagnostic-20261017-r1/report.json`.

## What was measured

For every visible O/N target, the audit stores the same-source glyph bbox and
raster ink fraction, target atom anchor, text-center/anchor distance, frozen
heatmap score/rank at the anchor cell, nearest local heatmap peak distance and
rank, and nearest dense predicted-anchor distance/rank.  Target geometry is
used only for post-hoc labels and aggregation; it is never supplied to the
model or decoder.

## Result

The glyph pixels themselves are not disappearing.  Typical O/N glyph boxes
are stable within each font family (cov13: O 24×23 px, N 26×23 px;
cov33: O 21×21 px, N 23×21 px) and matched/gap ink fractions are effectively
identical.  The separation is geometric:

| profile family / element | raw matched median nearest-peak distance | raw gap median nearest-peak distance | raw matched median anchor-cell score | raw gap median anchor-cell score |
|---|---:|---:|---:|---:|
| cov13 O | 5.2–5.4 px | 6.3–6.4 px | 0.009–0.010 | 0.007–0.010 |
| cov13 N | 4.6–4.8 px | 6.7–6.8 px | 0.046–0.059 | 0.037–0.054 |
| cov33 O | 5.6–5.7 px | 6.3–6.4 px | 0.0027–0.0031 | 0.0029–0.0035 |
| cov33 N | 4.5–4.7 px | 6.4–6.7 px | 0.023–0.026 | 0.007–0.016 |

The nearest local peak is usually present, but it is displaced by roughly
1.0–2.2 px more for gaps than for matched atoms.  For N, the gap's anchor-cell
score is sometimes not lower than matched (the model responds to nearby ink),
yet the predicted peak is still about 6.5–6.8 px away.  For O, especially
cov33, anchor-cell scores are near background for both matched and gap atoms;
the decisive signal is peak-to-anchor displacement rather than glyph darkness.

## Interpretation

This rules out “the raster glyph is absent” and weakens a pure objectness
threshold explanation.  The current atom field is responding to the visible
glyph/stroke region, but its local maximum is not consistently at the chemical
attachment anchor.  The error is therefore a coupled **anchor geometry +
heatmap target** problem:

* A text-center or raw-ink detector alone cannot safely create an atom; it
  would reproduce the displaced peak and risk line fragments.
* Lowering the heatmap threshold is unlikely to solve O gaps: cov33 O anchor
  scores are background-like even among raw-matched atoms.
* The useful next experiment is a target-preserving geometry diagnostic or
  small frozen-feature admission study that explicitly measures the
  text-center-to-attachment offset and keeps baseline candidates protected.
  It must be evaluated separately on cov13/cov33, O/N, serif, and cap/stroke
  controls.  No production training or decoder change is justified by this
  replay alone.

