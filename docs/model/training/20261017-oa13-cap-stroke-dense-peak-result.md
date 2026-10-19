# oa13 cap/stroke dense-peak diagnostic

The raw-gap targets are not missing all visual evidence.  At the production
6 px matcher radius, 76.7--77.9% of visible-text targets have an active raw
proposal nearby, while implicit-carbon targets are essentially fully covered.
The remaining visible-text targets have a nearest dense peak displaced by
roughly 1--2 px relative to matched atoms; many are in the top-128 list but
outside the production assignment radius.

This rules out a simple threshold or bond-rendering fix.  The remaining
bottleneck is the dense cell/peak objective and collision ownership between
text glyph response and the attachment anchor.  Existing heatmap-only and
offset-only GPU trials are already terminal NO-GO and must not be retried.

Next step is a CPU/forward-only admission audit for a multi-slot cell/peak
contract with protected old peaks.  It must establish positive count,
hard-negative precision, and 100% protected/selected retention before any new
GPU shadow is considered.  No production training is authorized by this
diagnostic.

Artifact:

`outputs/audits/chemvector-oa13-cap-stroke-pixel-geometry-20261017-r3/report.json`
