# r17 calibrated pair → graph integration v2 (2026-10-08)

This is the first strict identity-clean downstream integration of the pair
presence sidecar.  The vision trunk, atom set, unordered pair set, element/type
heads, and production checkpoint stayed frozen.  Only calibrated pair
admission changed; each proposed edge was accepted only if the safe compiler
continued to sanitize.

On the 32-image oa13/oa33 holdout:

- edge precision/recall: `.995902/.906716 → .995910/.908582`
- typed precision/recall: `.879098/.800373 → .887526/.809701`
- sanitize: `.875 → .875`
- exact typed topology: `.1875 → .1875`
- oa33 typed recall: `+.01866`; oa13 unchanged

All strict precision/recall/sanitize gates pass, while `production_go` remains
false.  The route accepted only one new holdout edge, so this is a bounded
shadow candidate, not evidence for GPU training or deployment.  Provenance is
bound to the targeted cohort, pair state, calibration report, and frozen
checkpoint; target fields are used only for post-hoc scoring.
