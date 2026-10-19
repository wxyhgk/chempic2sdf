# r17 t2 candidate-union and residual-route decision (2026-10-18)

After the type-readout shadow failed to produce an admissible route, two
CPU-only diagnostics tested the next hypothesis: rank below-threshold pair
candidates with frozen visual features and add a small residual quota.

Artifacts:

- `outputs/audits/chemvector-r17-candidate-union-visual-probe-cpu-20261018-t2-r4/report.json`
- `outputs/audits/chemvector-r17-residual-rank-quota-shadow-cpu-20261018-t2/report.json`
- `outputs/audits/chemvector-r17-raw-pair-union-cpu-20261018-t2-r2/report.json`

The visual hidden representation is genuinely informative: holdout residual
ranking AUC is about `0.9995`, and oa13/oa33 residual positives rank cleanly.
That is a capacity result, not a deployable route.  The fixed pair candidate
union reaches only `3256 / 3368 = 96.67%` of mapped holdout target edges.

The train-only quota sweep (`q = 0..3` additions/profile/image) selected
`q = 0` for every profile under the one-percent total edge-precision safety
floor.  Holdout therefore stayed exactly at baseline (`edge P=.997517,
edge R=.987101, typed P=.921788, typed R=.912162`); oa13 and oa33 typed-recall
gains were both false.

The raw-proposal union audit also rejects a direct raw merge.  On the targeted
slice, raw proposals added hundreds of distractor pair rows (up to `+526` for
oa33) but produced zero additional threshold-passing true edges; only one or
two further edges were geometrically reachable before presence scoring.

Decision: **NO-GO** for raw candidate union, global residual quota, and further
presence-threshold tuning.  Keep the production checkpoint frozen.  The next
useful experiment must be a jointly trained/read out endpoint-conditioned
pair+type scorer (or new targeted data that increases residual positive
density), with a residual-only precision contract.  It remains CPU-shadow
first; no GPU or production integration is authorized by these results.
