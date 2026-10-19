# r17 presence threshold calibration (CPU)

After the matched-source presence residual failed the precision gate at
threshold 0.95, a bounded train-only sweep over 0.80–0.99 was run. The
selection criterion required every train profile to lose at most 1pp edge and
typed precision relative to the 0.95 baseline, then maximized aggregate train
edge/typed recall.

The selected threshold was **0.97**. On the disjoint 16-molecule holdout, all
eight profile deltas were exactly zero: edge precision, edge recall, typed
precision, and typed recall were unchanged from threshold 0.95. Thus the
threshold moves a few train-side candidates but provides no cross-identity
recall gain. `production_go=false`; no type head or GPU run was started.

The extraction cache is retained for future read-only probes:

`outputs/audits/chemvector-r17-presence-threshold-calibration-cpu-20261022-v1/feature-cache.pt`

Authoritative report:

`outputs/audits/chemvector-r17-presence-threshold-calibration-cpu-20261022-v1/report.json`
