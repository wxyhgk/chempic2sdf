# r17 known-clean pair-incidence shadow (2026-10-08)

The per-profile replay audit showed that selector retention cannot repair
`oa13` and `oa33`: atom recall was already .997 and .991, while strict typed
edge recall remained .831 and .722.  A bounded CPU shadow therefore trained a
zero-initialized 262→32→1 residual over the frozen AtomPair features.  It did
not add atoms or edges; it only rescored the existing unordered candidate
pair set.  The cohort used 32 identity-clean train molecules and 16 disjoint
holdout molecules (128 holdout images), with 600 deterministic CPU steps.

| metric | baseline | pair residual |
|---|---:|---:|
| edge recall | .9432 | .9964 |
| typed edge recall | .9154 | .9613 |
| edge precision | .9811 | .9717 |
| typed edge precision | .9522 | .9375 |

The difficult profiles confirm the missing-edge diagnosis but fail the strict
precision gate:

| profile | typed recall | typed precision |
|---|---:|---:|
| oa13 | .8125 → .9279 | .9235 → .8935 |
| oa33 | .8079 → .9704 | .9647 → .9336 |

The route is **NO-GO for integration as trained**.  The next bounded probe,
if pursued, should be train-only precision calibration or a trust-region
residual (fixed threshold and residual cap), preferably routed only to oa13/
oa33, with profile gates requiring typed precision drop ≤.01, typed recall
gain >0, and no sanitize regression.  Do not start GPU training from this
shadow.

Artifact:

```text
outputs/audits/chemvector-r17-pair-incidence-known-clean-cpu-v1/report.json
```
