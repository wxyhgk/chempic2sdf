# r5 hybrid ownership merge shadow

## CPU admission

The target-free hybrid route keeps Hungarian score-distance ownership, merges
only provisional nodes that independently point to the same candidate within
20 px, then applies cycle demotion and the tau=0.4 valence guard.  The
train-selected CPU replay on the fixed 474-image holdout reached:

| route | edge P | edge R | oracle sanitize |
|---|---:|---:|---:|
| frozen raw cycle critic | 0.95852 | 0.85034 | 0.92194 |
| current Hungarian + guard | 0.96592 | 0.85003 | 0.94515 |
| hybrid T20 + guard | 0.96326 | 0.85127 | 0.95148 |

All six profile recall floors (>=0.75) and sanitize floors (>=0.90) passed.
The hybrid is therefore shadow-worthy relative to the frozen raw floor, but
its precision is explicitly lower than the current guard and it is not a
production route.

## Bounded GPU shadow

The isolated EdgeBondMLP shadow used the frozen 160-image train slice, the
same feature contract, a train-only recalibrated threshold, and a maximum of
800 steps with full-holdout checks every 200 steps.  It stopped at step 200
because it did not improve the hybrid baseline:

| | edge P | edge R | oracle sanitize |
|---|---:|---:|---:|
| hybrid baseline | 0.96326 | 0.85127 | 0.95148 |
| step 200 | 0.96015 | 0.84941 | 0.93460 |

The weak `cov13_stroke16_48` profile also fell below the 0.90 sanitize floor
(0.89873).  Decision: **NO-GO**.  The sidecar state is retained only as an
audit artifact; no production checkpoint or graph path was modified.

Report:

`outputs/audits/chemvector-identity-aligned-line-first-hybrid-merge-shadow-r5/report.json`

