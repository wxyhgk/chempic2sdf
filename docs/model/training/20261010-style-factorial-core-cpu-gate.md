# Style-factorial core CPU gate (2026-10-10)

This pass only audits the fixed proposal and label contract. It does not read
sealed data, open raster shards, modify the frozen checkpoint, or authorize a
GPU run.

The core stream is three identity-disjoint styles (`original`, `rdkit`, and
`coordgen`): 1,600 train molecules and 400 dev molecules, four samples per
molecule in the source factorial pool. `stress` remains a 400-molecule dev
diagnostic and is not mixed into the first adaptation.

The frozen-initial endpoint replay is a prefix check, not selector accuracy:

| style | raw recall | selected recall | selector drop |
|---|---:|---:|---:|
| original | 0.9962 | 0.9882 | 0.0081 |
| rdkit | 0.9956 | 0.9880 | 0.0077 |
| coordgen | 0.9964 | 0.9897 | 0.0068 |
| stress (diagnostic) | 0.9883 | 0.9307 | 0.0589 |

The core prefix check passes the provisional 0.98 selected-recall floor. The
stress gap is large enough that it must remain isolated.

## Fixed strict-label contract

The later CPU candidate replay must publish per-style strata using candidate
and rendered-atom coordinates in the 768 px frame:

- `strict_positive`: explicit strict member, nearest distance ≤6 px;
- `confirmed_negative`: explicit unmatched member, nearest distance >10 px;
- `near_unknown`: explicit unmatched member, nearest distance ≤10 px;
- unknown members inside strict-positive groups: neither class.

Only strict positives and confirmed far negatives enter the selector loss.
`near_unknown` has loss weight zero and is reported as a conservative false
selection. Protected/matched members are not loss rows. Assignment is
one-to-one Hungarian for auditing; no ground-truth candidate is injected.

The predeclared selector floors are strict conservative precision ≥0.85 and
recall ≥0.50 **in every core style**, plus explicit near-unknown and
hard-negative counts. Endpoint reachability cannot substitute for this gate.

## Decision

The static CPU gate passes manifest shape, identity separation, core prefix
reachability, and stress isolation. It intentionally remains **not admissible**
until a frozen-initial candidate replay supplies the strict/near-unknown/
hard-negative strata and a CPU visual gate passes all three core styles. The
existing 480-image crop probe has holdout precision 0.926 but recall 0.262,
so it does not satisfy the gate and must not trigger GPU training.

Artifact:

`outputs/audits/chemvector-style-factorial-core-cpu-gate-20261010-r1/report.json`
