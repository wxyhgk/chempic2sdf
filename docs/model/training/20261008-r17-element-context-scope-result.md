# r17 element-context aggregation scope audit (2026-10-08)

The element-context head and the graph replay were using different graph
contracts.  During training, each node receives mean/max context from the
frozen pair rows whose presence probability is at least `0.72`.  The replay
compiler instead aggregated every unordered pair of active nodes.  This is not
an equivalent representation: on a 9-node image, the training context has at
most eight rows per node while the replay context can contain all 36 rows.

`tools/probe_r17_element_context_scope_cpu.py` ran a deterministic, read-only
paired probe on four molecules per split (16 images per split, all four
profiles).  The frozen model, element head, type head, aromatic head, pair
threshold, and compiler were otherwise unchanged.

| split | all-pair element accuracy | selected-edge accuracy | all-pair sanitize | selected-edge sanitize |
|---|---:|---:|---:|---:|
| train | .1125 | 1.0000 | .6875 | 1.0000 |
| dev | .0583 | .9910 | .8125 | .7500 |
| fresh | .2956 | .9748 | .7500 | .8750 |

The all-pair route emits mostly `Si`/`P` for ordinary C/N/O nodes, which is a
clear out-of-distribution context failure.  The selected-edge route restores
the trained feature contract and gives five invalid-to-valid train cases and
two fresh cases; it has one dev valid-to-invalid case in this small probe, so
it is not a release result.

The repair must be made at the compiler boundary: aggregate element context
from exactly the same frozen `presence >= PAIR_THRESHOLD` pair rows used to
train the head.  Nodes with no selected incident pair must fail closed (or
use a separately trained degree-zero representation); they must not receive
an untrained all-pair vector.  After this repair, rerun the full identity,
mapped/unmapped, bond-type, and compile-decision attribution audit.  This
probe does not authorize GPU training or production integration.

Artifact:

```text
outputs/audits/chemvector-r17-element-context-scope-cpu-20261008-v1/report.json
```
