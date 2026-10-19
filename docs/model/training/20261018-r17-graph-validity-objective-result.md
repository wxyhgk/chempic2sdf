# r17 self-supervised graph-validity objective (2026-10-18)

A CPU-only 2-layer MLP was trained to rank one-edge deletion variants by
compiler validity.  Labels came solely from the frozen compiler; target graph
fields and sealed test data were not read during fitting or routing.

The route was required to compile every accepted transaction and was compared
with the deterministic confidence-sorted rollback baseline:

| split | baseline sanitize | learned route |
|---|---:|---:|
| train | .847656 | .875000 |
| dev | .871094 | .949219 |
| fresh | .921875 | .953125 |

Removed-edge counts and abstention rates were effectively identical.  The
learned objective produced no measurable gain, so its GPU/merge gate is
**NO-GO** and the head is not retained as a production candidate.

Artifact:

```text
outputs/audits/chemvector-graph-compile-validity-cpu-r17-v2/report.json
```

Future exploration should add genuinely new local visual/edge evidence or
multi-edge transactions, not repeat this self-supervised objective.
