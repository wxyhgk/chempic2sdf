# 2026-10-06 positive-only chemical replay probe

After the multi-agent review, the joint objective was corrected before the
next GPU run:

- chemical replay is applied only to confirmed positive pairs;
- confirmed non-edges still supervise presence only;
- unknown pairs remain excluded from both heads;
- asymmetric aromatic-anchor tests now cover positive-only masking and
  directionality.

The 10-step in-memory shadow passed with finite gradients, unchanged frozen
parameters, and unchanged presence-threshold masks. Its report distinguishes
virtual in-memory updates from persisted optimizer steps:

`outputs/audits/chemvector-joint-edge-shadow-20261006/report.json`

## Bounded GPU result

One 100-attempt trial was run from the unchanged initial checkpoint. Only the
1,285 parameters in the two pair output heads were trainable; all vision,
atom/selector, shared pair trunk, OCR, role, and bbox parameters were frozen.
The run used the fixed 1,200-image order and the existing 944-image public
development cohort. It did not read sealed-test content and did not modify or
deploy the initial checkpoint.

| cohort | initial exact | after 100 | initial valid | after 100 | fallback |
|---|---:|---:|---:|---:|---:|
| full molecule (800) | 170 | 169 | 552 | 551 | 51 → 51 |
| curriculum (144) | 63 | 64 | 138 | 138 | 0 → 0 |

The hard gate stopped the trial at step 100. There was one broken exact case,
`fresh_dev_001966_coordgen`, and no new fallback. The same edge-level cause
reported in the previous attribution remains: a single false edge crosses the
presence threshold and gives an `OH` oxygen an impossible valence. The
positive-only replay correction therefore did not make this recipe safe.

Result: **NO-GO; keep initial**. Do not extend this run, unfreeze the shared
trunk, or tune the threshold.

## What changes next

The evidence now separates two independent workstreams:

1. Add and validate a deterministic valence-safe edge acceptance/repair layer
   on the public development cohort. This is a decoder safety experiment, not
   a new training label.
2. For learning, move upstream to endpoint coverage. Across the fixed 1,200
   training images, the missing target atoms are concentrated in `font_large`
   and `serif`, especially visible O/N text and implicit-C peaks. A selector
   retention loss that simply raises all candidates is rejected by the earlier
   regression; the next selector trial must distinguish raw-candidate absence
   from selected-candidate rejection and use hard negatives plus retention.

Until those two diagnostics pass independently, another pair/chemical GPU
trial is not justified.
