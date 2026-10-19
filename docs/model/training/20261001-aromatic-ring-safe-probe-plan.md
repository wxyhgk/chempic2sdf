# Aromatic-ring-safe chemical-head probe

## Current decision

The 2026-09-29 chemical-head-only probe is retained as a diagnostic and is not
deployable. It improved two exact development images, but changed five images
from no fallback to RDKit kekulization fallback. The fixed replay found three
ring-gate margin crossings and two single-edge aromatic changes.

The next probe therefore adds a graph-level safety term to the chemical loss.
The term is implemented in `chemvector/training/curriculum/aromatic_ring_objective.py`
and is opt-in through
`chemical_head_loss_with_aromatic_safety`. Existing chemical-head callers keep
the old objective because the default safety weight is zero.

## Objective

For each confirmed 5- or 6-member aromatic target ring in one training sample,
let `p_i` be the chemical head's softmax probability for `AROMATIC`. Add

```
relu(0.45 - mean_i(p_i))^2
  + relu(0.10 - min_i(p_i))^2
```

to the existing positive-pair chemical cross entropy, multiplied by an
explicit probe weight. The production decoder still uses its existing
`mean >= 0.40` and `minimum >= 0.05` gate. The extra headroom prevents the
trained head from sitting directly on either hard boundary.

Only known positive pair labels are used to construct the target ring graph.
The objective uses a bounded cycle basis for this graph; production inference
continues to use its existing chordless-cycle promotion path.
Unknown pairs, atom presence, pair presence, OCR, and atom assignment do not
enter this loss. If a complete target ring is not reachable in the selected
pair rows, the term is a differentiable zero and the reachability count is
recorded.

## Proposed bounded run

- Initial checkpoint: the same 2026-09-15 initial bundle.
- Trainable parameters: `pair.chemical.weight` and `pair.chemical.bias` only.
- Training images: the same fixed 1,200 train rows and the same 944-image
  development cohort.
- Device and precision: one GPU, FP32, no autocast, no TF32.
- Maximum accesses: 1,200; checkpoints at 400, 800, and 1,200.
- Safety weights: one fixed value selected before launch, with the value and
  objective digest bound into a new authorization receipt.
- Presence/selected-pair parity, frozen-state hash, and OCR parity remain hard
  stop conditions.

## Acceptance gates

At the final checkpoint:

1. Full exact must be higher than initial.
2. Full valid must not decrease.
3. Full fallback must not increase; every newly introduced fallback is an
   immediate NO-GO.
4. Curriculum exact may not decrease.
5. At least 90% of reachable target aromatic rings must finish above both
   safety floors (`mean >= 0.45`, `minimum >= 0.10`) on the training trace.
6. No non-chemical parameter, presence output, selected pair set, OCR result,
   or default inference profile may change outside the candidate checkpoint.

The candidate remains a research artifact even if it passes. The sealed test is
not read by this probe.

## Required preflight before GPU launch

The objective and integration tests must pass. A read-only preflight must also
report, on the fixed training plan, the number of reachable aromatic rings and
the fraction already below each safety floor. The preflight must not write a
checkpoint, alter the initial bundle, or authorize a run. Only after that report
is reviewed should a new one-shot training receipt be created.
