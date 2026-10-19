# Style-factorial stress isolation audit (2026-10-10)

This is a read-only audit.  It joins the identity-bound style manifests and
the already completed frozen-initial endpoint replay.  It does not read image
payloads, checkpoints, or sealed data, and it performs no training or GPU
work.

## Evidence

The dev proposal has 1,200 core samples (`original`, `rdkit`, `coordgen`) and
400 stress samples, with no sample overlap.  Train/dev molecule overlap is
zero.  Stress is not just another font style: its generator enables
`crossed_bonds`, `crowded_subgraph`, `long_bond`, and `near_nonbonded` layout
operations (dev counts 103/92/111/94 respectively), uses font sizes 10--16
instead of the core 13--19 range, and samples multiple-bond offsets from
0.08--0.27 instead of 0.12--0.20.

Frozen-initial replay, aggregated over atoms, shows the separation clearly:

| stratum | raw recall | selected recall | selector-drop / raw | selector drops |
|---|---:|---:|---:|---:|
| core (three variants) | 0.9961 | 0.9886 | 0.75% | 283 / 37,577 |
| stress | 0.9883 | 0.9307 | 5.89% | 732 / 12,428 |

The stress selector-drop rate is about 7.8x the core rate.  This is a
retention problem, not evidence that the stress images should be relabeled as
ordinary negatives.

## Decision

Core-only training is admissible as the first adaptation candidate, but only
after a CPU strict gate.  Stress remains an independent robustness holdout;
it must not be mixed into the first training stream or used as confirmed
negative supervision.  Any future stress adaptation needs its own objective
and a fresh identity-bound release.

Required gate before GPU authorization:

1. The old 480-image prefix has no selector-retention regression.
2. Every core variant separately reaches strict precision ≥0.85 and recall
   ≥0.50.
3. Stress is reported separately (raw recall, selected recall, operation
   buckets, and abstention/unknown counts); it is not averaged away into the
   core score.
4. The candidate/unknown contract remains unchanged; stress misses are not
   converted into confirmed negatives.

Artifact: `outputs/audits/chemvector-style-factorial-stress-isolation-20261007-r1/report.json`.
