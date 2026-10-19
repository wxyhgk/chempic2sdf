# r9 direct AtomPair residual GPU shadow

This is one quarantined GPU shadow after the reviewed r9 deterministic ceiling
and 500-step CPU scorer. It does not alter the production checkpoint or graph
path. The runner is `tools/shadow_direct_graph_pair_scorer_gpu_r9.py`; the
receipt is `outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-gpu-shadow-r9/report.json`.

## Protocol

- Frozen production continuation bundle; SHA before/after:
  `09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`.
- 96 train images (24 per r6 profile), 95 molecule-disjoint fresh holdout
  images; proposal, AtomObject, DirectAtomPair, OCR, and production graph are
  frozen.
- Only `DirectPairResidual(262 -> 32 -> 1)` (8,973 parameters) is optimized on
  CUDA. Threshold calibration is train-only at every checkpoint.
- Maximum 800 steps / 1,800 seconds; complete fresh holdout is checked at
  steps 200, 400, 600, and 800. Sealed test is not read.

## Result

| metric | frozen baseline | GPU shadow (step 800) |
|---|---:|---:|
| edge precision | 0.994845 | 0.988693 |
| edge recall | 0.934625 | **0.952785** |
| dangling rate | 0.003851 | 0.005000 |
| conditional type accuracy | 0.963731 | 0.960610 |

Profile edge recall for the shadow is `0.990431 / 0.937799 / 0.959799 /
0.923444` (all above 0.75); profile dangling and type floors also pass. The
first checkpoint already strictly improved holdout recall (`0.956416` at step
200), and all four checks passed.

## Decision boundary

The run is a successful **shadow-only** experiment. It demonstrates that the
direct AtomPair residual can improve the fixed holdout while preserving the
reviewed precision, dangling, type, and profile floors. The sidecar state is
saved only under the GPU-shadow audit directory. It is not a production
checkpoint and must not be promoted without a separate integration audit,
chemistry/sanitize evaluation, and an explicit deployment decision.
