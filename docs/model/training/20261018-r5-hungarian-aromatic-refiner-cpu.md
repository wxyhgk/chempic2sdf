# r5 Hungarian aromatic refiner CPU preflight

The cycle-only demotion probe improved full-holdout oracle sanitization to
`0.9219`, so a learned aromatic refiner was tested before spending another
GPU shadow.  Ownership is fixed to the train-selected Hungarian policy
(`radius=24 px`, score regularizer `lambda=8 px`); edge presence and bond
type logits are frozen.  The CPU MLP receives only frozen edge geometry/logits,
type probabilities, and target-free cycle/endpoint-degree features.  Labels
are joined after feature construction and are never written to the sidecar.
The refiner can only demote an AROMATIC prediction to SINGLE.

Receipt:

```text
outputs/audits/chemvector-identity-aligned-line-first-hungarian-aromatic-refiner-r5-cpu-v2/report.json
```

The 500-step CPU fit uses 160 train and 80 holdout images; threshold is
train-only.  The 80-image holdout contains no aromatic ground-truth bonds, so
it is not a meaningful aromatic generalization gate.  The fixed train-only
state replayed over all 474 holdout images gives:

- edge precision/recall unchanged at `0.9585 / 0.8503`;
- baseline type accuracy `0.9716`, refined `0.9735`;
- baseline oracle sanitization `0.7911`, refined `0.7996`;
- only 9 selected edges demoted; cycle-only demotion remains much stronger
  (`0.9219` sanitize).

The CPU gate is `NO-GO` for a GPU refiner (`full sanitize < 0.90`).  This is
evidence that the cycle/degree features do not generalize enough to replace
the deterministic cycle critic; do not launch another GPU run or alter
production.  If aromatic work resumes, enlarge the targeted train strata and
fix profile/style coverage first.
