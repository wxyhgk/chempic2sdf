# Targeted 400-image focal-hard shadow (2026-10-06)

The fixed-cohort experiment identified a better high-resolution objective, so a
separate deterministic train-only pool was prepared from the admitted train
manifest.  The pool excludes the fixed 1,200-image evaluation cohort.  Its
400-row identity manifest is nested in the 800-row pool and has hash
`4d3f9fcec3bf8eb06cffb43ec50426acee2114b0cba40e938f7f073f087f1e42`.

Only the frozen production model and an in-memory 384×384 branch were used;
the branch ran 200 virtual steps with focal heatmap plus top-64 hard-negative
suppression (weight 0.25).  The fixed 1,200 images were evaluated afterward.
No scorer optimizer, selector integration, checkpoint, validation witness, or
sealed test was used.

## Field result

| rescue K | sidecar peak precision | newly reachable fixed-cohort gaps |
|---:|---:|---:|
| 8 | 96.3% | 20 |
| 16 | 93.0% | 43 |
| 32 | 76.5% | 81 |

At K=16 the primary-plus-sidecar union matched 36,117 targets and recovered 47
new gap targets on the fixed cohort.  This is a strong field-capacity signal,
but it is not a deployment result: the sidecar still has to survive locked
assignment and downstream graph gates.

## Scorer readiness

The same branch/candidate contract was passed to a read-only ranking audit.
The train side contained only 20 residual positives (AUPRC 0.371), while the
holdout contained 41 (AUPRC 0.449).  A tiny threshold happened to give 2/2 on
train, but that is statistically meaningless and is not accepted.  The strict
pre-registered gate is:

```text
train positives >= 50
train AUPRC >= 0.50
train precision >= 0.85
```

It fails on coverage and AUPRC, so `bounded_training_admissible=false`.

Reports:

- [field shadow report](/home/zhenghe/GK/Project/chempic2sdf/outputs/audits/chemvector-atom-highres-targeted-400-focal-hard-20261006-r2/report.json)
- [readiness report](/home/zhenghe/GK/Project/chempic2sdf/outputs/audits/chemvector-targeted-400-focal-hard-readiness-20261006-r2/report.json)
- [prominence audit](/home/zhenghe/GK/Project/chempic2sdf/outputs/audits/chemvector-targeted-400-prominence-readiness-20261006-r3/report.json)
- [pool preflight](/home/zhenghe/GK/Project/chempic2sdf/outputs/audits/chemvector-highres-targeted-pool-preflight-20261006-r3/report.json)

Decision: retain focal-hard as the leading field objective, but do not train
RescueScorer or alter the production selector.  The next admissible step is a
larger, still disjoint 800-row pool only if we explicitly need more residual
positive coverage; otherwise the current evidence is sufficient to focus on
candidate geometry/prominence rather than model capacity.

For completeness, a 200-step replay over the nested 800-row pool recovered 33
K=16 union gaps on the fixed cohort (K16 peak precision 94.3%).  It is not a
fair 800-image training comparison because the bounded branch budget is capped
at 200 steps; an attempted 800-step run was rejected by that safety cap.  We
therefore do not use the 800 result to authorize more training.

The follow-up prominence audit tested a 5×5 ring prominence feature and fixed
combinations with score, distance, and darkness. It did not improve ranking:
prominence AUPRC was 0.131 train / 0.151 holdout, versus raw score 0.371 / 0.449;
the fixed score+distance+prominence top-1 precision was 11.8% train / 14.2%
holdout. The strict gate remained false (20 positives, train AUPRC 0.371), so
this feature is rejected and no scorer training was started.

Finally, a residual-only field objective (positives = raw-unmatched target
cells, baseline atoms masked out) was shadowed on the same 400 pool. It raised
K=16 union recovery to 61, but its scorer-readiness replay produced no
train-sidecar candidates after the immutable baseline filter, even when the
diagnostic guard was relaxed. This is a candidate-contract/coverage failure,
not permission to bypass the baseline lock; the residual-only route is
therefore also NO-GO for scorer training.

The locked-assignment follow-up widened the diagnostic peak pool from 16 to
4,096 and tested both the 12px guard and a zero-distance diagnostic guard. It
still produced no train-sidecar candidate after residual-only branch training;
the readiness report records `reason=no_train_sidecar_candidates`. This rules
out “top-K was too small” as the explanation and keeps the production baseline
untouched.

The discrepancy is now understood: the field shadow's `union_new_gap` metric
counts any raw high-resolution peak that wins global Hungarian assignment,
including peaks within the baseline-selected neighborhood. The readiness
contract removes those baseline-near peaks before scoring and then locks the
baseline assignment. Under that stricter contract the residual branch has no
stable independent sidecar candidates; the larger union number must not be
interpreted as deployable candidate precision.

A second geometry audit added frozen-head consistency, fractional cell offset,
and local image gradient. None carried useful residual signal: holdout AUPRC
was 0.101 (consistency), 0.144 (fractional offset), and 0.109 (gradient), all
below raw score 0.449. The fixed score+distance+consistency+gradient ranking
also fell to 13.2% top-1 precision. These features are rejected as well.
