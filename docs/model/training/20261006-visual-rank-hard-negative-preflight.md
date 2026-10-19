# Visual-group hard-negative/rank-aware route: CPU preflight

The 480-image visual-existence shadow used a 609-parameter group BCE head.  It
kept the baseline prefix but failed the holdout precision gate.  The failed
head must not be retrained with a new threshold or more steps.

## Proposed route (not authorized)

Keep `initial.pt`, the vision/atom/bond/set path, OCR, and all protected
selected groups frozen.  A small member scorer may be considered only after a
new release admission.  For each strict-positive group (raw candidate exists,
selected candidate was rejected), compare its member score against at most the
eight highest frozen-score confirmed negatives from the same image.  Unknown
members are excluded.

With margin `m=0.5`, the isolated loss would be:

```text
L_rank = mean softplus(m - score_positive + score_hard_negative)
L_group = logsumexp(member_scores / tau)       # optional, weight <= 0.25
L = L_rank + lambda_group * L_group
```

The old selected prefix is never rescored.  A rescue policy would be
baseline-first with a maximum of 16 new groups per image; one train-only K (or
threshold) would be fixed before one holdout evaluation.

## CPU-only result

`tools/audit_targeted_selector_visual_pool_rank_preflight.py` reads only the
released train/holdout group caches.  It performs no model/checkpoint read,
GPU work, optimizer step, or sealed-test read.

| split | strict P | confirmed N | pairwise AUC | hard pairs | violation | median rank | top-8 | top-128 precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 359 | 24,896 | 0.8765 | 2,872 | 0.5992 | 6 | 0.6017 | 0.0625 |
| holdout | 286 | 12,895 | 0.8712 | 2,288 | 0.6215 | 7 | 0.5629 | 0.0625 |

The signal is rankable in aggregate, but the operational top-K precision is
far below `0.85`.  Therefore the preflight is **NO-GO** and no GPU training is
authorized.  A future release would need, at minimum, identity-disjoint
positive/hard-negative counts, holdout AUC `>=0.80`, top-128 precision `>=0.85`,
and baseline-prefix retention `100%` before requesting a single bounded
100-step/30-minute shadow.

Report:
`outputs/audits/chemvector-targeted-selector-visual-pool-rank-preflight-20261006-r2/report.json`
