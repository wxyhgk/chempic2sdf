# 2026-10-05 edge-label reachability audit

This is a read-only replay of the exact 1,200 training rows bound to the
chemical-head preparation. It uses the zero-step `initial.pt` checkpoint and
the current atom matching and pair supervision contract. No optimizer step was
run, no production profile was changed, and the sealed test set was not read.

Report: `outputs/audits/chemvector-edge-label-reachability-20261005/report.json`.

## Result

| quantity | count |
|---|---:|
| candidate pairs | 636,354 |
| known pairs (both endpoints matched) | 622,702 |
| unknown pairs (at least one endpoint unmatched) | 13,652 |
| confirmed positive pairs | 38,583 |
| confirmed non-edge pairs | 584,119 |
| sampled rows under current 6:1 / minimum-48 sampler | 268,111 |
| sampled confirmed non-edges | 229,528 |
| unreachable target bonds | 793 |

The known-pair rate is 97.85%; the positive rate among known pairs is 6.20%.
The 793 unreachable bonds are not chemical-class ambiguity: at least one
target endpoint was not matched by the frozen atom stage, so no pair-level
chemical loss can reach them.

## Chemical-class reachability

All four bond classes are present in confirmed positive supervision:

| class | confirmed positive rows |
|---|---:|
| SINGLE | 22,959 |
| DOUBLE | 2,787 |
| TRIPLE | 58 |
| AROMATIC | 12,779 |

The sampler retains every positive row, so these counts are identical in the
sampled positive set. Therefore the immediate blocker is not that aromatic or
triple labels are absent from the 1,200-image plan.

## What the current objective actually teaches

`chemical_head_loss` applies cross-entropy only to confirmed positive rows.
Confirmed non-edges are available to the pair/presence contract, but this
chemical-head-only stage does not use them as chemical negatives. With a
non-zero trust-region option, sampled rows can additionally receive initial
logit KL, but that is a stability constraint rather than a new chemical label.

The next experiment should therefore not be another scalar trust-weight sweep.
Before any new training authorization, prepare a joint, explicitly masked
objective with:

1. presence supervision on confirmed non-edges only;
2. chemical CE on confirmed positive rows only;
3. an asymmetric initial-aromatic floor/anchor for already-stable aromatic
   edges; and
4. replay checks for the 793 unreachable target bonds and the 13,652 unknown
   pairs, which must remain excluded rather than being converted into negatives.

The fixed 944-image development gate remains mandatory: no full exact or valid
regression, no new fallback, and no curriculum exact regression before any
checkpoint can be considered.
