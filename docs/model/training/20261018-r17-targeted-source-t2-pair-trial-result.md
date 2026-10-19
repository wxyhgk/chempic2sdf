# r17 targeted source t2 pair-incidence trial

日期：2026-10-18

## Source gates

The fresh source is:

`outputs/datasets/chemvector-r17-targeted-source-20261018-t2/`

It contains 64 train and 32 holdout molecules, 8 style profiles, and 768
images.  The rendered source satisfies all seven molecule-level quotas on both
splits: visible O/N aromatic, visible O/N double, visible O single, N triple,
and graph triple.  Metadata admission reports zero overlap on molecule,
canonical, connection, and scaffold identity; ledger, image hashes, and split
disjointness pass.

## Bounded CPU trial

The frozen pair-incidence residual was trained for exactly 600 CPU steps using
the 64/32 molecule cohort:

`outputs/audits/chemvector-r17-targeted-source-pair-incidence-cpu-20261018-t2/report.json`

The frozen checkpoint and unordered pair set were unchanged.  Holdout edge
recall improved from `0.98710` to `0.99877`; typed recall improved from
`0.91216` to `0.91677`.  The difficult profiles improved in typed recall:

- oa13: `+0.00482`
- oa33: `+0.01703`

However, typed precision fell beyond the safety budget:

- oa13: `−0.01723`
- oa33: `−0.02369`

Therefore `each_profile_typed_precision_drop_le_0.01=false`.  This trial is a
diagnostic success for reachability but a release failure for precision.

## Decision

Do not run calibration or graph integration for this state, and do not start
GPU training.  The result indicates that the expanded source reaches the
missing oa13/oa33 edges, while the current residual objective admits too many
wrong bond types.  The next experiment must change the bounded CPU objective
(for example, add a train-only precision-constrained/type-consistency term or
an abstention margin) and be rerun on the same admitted cohort.  It must retain
the same 600-step limit and the same independent holdout gates; production
checkpoint and sealed test remain untouched.
