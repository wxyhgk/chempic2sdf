# Graph-constrained rescue review

Historical endpoint and pairwise experiments were checked before opening a
new route:

- P13/P14 full candidate-set KEEP/MOVE context could not produce a safe
  threshold; adding local peaks and the full top-128 geometry did not order
  rescue over harmful moves.
- P15 frozen bond-endpoint features had OOF AUC lift `-0.001925` over the
  baseline. The explicit port sidecar collapsed on holdout (AUPRC `0.258`;
  full frozen bond embedding holdout AUPRC `0.0535`).
- The larger bond-port reachability audit found only `62.22%` of unmatched
  targets near a free predicted port; attached candidate precision was about
  `0.92%` and strict predicted-port/GT-bond compatibility was only 47 rows.
- Local pair residual training reached holdout precision only `0.8425` at its
  best train-selected point and caused graph-level regressions. Full pair
  training improved course images but reduced full-molecule exact from
  `170/800` to `167/800` and increased fallback.
- Public valence guarding was safe as a post-hoc veto (944-image replay:
  one rescue, zero breaks) but has no candidate-recall power.

## Only admissible next audit

If a graph-constrained rescue audit is opened, it must be zero-training and
fail closed. For each dropped candidate, use only frozen predicted outputs:

1. at least one predicted bond endpoint within the fixed port radius;
2. one-to-one endpoint occupancy;
3. an incident predicted bond to an already retained atom/candidate;
4. no isolated component and no valence/chemical guard violation.

Protected baseline candidates pass through unchanged. Near-unknown and unknown
rows are not training negatives. Ground-truth distance, target IDs, layers,
and rendered graph are labels for the audit only, never forward features.

The audit must report strict-only recovery ceiling, hard-negative survival,
protected retention, and per-style results. It is not worth training unless
strict recovery is at least 0.50 while hard-negative survival is at most
0.15. Historical reachability makes meeting this unlikely; no GPU or scorer
training follows from this audit.
