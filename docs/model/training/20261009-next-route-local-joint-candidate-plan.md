# Next route: local atom+bond joint candidate audit

The port-conditioned endpoint scorer is closed as **NO-GO**. The explicit
feature ablation removed embedding memorization but still reached only
`0.261/0.258` train/holdout AUPRC and no precision-admissible threshold. The
frozen port route also reaches only about `62.8%` of unmatched holdout targets;
strict predicted-port/target-bond compatibility is only `47` holdout cases.

## One bounded next step

Do a forward-only audit of a local atom-pair/bond candidate representation on
the existing 1,200-image training-only cohort:

1. Keep `initial.pt`, selected atoms, and the current bond detector frozen.
2. Enumerate only local predicted atom pairs (distance/geometry bounded), not
   free endpoint claims.
3. Label a positive only when both selected atoms map to distinct target atoms
   and their target pair is the same `bond_atom_refs` identity. Unknown pairs
   remain excluded; confirmed non-edges are hard negatives.
4. Report candidate coverage, strict positives by bond type, per-image top-K
   precision, and baseline-prefix invariants before any optimizer step.
5. Train at most one small sidecar only if the forward audit has at least 50
   train positives, a non-trivial holdout positive count, and no ownership or
   split failure. Otherwise stop the route without training.

The existing current-initial replay already provides the useful scale check:
`38,583` strict positive pairs and `584,119` confirmed non-edge pairs across
1,200 images, with only `793` unreachable target bonds. This is materially
better supervision than nearest-atom port labels and is the only route worth a
single bounded follow-up under the GPU budget.

No production checkpoint, sealed test, old set-head residual path, or oracle
metric is authorized by this plan.
