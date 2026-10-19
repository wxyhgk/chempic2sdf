# Vertex-to-bond integration shadow (2026-10-17)

This is a read-only compatibility audit between the r5 vertex field and the
frozen production bond proposal head.  It is not a training run, a checkpoint
promotion, or a production change.

## Inputs and isolation

The audit reads the fixed r5 train/holdout manifest, the frozen initial
checkpoint, frozen bond proposals from `structure_forward`, and (when supplied)
an in-memory-equivalent vertex-head state receipt.  The vertex state must be a
separate `torch.save` payload containing only the new vertex head and the
explicitly trainable stride-4 block.  The production checkpoint is loaded into
a fresh model and is never overwritten.  Missing vertex state is a deliberate
`NOT_READY` result; the audit must not silently instantiate an untrained head.

Candidate generation never reads atom IDs, target coordinates, target bonds,
nearest-GT distances, or graph mappings.  Ground truth is consulted only after
assignment for label-only diagnostics.  Three paths are reported:

* `old`: frozen production atom proposals + frozen bond proposals;
* `new`: supplied r5 vertex candidates + frozen bond proposals;
* `oracle`: target anchors + frozen bond proposals, an upper bound only.

All paths use the same fixed decoder: score threshold 0.05, at most 128
candidates, 6 px candidate deduplication, and a 12 px endpoint radius.  Each
endpoint can be claimed once per bond; a candidate may serve multiple bonds at
a junction, but never both endpoints of one bond.

## Required readouts

For train and holdout separately, and for O, N, implicit C, degree strata and
profile, report endpoint attachment rate, complete two-endpoint bond rate,
distance quantiles, unmatched endpoints, duplicate/reuse conflicts, and
junction/non-junction rates.  Only after this geometry table, report target
identity compatibility, complete bond recovery, and conditional bond-type
accuracy.  Oracle numbers are explicitly labelled as an upper bound and are
never merged into `new` precision or recall.

## Stop gates

The fixed holdout is `GO` only if `new` improves old endpoint attachment by
0.20 and complete-bond rate by 0.15, correct attached-endpoint precision is at
least 0.85, complete-bond precision at least 0.80, bond recovery at least 0.50,
visible O/N recovery at least 0.50, implicit-C recovery at least 0.70, and
conditional type accuracy at least 0.75.  Candidate reuse conflicts must be at
most 0.05.  Any missing state, use of GT during candidate generation, a failed
checkpoint digest, or an oracle-only success is `NO-GO`.

The report always records `training_performed=false`, `deployment_authorized=false`,
and `sealed_test_read=false`.
