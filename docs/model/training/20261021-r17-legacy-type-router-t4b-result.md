# r17 legacy type-head transfer router (CPU)

This probe reused the already-trained generic `m128` type-context head. It did
not retrain that head. A single margin threshold was selected on t4b train to
choose between its argmax and the frozen chemical-head argmax; the candidate
edge set was unchanged.

The train-only selector chose `old_margin - chemical_margin >= 1.55`. On the
disjoint t4b holdout, typed deltas were:

```text
oa00 +1.25 pp   oa02 +1.35 pp   oa11  0.00 pp   oa13 +2.50 pp
oa20  0.00 pp   oa22  0.00 pp   oa31 -2.44 pp   oa33 +2.47 pp
```

The `oa31` regression violates the per-profile precision gate and the
all-profile recall-gain gate also fails. Therefore transfer of the old head is
not safe as a global router. The few improvements confirm that the old visual
features contain some useful information, but the representation is not
stable enough across rendering profiles to route by confidence alone.

Decision: `production_go=false`; do not start GPU training for this route. The
next useful work is upstream style-diverse visual training or a profile-aware
feature representation, with a fresh identity-disjoint holdout.

Authoritative report:

`outputs/audits/chemvector-r17-legacy-type-router-t4b-cpu-20261021-v1/report.json`
