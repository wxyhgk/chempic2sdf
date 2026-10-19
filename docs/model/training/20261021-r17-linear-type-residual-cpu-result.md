# r17 linear type residual (CPU)

This probe reduced the type residual to a single linear layer (`features` plus
the four frozen chemical logits → four residual logits, 1,068 parameters).
Training used only selected true t4b train edges; candidate presence and edge
sets stayed frozen.

The first draft was rejected during contract review because it treated the
last four visual features as chemical logits. It was not used as evidence. The
corrected run explicitly concatenates `features` and `chemical_logits` in both
training and replay and checks that a zero residual exactly preserves the
frozen argmax.

Corrected holdout typed deltas:

```text
oa00 -2.50 pp   oa02 +1.35 pp   oa11 +1.30 pp   oa13 +1.25 pp
oa20 +1.32 pp   oa22  0.00 pp   oa31 -1.22 pp   oa33 +1.23 pp
```

The per-profile precision gate fails (`oa00`, `oa31`) and there is no gain in
oa22. Reducing capacity therefore does not solve the profile-dependent
aromatic/type instability. `production_go=false`; no GPU training or merge is
authorized for this route.

The train/holdout contrast is itself diagnostic: the residual improved train
typed precision by roughly 3.9--8.9pp per profile, while the same fixed weights
lost 2.5pp on oa00 and 1.2pp on oa31 holdout. This is direct evidence of
overfitting or a train/holdout style/structure shift, so enlarging the head is
not the next move.

Authoritative corrected report:

`outputs/audits/chemvector-r17-linear-type-residual-cpu-20261021-v4/report.json`
