# Graph-context aromatic sidecar (2026-10-18)

The independent AtomObject aromatic head was rejected on the identity-disjoint
r16 split (holdout accuracy `.8224`, aromatic recall `.6835`).  The remaining
signal is graph-level: frozen AtomPair evidence plus target-free cycle support.

An isolated `LayerNorm(526) -> Linear(64) -> Linear(1)` edge sidecar was trained
for 500 CPU steps.  Its input contains frozen pair hidden features, frozen
chemical logits, endpoint AtomObject features, geometry, endpoint degree, and a
target-free alternate-path/cycle bit.  The candidate edge set is frozen; the
sidecar only scores aromatic membership.  No profile ID is an input.

For the scaffold-disjoint r17 slice, 64 molecules per split were used and all
four clean profiles were mixed into one shared head (256 images per split).

| split | candidate edges | aromatic precision | aromatic recall |
|---|---:|---:|---:|
| dev | 3,536 | .98964 | .91833 |
| fresh holdout | 3,553 | .98402 | .92491 |

The per-profile CPU probes also showed that oa00, oa11 and oa20 pass the `.90`
fresh precision/recall threshold.  The difficult oa33 profile exposes a
proposal/cycle issue: with the hard cycle gate its recall was `.7679`, while a
diagnostic no-cycle replay reached `.9972`; therefore cycle gating must become a
transaction with rollback/abstention, not an unconditional suppression rule.

Artifacts:

```text
outputs/audits/chemvector-graph-context-aromatic-cpu-r17-all-m64-t0p72-c1/report.json
outputs/audits/chemvector-graph-context-aromatic-cpu-r17-oa00-m64/report.json
outputs/audits/chemvector-graph-context-aromatic-cpu-r17-oa11-m64/report.json
outputs/audits/chemvector-graph-context-aromatic-cpu-r17-oa20-m64/report.json
outputs/audits/chemvector-graph-context-aromatic-cpu-r17-oa33-m64/report.json
```

This is still a sidecar-only CPU result.  It is not a production checkpoint,
not a sealed-test result, and not yet a graph-level chemistry gate.  The next
required audit is a target-free complete-ring transaction with predicted
elements, candidate-vs-oracle compile metrics, per-profile paired diffs, and
rollback/abstain accounting.  That audit is now recorded at
`outputs/audits/chemvector-graph-context-aromatic-compile-r17-all-m16-v5/report.json`.
