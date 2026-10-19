# Frozen bond/port-context CPU preflight

This bounded experiment tested whether atom candidates become separable when
the scorer sees frozen predicted bond endpoints in addition to atom scores.
The replay used a fixed lexicographic slice: 120 train samples per core
variant and 40 samples per evaluation variant (coordgen, original, rdkit,
and stress). The initial checkpoint and all bond endpoint/presence outputs
were frozen; no GPU, checkpoint, or sealed test was used.

Features included atom score/presence, candidate rank and candidate-to-
candidate competition, nearest predicted bond endpoint distance and
proximity, endpoint count, and endpoint presence-logit margins. Ground-truth
distance was never read as a feature. Strict and hard rows were the only loss
rows; near-unknown rows were masked; protected rows were context only and
were excluded from the scorer evaluation denominator in r2.

Machine-readable result:

`outputs/audits/chemvector-style-factorial-bond-context-cpu-preflight-20261007-r2/report.json`

| dev variant | strict precision | strict recall |
|---|---:|---:|
| coordgen | 0.000 | 0.000 (0/14) |
| original | 0.000 | 0.000 (0/8) |
| rdkit | 0.000 | 0.000 (0/12) |
| stress | 0.000 | 0.000 (0/78) |

The fixed prior-derived gate caused near-total abstention on this small
slice, and the route fails the gate. It is not eligible for GPU training.
The slice has only 17–29 strict positives per train variant, so the result is
diagnostic rather than a claim that bond context is impossible at scale; any
future attempt would require a new, larger identity-disjoint CPU release and
authorization, not a retry or threshold sweep of this run.
