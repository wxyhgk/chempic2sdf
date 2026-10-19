# r17 fixed-edge bond-type transaction shadow

We tested a different repair family on the m128 identity-disjoint panel. The
candidate edge set was frozen; for invalid graphs, a target-free CPU policy
enumerated bounded one/two-edge SINGLE/DOUBLE/TRIPLE edits on low-confidence
non-aromatic edges and accepted only compiler-valid transactions.

The apparent sanitize improvement was large:

| split | baseline | type transaction |
|---|---:|---:|
| train | .8340 | .9258 |
| dev | .8555 | .9414 |
| fresh | .9022 | .9687 |

However, the identity/type audit showed the repair was correcting compiler
validity by changing already-correct bond types:

| split | changed types | correct before | correct after |
|---|---:|---:|---:|
| train | 86 | 86 | 0 |
| dev | 33 | 31 | 2 |
| fresh | 50 | 43 | 7 |

Edge recall and edge set were unchanged, and aromatic edges were untouched, but
fresh overall type accuracy fell from `.9623` to `.9572`. A stricter one-edge,
one-candidate variant reduced the damage but failed the required fresh sanitize
gain (`.9217`, only `+.0196`).

Decision: **NO-GO**. Compiler validity is again an unsafe proxy for molecular
identity. Do not train or merge this transaction policy. The next admissible
experiment must supply independent local visual bond evidence and require both
sanitize improvement and non-decreasing bond-type identity.
