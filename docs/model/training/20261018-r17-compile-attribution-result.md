# r17 compile error attribution (2026-10-18)

The read-only attribution replay keeps the same frozen candidate edges and
strict aromatic transaction, then replaces one predicted field at a time with
ground truth only for diagnosis.  It does not change routing or production.

| split / variant | sanitize |
|---|---:|
| dev candidate | .8125 |
| dev oracle elements | .9375 |
| dev oracle types | .8281 |
| dev oracle all | .8438 |
| fresh candidate | .8281 |
| fresh oracle elements | .9375 |
| fresh oracle types | .8281 |
| fresh oracle all | .8906 |

The dominant recoverable gap is element-conditioned chemistry: replacing only
predicted elements raises fresh compile from `.8281` to `.9375`, while replacing
bond types alone does not move it.  The remaining oracle-all failures are
topology/aromatic consistency, not a reason to launch another aromatic GPU
run.  This supports a single bounded CPU element/valence repair audit before
any consideration of GPU shadow training.

Artifact:

```text
outputs/audits/chemvector-r17-compile-attribution-all-m16/report.json
```

Production checkpoint SHA is unchanged and sealed test was not read.

The same context head was then trained/evaluated on the larger 64-molecule
identity slice (256 images per split), together with a matching m64 type
residual. Fresh element accuracy was `.9875`; the predicted-context compile
replay reached `.9219` sanitize, but dev was only `.8711`. Therefore the
larger replay still fails the joint dev/fresh chemistry gate. This is a useful
CPU result, not a GPU authorization.

Artifacts:

```text
outputs/audits/chemvector-element-context-head-cpu-r17-all-m64/report.json
outputs/audits/chemvector-type-residual-cpu-r17-all-m64/report.json
outputs/audits/chemvector-r17-compile-attribution-all-m64-v2/report.json
```
