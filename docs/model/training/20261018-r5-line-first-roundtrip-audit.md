# r5 line-first graph/chemistry roundtrip audit

This read-only replay serializes only mapped, distinct line-first candidate
edges and runs RDKit. Proposal construction never reads target atoms or bonds;
those labels are joined post-hoc for metrics. No GPU or production state is
modified.

Receipt:

```text
outputs/audits/chemvector-identity-aligned-line-first-graph-chemistry-roundtrip-r5/report.json
```

Fixed holdout (80 images):

- edge precision `0.9893`, edge recall `0.8400`;
- conditional bond-type accuracy `0.9892`;
- 14 incomplete endpoint edges dropped from serialization (`2.74%` of raw
  selected edges), no self-loops, isolated mapped-node rate `3.10%`;
- candidate-only carbon topology RDKit sanitize `70/80 = 0.875`;
- post-hoc oracle-element overlay sanitize `76/80 = 0.950`.

The gap between carbon-only and oracle-element sanitize is expected: this is a
bond/topology replay, not an atom-identity release. `cov13_stroke16_48` is the
weakest profile (dangling `6.67%`), so any future graph shadow must retain
per-profile reporting rather than relying on the aggregate gate. The artifact
remains diagnostic (`production_go=false`).

The same replay over all 474 accepted holdout images is a stricter decision
check (`...roundtrip-r5-v2/report.json`): edge precision `0.9598`, recall
`0.7981`, type accuracy `0.9667`, dangling `5.22%`, candidate-only sanitize
`0.7595`, and oracle-element sanitize `0.8038`. It therefore fails the full
gate. In particular, the 80-image slice must not be used to authorize a graph
shadow; the cov13 profiles need targeted repair first.
