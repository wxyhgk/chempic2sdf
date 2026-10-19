# r5 line-first node identity gate

The candidate-level node probe scored every frozen vertex proposal separately
and failed (`P=0.8589`, `R=1.0` on holdout). This was largely a duplicate
ownership denominator problem: several scored candidates can surround one
line node.

The corrected CPU contract first performs a deterministic, target-free global
nearest assignment from each provisional LSD node to at most one scored vertex
candidate within 16 px. A node-level MLP then receives candidate score/position,
node-candidate distance, cluster size/spread/degree, segment length, local ink,
and candidate-neighborhood counts. Labels are joined only after proposal
construction; the threshold is train-only.

Receipt:

```text
outputs/audits/chemvector-identity-aligned-line-first-node-identity-mlp-r5-cpu-v3/report.json
```

Holdout (fixed 80 images): 612 assigned nodes, 573 true nodes, precision
`0.9363`, recall `1.0000`. The learned threshold retains all assigned nodes,
so these numbers are the deterministic one-to-one assignment ceiling rather
than evidence of learned node discrimination. The CPU gate is technically
pass, but a GPU node shadow has no attributable expected gain and should not
consume budget before graph/chemistry assembly replay.

The original candidate-level artifact remains unchanged; no production state
or sealed test was touched.
