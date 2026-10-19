# r5 line-first node/graph shadow

The endpoint residual route is blocked by candidate-pair reachability.  The
next isolated experiment therefore keeps the line-first proposal and the
already-trained edge/type sidecar fixed, and trains only a small node-keep
head.  Node features are derived from the image, LSD endpoint cluster, and
the frozen vertex candidate score/geometry.  No target field is available to
the inference path; labels are joined in memory after proposal construction.

## Protocol and stop gate

- fixed identity-aligned slice: 160 train / 80 holdout for the bounded GPU
  shadow; threshold calibrated on train only;
- one GPU, at most 800 optimizer steps, no checkpoint write to `initial.pt`;
- edge/type sidecar remains frozen; incomplete node endpoints are rejected
  before graph emission;
- stop immediately on non-finite loss, changed production SHA, sealed-test
  access, or any production graph/selector mutation;
- accept the isolated shadow only if holdout node precision/recall are at
  least `0.90/0.85`, edge precision/recall at least `0.90/0.75`, conditional
  type accuracy at least `0.90`, and emitted dangling/self-loop counts are
  both zero.  A pass never authorizes production.

The training receipt is:

```text
outputs/audits/chemvector-identity-aligned-line-first-node-graph-shadow-r5/report.json
```

It passes on the fixed 80-image holdout (`node P/R=.9363/.9095`, edge
`P/R=.9893/.8400`, type `.9892`, emitted dangling/self-loop `0/0`).

Before any further decision, the exact train-only state is replayed on all
474 accepted holdout images:

```text
outputs/audits/chemvector-identity-aligned-line-first-node-graph-shadow-r5-full/report.json
```

The full replay also passes aggregate floors (`node P/R=.9301/.8964`, edge
`P/R=.9598/.7981`, type `.9667`, emitted dangling/self-loop `0/0`).  Two
profiles remain below the same recall floors (`cov13_base48` edge recall
`.7161`, node recall `.8279`; `cov13_stroke16_48` edge recall `.7180`, node
recall `.8474`).  These are diagnostic weaknesses, not a promotion signal;
the chemistry/topology roundtrip remains a separate gate and production stays
frozen.
