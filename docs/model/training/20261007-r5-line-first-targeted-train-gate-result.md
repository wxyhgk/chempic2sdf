# r5 line-first targeted train gate

## What was run

The fixed r5 train identities were re-rendered into a quarantine-only panel:

```text
outputs/datasets/chemvector-identity-aligned-line-first-targeted-train-r5
```

It contains 116 train molecules × 3 cov13 profiles (`cov13_base48`,
`cov13_cap52`, `cov13_stroke16_48`) = 348 images.  It has zero overlap with
the fixed holdout identities.  No model, sealed test, production checkpoint,
or optimizer was used.

## CPU results

The target-free LSD endpoint proposal passes the targeted reachability gate:

| profile | edge precision | bond recall |
|---|---:|---:|
| cov13_base48 | 0.9888 | 0.9588 |
| cov13_cap52 | 0.9888 | 0.9626 |
| cov13_stroke16_48 | 0.9889 | 0.9655 |

After frozen scored-candidate alignment and the existing edge/type sidecar,
the graph round-trip remains below release quality:

- edge precision: `0.9826`
- edge recall: `0.8799`
- bond-type accuracy: `0.9854`
- dangling rate: `0.0297`
- oracle-element sanitize: `0.7787`

The failure is concentrated in aromatic/multiple-bond topology.  Aromatic
rows sanitize at `0.7251`; single-only rows sanitize at `0.9394`.  This means
the line detector and bond-order classifier are not the immediate bottleneck;
the next bounded work is target-free aromatic closure/ownership repair and a
fresh round-trip gate.

A conservative target-free aromatic-closure probe was also run on the targeted
train panel and fixed holdout.  It proposed zero additional edges (the current
candidate graph does not expose the required pair of degree-one aromatic
endpoints), so all metrics were unchanged.  This rules out a cheap post-hoc
closure rule as the next fix; the missing information is earlier endpoint /
candidate ownership or ring-topology representation.

An alignment-radius sweep gives the same conclusion.  Increasing the radius
from 16px to 24px raises fixed-holdout edge recall from `0.798` to `0.829` and
lowers dangling from `0.052` to `0.010`, but oracle sanitize remains `0.804`
(targeted train `0.776`).  Smaller radii are worse.  Radius tuning alone is
therefore not a release path.

Authoritative reports:

- `outputs/audits/chemvector-identity-aligned-line-first-targeted-train-gate-r5/report.json`
- `outputs/audits/chemvector-identity-aligned-line-first-targeted-train-roundtrip-r5/report.json`
- `outputs/audits/chemvector-identity-aligned-line-first-targeted-failure-strata-r5/report.json`
- `outputs/audits/chemvector-identity-aligned-line-first-aromatic-closure-probe-r5/report.json`
- `outputs/audits/chemvector-identity-aligned-line-first-alignment-radius-sweep-r5/report.json`

## Decision

`NO-GO` for another GPU shadow.  Do not train the edge head, add a new node
head, or promote the line-first sidecar until the topology gate reaches
oracle-element sanitize ≥ 0.90 and full-holdout dangling ≤ 0.05.  Production
initial remains unchanged.
