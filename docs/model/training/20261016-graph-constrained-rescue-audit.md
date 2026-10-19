# Graph-constrained rescue audit (2026-10-16)

## Scope

This was a read-only upper-bound audit on the frozen `initial.pt` checkpoint
(SHA256 `09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`).
It replayed the non-sealed style-factorial development pool: 396--398 images
per style after the recorded coordinate-collision exclusions. No optimizer,
checkpoint write, target geometry, graph/layer supervision, or sealed test was
used.

For each raw candidate, the fixed rule required a predicted bond endpoint
within 12 px, bond presence logit above `-0.5`, one-to-one endpoint ownership,
and the opposite endpoint to be within 12 px of a baseline protected atom.
Near/unknown candidates were not treated as negatives; cache labels were read
only after the rule to calculate audit metrics.

## Result

| style | strict dropped | strict rescued | strict ceiling | hard FP survival | conservative rescued precision |
|---|---:|---:|---:|---:|---:|
| coordgen | 88 | 35 | 39.8% | 3.78% | 2.48% |
| original | 105 | 55 | 52.4% | 3.78% | 3.82% |
| rdkit | 102 | 48 | 47.1% | 3.38% | 3.74% |
| stress | 728 | 148 | 20.3% | 3.36% | 10.73% |

The strict recall ceiling misses the 50% gate for coordgen and rdkit, and the
conservative precision is far below the 85% gate for every style. Therefore a
graph-only rescue/veto is **NO-GO**. It cannot safely recover dropped atoms;
the low hard-negative survival does not compensate for the precision failure.

The smaller 20-images-per-style predicted-port smoke independently showed the
same failure (core conservative precision 0.20--0.32% and nearly all accepted
attachments hard negatives), so scaling that route is not justified.

## Decision

Freeze selector rescue research. Keep the already validated raw-atomness
sidecar as the only usable auxiliary path; it remains separate from selector
decoding and does not modify the production checkpoint. Do not authorize a GPU
run for graph rescue, endpoint reranking, or another pointwise selector gate.

Audits:

- `outputs/audits/chemvector-style-factorial-graph-veto-20261016-r1/report.json`
- `outputs/audits/chemvector-style-factorial-predicted-port-attachment-cpu-20261015-r1/report.json`

The isolated API contract for the validated-but-not-deployed atomness path is
`chemvector/components/graph/proposals/raw_atomness_sidecar.py`. It preserves
the protected baseline prefix and represents low-confidence rows as abstains;
it does not create graph nodes or authorize a checkpoint.

## Next bounded work

The next data action is targeted expansion, not another selector architecture:
new molecule identities (not copied style variants), with clean baselines and
profiles that produced the most strict drops (small/large fonts, unseen serif
fonts, thin/thick strokes, tight/wide multiple-bond offsets). The expanded
pool must pass identity/connection-disjoint replay and a strict-only CPU gate
before any sidecar training is considered. The raw-objectness shadow has no
deployable weights, so it must not be treated as a production model.
