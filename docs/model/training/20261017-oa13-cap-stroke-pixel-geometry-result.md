# oa13 cap/stroke pixel-geometry diagnostic

The 480-image cap/stroke panel was replayed against the frozen initial and
then inspected before selector matching.  For every profile and every target
atom, at least one active raw proposal lies within the 16 px matching radius:

| profile | visible-text within 16 px | implicit/unlabeled within 16 px |
|---|---:|---:|
| `cov13_base` | 100% | 100% |
| `cov13_cap52` | 100% | 100% |
| `cov13_stroke16` | 100% | 100% |

The raw reachability gate still remains 0.936--0.940 because one-to-one
matching loses targets when several target atoms compete for the same local
proposal cluster.  Candidate multiplicity is high (for example, 574 of 579
visible-text targets in `cov13_base` have more than one nearby proposal), so
the observed gap is not evidence that the serif glyph has no visual response.

This changes the diagnosis: the immediate bottleneck is proposal geometry /
assignment collision and target capacity, not bond cap or stroke rendering.
It does not authorize a selector rescue: the earlier selector/endpoint rescue
families remain frozen NO-GO.  Any future training trial must first prove that
it improves one-to-one atom coverage while preserving hard-negative precision,
and must use a fresh bounded shadow artifact rather than `initial.pt`.

Artifact:

`outputs/audits/chemvector-oa13-cap-stroke-pixel-geometry-20261017-r2/report.json`
