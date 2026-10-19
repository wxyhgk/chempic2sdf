# r17 residual-density t3d CPU result (2026-10-19)

The first t3 source could not satisfy the requested quotas after historical
identity filtering.  A smaller, explicitly quarantined source was then built:

- 8 train molecules / 8 holdout molecules
- profiles: oa13 and oa33 only
- 32 images total
- holdout endpoint quotas include aromatic, triple, ring, and O/N endpoint
  degree strata
- prior-use ledger identity admission passed
- model strata audit confirmed the holdout quotas

The source remains `training_admissible=false`; it is a smoke cohort, not a
release dataset.

Artifact:

`outputs/audits/chemvector-r17-residual-density-pair-incidence-cpu-20261019-t3d/report.json`

A 100-step CPU-only frozen pair-presence residual shadow produced:

| profile | typed-recall delta | typed-precision delta | decision |
|---|---:|---:|---|
| oa13 | 0.00 pp | −5.51 pp | reject |
| oa33 | +2.86 pp | −2.76 pp | reject |

The result is informative: increasing residual-density can expose additional
oa33 edges, but the current residual objective adds too many false or wrongly
typed edges.  `checkpoint_unchanged=true`, no GPU was used, no sealed test was
read, and no production artifact was written.

Decision: do not promote this sidecar and do not launch GPU training.  The
next bounded run must use hard-negative/base-logit-anchored residual training
and a residual-only precision gate on a larger t3 source; if that fails, the
pair-incidence route is closed.

## t3e follow-up

A larger two-profile smoke cohort was then admitted separately:

- 16 train + 16 holdout molecules, oa13/oa33 only;
- 64 rendered images;
- ledger, provenance, identity, image uniqueness, and split disjointness all
  passed; full-profile release gate intentionally remained false.

On 200 CPU steps, the frozen residual head gave:

- oa13 typed recall `+6.00 pp`, typed precision `+0.49 pp`;
- oa33 typed recall `+1.96 pp`, typed precision `−1.92 pp`.

Raising the route threshold to `0.95` preserved recall gains but did not fix
the oa33 typed-precision drop.  Changing negative ratio from 3 to 1 was
identical on t3d.  A t3e type-readout shadow also fell back to baseline.  The
gain is real but not safe: the remaining errors are frozen chemical-type
misclassifications on otherwise true additions.  Therefore t3 remains a
diagnostic data source only; no GPU or production integration is authorized.

For attribution, fixing the t3e route at threshold `.95` and applying a
train-only type readout only to the nine additions produced 7 correctly typed
additions.  The resulting typed precision/recall was `.95/.9596` overall;
the per-profile typed results were oa13 `.94/.94` and oa33 `.96/.96`.
This is a diagnostic ceiling from the same smoke cohort, not an independent
release result, so it remains NO-GO for GPU or production.
