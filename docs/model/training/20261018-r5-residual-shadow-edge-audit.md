# r5 residual-only shadow: candidate versus edge audit

The residual-only proposal shadow completed one bounded 800-step run on the
fixed 160/80 slice. Its candidate-level and edge-level quantities must not be
collapsed:

| holdout quantity | result | gate |
|---|---:|---:|
| residual targets | 175 | denominator |
| decoded residual candidates | 270 | denominator |
| correctly owned residual atoms | 165 | — |
| residual precision | 0.6111 | **fail** (>= 0.85) |
| residual recall | 0.9429 | pass (>= 0.80) |
| hard collision rate | 0.0000 | pass (<= 0.10) |
| combined baseline+residual bond coverage | 0.9818 | pass (>= 0.95) |
| one-residual bond coverage | 0.3727 | **fail** (>= 0.50) |
| both-residual bond coverage | 0.0109 | diagnostic |

Visible heteroatom recall is high (O `0.9892`, N `0.9434`), but profile-level
residual precision remains only `0.48`--`0.756`. Thus the residual head finds
the missing O/N locations but emits too many extra candidates. The aggregate
combined bond coverage is not an edge-recovery result: it includes bonds
already covered by the immutable baseline. It cannot authorize graph merge or
production use.

The attempted independent integration audit was stopped before producing a
receipt because the fixed shadow gate already failed and the audit would only
consume more GPU without changing the decision. The authoritative shadow
receipt remains:

```text
outputs/audits/chemvector-identity-aligned-r5-residual-only-shadow/report.json
```

The sidecar is `NO-GO`; `initial.pt` and production state are unchanged. Do
not tune a holdout threshold. A future route would need explicit residual
candidate precision/ownership supervision (especially O/N hard negatives) and
then a fresh preregistered edge audit, not another pair scorer or promotion.
