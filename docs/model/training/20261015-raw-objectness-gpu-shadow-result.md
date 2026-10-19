# Raw atom-objectness GPU shadow

The raw-candidate atomness objective passed a CPU preflight, so one bounded
in-memory GPU shadow was authorized.  This was a new scalar sidecar only:

- positives: `protected` + `strict_positive` raw matches;
- negatives: confirmed `hard_negative` candidates;
- masked: `near_unknown` and `selected_unmatched`;
- features: frozen `raw_score`, `presence_logit`, `x_norm`, `y_norm`;
- 100 AdamW steps, batch 2,048, fixed zero-logit gate;
- production model, selector, OCR, and initial checkpoint frozen;
- no checkpoint written and no deployment authorization.

The shadow completed on one RTX 5090.  Results:

| dev style | conservative precision | raw-atom recall |
| --- | ---: | ---: |
| coordgen | 0.981 | 0.990 |
| original | 0.980 | 0.988 |
| rdkit | 0.981 | 0.990 |
| stress (separate) | 0.929 | 0.969 |

The initial checkpoint SHA remained unchanged.  The corrected audit is:

`outputs/audits/chemvector-style-factorial-raw-objectness-gpu-shadow-20261010-r1/final-audit.json`

This validates a raw atomness sidecar, not strict selector rescue.  The
follow-up graph-constrained and predicted-port audits found that frozen bond
endpoints cannot safely rescue dropped strict candidates (core conservative
precision below 4%).  Selector rescue is therefore frozen; any future work
must keep protected candidates on the frozen production path and treat raw
atomness as an independent sidecar only.
