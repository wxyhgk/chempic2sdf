# Dual raw-objectness CPU preflight (2026-10-15)

Two scalar heads were trained with natural candidate prevalence and a fixed
zero-logit gate. Head A predicts raw atomness (`protected` + `strict_positive`
versus `hard_negative`). Head B predicts strict-drop rescue (`strict_positive`
versus `hard_negative`); protected, near-unknown, and selected-unmatched rows
were masked for B. Both heads use only `raw_score`, `presence_logit`, `x_norm`,
and `y_norm`.

Head A passes the core styles:

| style | conservative precision | recall |
|---|---:|---:|
| coordgen | 0.979 | 0.994 |
| original | 0.978 | 0.992 |
| rdkit | 0.979 | 0.992 |
| stress (holdout) | 0.926 | 0.973 |

Head B fails badly under the fixed natural-prevalence protocol (core precision
below 0.001 and recall below 0.05). This is expected evidence that “raw atom
objectness” and “recover every dropped strict candidate” are different tasks;
the former is viable as a bounded sidecar while the latter is not.

Decision: only the atomness sidecar may proceed to a separately authorized
GPU shadow. Do not authorize a rescue/selector head, rebalance labels, or tune
the gate based on this run.

Artifact:

`outputs/audits/chemvector-style-factorial-dual-objectness-cpu-preflight-20261010-r1/report.json`
