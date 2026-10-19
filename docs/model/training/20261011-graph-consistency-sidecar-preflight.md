# Graph-consistency sidecar CPU preflight

The frozen initial model was replayed on a fixed 120-train/40-dev slice per
core style (stress separately). A two-layer, 32-dimensional candidate graph
head consumed frozen atom scores, candidate geometry, and predicted bond
endpoint/presence context. Protected baseline rows were retained unchanged;
strict/hard rows alone entered the loss; near-unknown and selected-unmatched
rows were masked. No GT distance, target atom ID, or target graph entered the
features. The gate was fixed at zero logits after 300 CPU steps.

Report:

`outputs/audits/chemvector-style-factorial-graph-consistency-cpu-preflight-20261007-r1/report.json`

| dev style | conservative strict P | strict R | hard-negative survival |
|---|---:|---:|---:|
| coordgen | 0.005 | 0.571 | 0.364 |
| original | 0.004 | 0.750 | 0.363 |
| rdkit | 0.006 | 0.750 | 0.365 |
| stress | 0.019 | 0.423 | 0.429 |

Protected retention was exactly 1.0, but precision and hard-negative gates
failed by a large margin. Candidate graph message passing plus predicted bond
context is therefore a CPU NO-GO and receives no GPU authorization. Do not
repair this result by scanning the gate or adding steps; the route is closed
unless a new data/label contract is proposed.
