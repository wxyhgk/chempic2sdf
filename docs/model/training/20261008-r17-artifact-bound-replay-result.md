# r17 artifact-bound replay (2026-10-08)

The frozen graph artifact was independently consumed without calling the model.

- artifact: `outputs/audits/chemvector-r17-known-clean-graph-artifact-v1/graph-artifact.jsonl`
- rows: 1536; exact sample-key/manifest match
- artifact SHA256: `e25c5d3bd7b499e192cea52582452efffdc1a2907b39ab4d8615e2030a1a1441`
- target-free schema and identity checks: pass
- production checkpoint SHA: unchanged

The validator is `tools/audit_r17_artifact_bound_graph_replay.py` and its report is
`outputs/audits/chemvector-r17-known-clean-artifact-bound-replay-v1/report.json`.
This is an artifact-binding and endpoint audit, not a deployment or training
authorization. The known-clean cohort still overlaps the historical slice80, so
the strict slice80-disjoint cohort must be used for any subsequent sidecar shadow.
