# r5 vertex integration shadow protocol

The r5 dense vertex run demonstrated candidate recall, but it did not save a
head checkpoint or candidate cache.  Its metrics receipt is therefore not an
inference artifact and cannot be connected to the production graph path.

`tools/audit_identity_aligned_vertex_integration_shadow.py` defines the next
read-only gate.  It accepts a target-independent `candidates.jsonl` only.  A
row has the form:

```json
{"sample_key":"...", "source":"shadow",
 "candidates":[{"x":0.42,"y":0.51,"score":0.93}]}
```

The artifact must contain exactly the 1,170 accepted r5 rows and no target,
element, graph, or match fields.  The audit then:

1. runs frozen `initial.pt` and records the production candidate/bond input;
2. copies that baseline prefix byte-for-byte;
3. removes shadow duplicates within 6 px and performs an in-memory,
   baseline-first merge;
4. attaches residual points to unoccupied frozen bond endpoints within 12 px;
5. checks complete-edge versus dangling-endpoint counts, allowing a valid
   residual-residual partner while requiring two distinct endpoint owners,
   without calling the
   production selector, pair head, graph builder, or SDF writer.

The output is non-admitting even when all metrics pass.  The hard gate requires
unchanged checkpoint and baseline graph-input hashes, zero shadow rows in the
production forward boundary, perfect baseline prefix retention, shadow
candidate precision at least 0.85, zero dangling endpoints, and no baseline
regression.  The current run is intentionally `NO-GO`: the candidate artifact
does not exist, so the report is incomplete and fail-closed.  Do not retrain
solely to satisfy this gate and do not promote a checkpoint.
