# r17 m128 panel and target-free rollback shadows

This is a CPU-only, identity-disjoint panel expansion. The production
checkpoint remains unchanged (`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`), and the sealed test was not read.

## Panel

The m128 cache contains 128 molecules per train/fresh split and 64 for dev,
with four balanced render profiles and zero molecule overlap. The aromatic
sidecar reached fresh precision/recall `0.9891/0.9397`; the type and element
context sidecars reached `0.9835` and `0.9900` accuracy. These are sidecar
diagnostics, not production metrics.

Compile attribution with the m128 sidecars was:

| split | candidate-context sanitize |
|---|---:|
| train | 0.8340 |
| dev | 0.8555 |
| fresh | 0.9022 |

One fresh image had no candidate pair record; it is counted as an explicit
no-candidate abstention by the rollback report.

## Rollback policies

| policy | train | dev | fresh | identity finding |
|---|---:|---:|---:|---|
| ordinary one-edge rollback | 0.8652 | 0.9336 | 0.9452 | 15 true target-edge deletions, 10 unmapped |
| connected candidate graph | 0.8613 | 0.8945 | 0.9256 | 8 true non-aromatic deletions, no unmapped |
| connected + preserve candidate-cycle edges | 0.8340 | 0.8555 | 0.9022 | 0 transactions; pure abstention |

The ordinary policy's sanitize gain is therefore not graph identity repair.
The connected policy is safer but misses the pre-registered fresh gain gate;
preserving every candidate cycle removes all accepted transactions. No policy
is authorized for GPU training or production integration.

## Decision

Freeze these sidecars as shadow evidence. Do not train another compiler-validity
head. The next useful experiment must add local visual edge evidence or a
multi-edge transaction objective with explicit identity/edge-recall accounting;
sanitize alone is not an acceptable optimization target.
