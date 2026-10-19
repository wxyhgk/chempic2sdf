# Frozen-initial replay on the style-factorial dev pool (2026-10-10)

The 1,600-molecule dev proposal was replayed with the immutable `initial.pt`.
This was forward-only; no optimizer step, checkpoint, production write, or
sealed access occurred. Eight samples were rejected by the committed
coordinate-collision rule, leaving 1,592 replayable images.

| variant | raw recall | selected recall | selector-drop rate |
|---|---:|---:|---:|
| original | 0.9962 | 0.9882 | 0.0080 |
| rdkit | 0.9956 | 0.9880 | 0.0077 |
| coordgen | 0.9964 | 0.9897 | 0.0068 |
| stress | 0.9883 | 0.9307 | 0.0582 |
| all replayable | 0.9941 | 0.9741 | 0.0202 |

The ordinary three variants have good endpoint reachability, while `stress`
causes a large selector-retention drop. Therefore the pool is not admissible
as one undifferentiated training stream. `stress` should remain a held-out
robustness stratum until a separate objective is shown to preserve the frozen
prefix; it must not be allowed to dominate the first adaptation.

Decision: **no GPU training yet**. The next CPU gate is a core-stream replay
(`original/rdkit/coordgen`) with strict/near-unknown/hard-negative strata and
an independent stress-only report. A training authorization requires no loss
on the old 480-image prefix and strict conservative precision ≥0.85, recall
≥0.50 on every declared core style stratum.

Artifact:

`outputs/audits/chemvector-style-factorial-endpoint-reachability-20261007-r1/report.json`
