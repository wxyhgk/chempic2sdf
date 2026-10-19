# r17 t2 type-readout shadow decision (2026-10-18)

The bounded CPU shadow `tools/train_r17_pair_type_readout_shadow_cpu.py` was
run against the disjoint t2 train/holdout cache.  It used the frozen
production checkpoint, frozen pair-presence sidecar, and frozen candidate
pair set.  No sealed test content was read and no production artifact was
written.

Artifact:

`outputs/audits/chemvector-r17-targeted-source-pair-type-readout-shadow-cpu-20261018-t2/report.json`

The train-only route selection found no admissible score delta and fell back
to the baseline route (`additions = 0`).  Consequently the holdout was
identical to baseline:

| metric | baseline = shadow |
|---|---:|
| edge precision | 0.99752 |
| edge recall | 0.98710 |
| typed precision | 0.92179 |
| typed recall | 0.91216 |

All safety checks passed (`checkpoint_unchanged`, CPU-only, no sealed-test
read, no production write), but both required stress gains failed:

- oa13 typed-recall gain: false
- oa33 typed-recall gain: false

Decision: **NO-GO**.  Do not integrate this readout, launch GPU training, or
continue tuning pair-presence/type thresholds on t2.  The next diagnostic
must expose a larger candidate union and evaluate visual endpoint ownership
jointly with type scoring.
