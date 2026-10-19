# Strict raw-dense split trial (2026-10-06)

This was one bounded in-memory trial from the frozen production checkpoint
`initial.pt` (SHA-256
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`).  The
fixed 1,200-image training-only cohort was split by group, molecule, and
canonical SMILES into 480 train and 720 holdout images with zero overlap.

The trial trained only a zero-start 1,313-parameter 384px raw-pixel residual
for 100 steps.  The frozen 192px heatmap, atom decoder, selector, pair head,
OCR, and chemistry heads were not updated.  Sidecar peaks were evaluated
independently with a 12px guard and a strict 6px target match; they were not
sent to the production selector or graph merger.

## Result

| split | images | candidates | strict matches | strict gap hits | precision |
|---|---:|---:|---:|---:|---:|
| train | 480 | 7,680 | 0 | 0 | 0.0000 |
| holdout | 720 | 11,520 | 0 | 0 | 0.0000 |

The trial failed all positive-recovery gates and is **NO-GO**.  Per protocol it
was not extended to 200 steps, retuned, or connected downstream.  The dense
reachability preflight remains useful as a capacity observation (83/149
reachable raw gaps), but it does not establish a learnable or deployable
residual field.

Machine report:
`outputs/audits/chemvector-raw-dense-split-trial-20261006-r1/report.json`.

Safety: `training_performed=true` refers only to the in-memory sidecar;
`production_checkpoint_written=false`, `sealed_test_read=false`, and the
production SHA was unchanged.
