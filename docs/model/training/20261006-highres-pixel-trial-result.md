# High-resolution pixel rescue trial (2026-10-06)

This was a bounded in-memory trial, not a production training run.  The
production `initial.pt` remained frozen and was verified with SHA-256
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`.

The new branch is a 1,313-parameter raw-pixel 384x384 residual over a
bilinearly enlarged copy of the existing 192x192 heatmap.  Its final layer is
zero-initialized.  The trial used 120 raw-gap training images for 200 virtual
steps and then replayed the full fixed 1,200-image training cohort.  The
1,080 images not used for updates are treated as a deterministic holdout.
Target cells use the existing q1023-to-floor convention; 384-cell collisions
were checked and found in zero images.  Rescue peaks are matched to targets by
one-to-one Hungarian matching at 6px.

## Holdout result

| primary 128 + sidecar | union matched | new gap targets |
|---:|---:|---:|
| K=8 | 32,102 / 32,185 | 3 |
| K=16 | **32,113 / 32,185** | **14** |
| K=32 | 32,125 / 32,185 | 26 |

The sidecar-only K=16 precision was 92.6% on the holdout.  Improvements were
mostly `font_large`/`serif` visible-text cases; `line_thick` implicit-C gaps
were not recovered.  The primary 192 candidates were not modified or
discarded in this audit.

## Decision

```text
high_resolution_field_capacity_signal_go
integration_pending
```

This is enough evidence to continue to a rescue-union/selector audit, but not
to deploy or start a full training run.  The sidecar currently exceeds the
production 128-candidate contract and has not been evaluated through selector,
pair, valid, or exact-molecule metrics.  The next experiment must fix K=16,
keep primary candidates immutable, and reject any per-ID downstream
regression.  No checkpoint was written.

Machine reports:

- `outputs/audits/chemvector-atom-highres-pixel-trial-20261006-r6/report.json`
- `outputs/audits/chemvector-atom-highres-pixel-trial-20261006-r6/holdout-audit.json`
