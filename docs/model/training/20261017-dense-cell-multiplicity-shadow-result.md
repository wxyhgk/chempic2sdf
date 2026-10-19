# Dense-cell multiplicity shadow result

This was one isolated GPU shadow, not a production training run.

## Protocol

- Frozen checkpoint: `outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt`
- Initial SHA-256: `09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`
- Trainable scope: a new one-channel 1x1 extra-heatmap head over frozen stride-4 vision features
- Frozen: vision, base atom heatmap/offsets, selector, OCR, bond heads
- Fixed seed `20261017`, one GPU, 100 optimizer steps, 300-second ceiling
- Production checkpoint write: false; deployment: false; sealed-test read: false

Receipt: `outputs/audits/chemvector-dense-cell-multiplicity-shadow-authorization-20261017-r1/receipt.json`

## Result

| split | images | extra candidates | matched | strict collision recovery | precision |
|---|---:|---:|---:|---:|---:|
| train | 767 | 5,688 | 244 | 83 | 4.29% |
| holdout | 192 | 1,453 | 55 | 17 | 3.79% |

The sidecar did recover some previously unmatched targets, but it produced a
large number of false candidates. The required holdout precision was `>= 0.85`;
observed precision was `0.03785`. The gate therefore failed and the sidecar is
not eligible for selector/graph integration. No retry, threshold sweep, or
production continuation is authorized by this result.

Artifacts:

- `outputs/audits/chemvector-dense-cell-multiplicity-shadow-20261017-r1/report.json`
- `outputs/audits/chemvector-dense-cell-multiplicity-shadow-20261017-r1/sidecar.pt`

The production checkpoint SHA-256 after the run is unchanged. Existing CPU
contract tests pass (`3 passed`).
