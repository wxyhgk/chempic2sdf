# Current-initial endpoint ranking audit (2026-10-06)

This is a read-only replay on the fixed public 1,200-image cohort using
`outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt`
(SHA-256 `09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`).
The selected baseline is immutable: baseline IDs are assigned first, candidate
points are kept at least 12 px from baseline points, and residual ownership is
matched at 6 px. No checkpoint or production decoder was changed.

The audit compares heatmap ranking with deterministic combinations of the
already-emitted role, local-prominence, and anchor-offset signals. Results are
in the corrected replay
`outputs/audits/chemvector-anchor-rank-features-20261006-r2/report.json`:

| ranking | K=16 positives / precision | K=32 positives / precision | K=64 positives / precision |
| --- | ---: | ---: | ---: |
| heatmap | 195 / 1.016% | 207 / 0.539% | 210 / 0.273% |
| heatmap × role | 195 / 1.016% | 207 / 0.539% | 210 / 0.273% |
| heatmap + prominence | 186 / 0.969% | 203 / 0.529% | 207 / 0.270% |
| heatmap + role + prominence | 186 / 0.969% | 203 / 0.529% | 207 / 0.270% |
| heatmap × role × offset | 196 / 1.021% | 207 / 0.539% | 210 / 0.273% |

The selected baseline leaves 505 residual targets. K=64 recovers only 210
(41.6% residual recall), while adding thousands of false-positive sidecar
points. Therefore a global-rank or feature-reranking bounded trial is not
admissible.

The dense reachability replay shows all 232 raw-gap targets have a dense cell
within 6 px, but only 45 are in the top-128 score cells and none is a 3x3 local
peak. The gap is concentrated in visible text endpoints (193/232; O=112,
N=64) and `font_large`/`serif` styles. The next useful experiment is a
train-only endpoint-objectness/positive-assignment study using this current
initial contract. It must first reproduce immutable baseline IDs and satisfy:

1. old selected retention = 100%;
2. at least 50 train positives and train AUPRC >= 0.50;
3. train candidate precision >= 0.85;
4. holdout locked residual precision >= 0.85 and at least 14 new matches;
5. no change to production checkpoint or sealed-test identities.

Until those gates pass, do not train a rescue scorer or launch a global-rank
trial.

## Follow-up 129-parameter probe

The existing bounded heatmap-readout probe was subsequently replayed in memory
for 100 optimizer steps and stopped as soon as selected ownership changed
(`outputs/audits/chemvector-heatmap-rank-retention-trial-20261006/report.json`).
It recovered 3 raw train targets and 24 raw holdout targets, but lost 1 raw and
3 raw baseline IDs; more importantly, it lost 1 selected train ID and 17
selected holdout IDs. Its deployment/trial gate is therefore false. This
confirms that a global score update, even when restricted to 129 parameters,
cannot be accepted without a separate protected replacement/sidecar contract.
