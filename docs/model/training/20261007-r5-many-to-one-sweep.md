# Legacy many-to-one ownership sweep

The old detector and frozen edge/type sidecar were replayed on CPU with
many-to-one score-distance ownership.  We swept radii `16/20/24/28` px,
score lambdas `0/4/8`, and topology-guard tau `0.2/0.4/0.6` (36 settings).
Selection used train only (`P >= 0.96592`, `R > 0.85034`, then maximum recall).

The selected train setting was `radius=28`, `lambda=8`, `tau=0.4`; the full
holdout gave `P/R/sanitize = 0.96277/0.85560/0.95148`.  Profile recalls were
`0.803/.870/.805/.894/.874/.887`, and profile sanitize rates all exceeded
0.90.  However, no setting met both the holdout precision floor `0.96592` and
recall target `0.85034`; widening/relaxing ownership trades precision for
recall and does not improve the route enough to authorize training or release.

Report: `outputs/audits/chemvector-identity-aligned-line-first-many-to-one-sweep-v2-r5/report.json`.

