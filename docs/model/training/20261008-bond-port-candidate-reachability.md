# Bond-port constrained residual endpoint audit (2026-10-08)

## Purpose

The high-resolution residual branch previously recovered endpoints without
changing the frozen selector, but it generated too many candidates.  This
audit asks a narrower question: can a residual point be connected to an
unoccupied endpoint of a frozen predicted bond?  The production checkpoint is
unchanged (`initial.pt`, SHA-256
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`).

The baseline selector runs first.  Residual rows bypass the old set head and
are allowed to claim each bond port at most once.  A deterministic
predicted-port-to-ground-truth-bond match is used only for an oracle audit
label; it is never used to create candidates or select a threshold.  The
120/1080 split is group-, molecule-, and canonical-SMILES-disjoint.  No sealed
data is read and no checkpoint is written.

Machine reports:
`outputs/audits/chemvector-bond-port-candidate-reachability-20261008-r7/report.json`
and the larger split
`outputs/audits/chemvector-bond-port-candidate-reachability-20261008-r8/report.json`.

## Result

| holdout quantity | value |
|---|---:|
| residual candidates | 34,488 |
| locked target matches | 210 |
| free-port attached candidates | 17,341 |
| free-port attached target matches | 173 (0.998%) |
| unmatched targets | 450 |
| unmatched targets within 12 px of a free predicted port | 280 (62.22%) |
| strict oracle-compatible attached rows | 62 |

The fixed branch-score and score-plus-port rankings have AUPRC 0.1898 and
0.1980 on attached holdout rows.  Top-8 per image precision remains 1.8%.
Thus a geometric port gate alone does not make the residual scorer trainable.
The 62/62 oracle-compatible rows are an upper-bound diagnostic (they use GT
bond identity for labeling), not deployable precision.

The same 200-step audit on a larger 480/720 group-disjoint split produced 88
train positives and 133 holdout positives, but attached precision remained
0.91% (train) and 0.92% (holdout), with AUPRC 0.137 and 0.144. The strict
oracle-compatible subset was only 23 train and 47 holdout. More images improve
counts but do not improve candidate separability.

## Decision

**NO-GO for scorer training or deployment.**  The audit does establish a
useful capacity ceiling: about 62% of unmatched targets are geometrically near
a free predicted port, while only 62 rows satisfy strict predicted-port/GT
bond compatibility in this branch.  Any next learned endpoint experiment
must be conditioned on the local bond-port evidence and must report
non-oracle precision on the fixed holdout.  The baseline selector, set head,
and checkpoint remain frozen. The current implementation and stricter r7/r8
reports are in `tools/audit_bond_port_candidate_reachability.py` and
`outputs/audits/chemvector-bond-port-candidate-reachability-20261008-r7/` and
`...-r8/`.
