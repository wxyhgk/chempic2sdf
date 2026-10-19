# Visible O/N pilot frozen reachability audit (2026-10-18)

This is a read-only replay of the identity-disjoint visible O/N data-first
pilot.  It reads only the public `train-*.tar` shards named by
`outputs/audits/chemvector-visible-on-data-first-pilot-20261017-r1/manifest.json`.
The initial checkpoint is frozen in CUDA eval/no-grad mode; no optimizer,
checkpoint, sealed split, target injection, or training decision is involved.

## Contract and result

The replay consumed 800 train and 200 holdout images (four renderer variants
per molecule, 200/50 molecule groups).  Strict ownership is one candidate
within 6 px, with no candidate near another target, a 4 px margin, and no
candidate reuse.  A 16 px match is reported only as a target-free upper-bound
diagnostic.

| split / arm | strict ownership | any candidate ≤6 px | any candidate ≤16 px | candidate precision |
|---|---:|---:|---:|---:|
| train / raw (47,976 targets) | 0.8958 | 0.9956 | 0.9997 | 0.4199 |
| train / selected | 0.8909 | 0.9842 | 0.9897 | 0.8920 |
| holdout / raw (9,320 targets) | 0.9814 | 0.9968 | 1.0000 | 0.3579 |
| holdout / selected | 0.9750 | 0.9833 | 0.9876 | 0.9854 |

On the holdout, visible O strict recall is `0.9800` (1,652 targets) and
visible N is `0.9798` (892).  O degree-1/2 recall is `.9780/.9850`; N
degree-1/2/3 is `.9750/.9764/.9885`.

The profile field in this pilot is renderer variant rather than `oaXX`.
For the raw arm, holdout strict recall is `coordgen .9923`, `original .9922`,
`rdkit .9932`, and `stress .9543`.  The selected arm retains `.9932/.9932/.9936`
on the first three variants but drops stress to `.9274`; train selected stress
is `.8604`.  Implicit-C degree-2 is the weakest train raw stratum at `.8399`
(selected `.8363`, holdout selected `.9742`).

## Decision

The visible O/N pilot does not show a missing-proposal ceiling: almost every
visible O/N target has a nearby frozen raw candidate.  The selector converts a
low-precision raw field into a high-precision set, but loses roughly 2.7
percentage points of stress recall on the holdout.  The next useful objective
is therefore stress-specific candidate retention/ownership and implicit-C
robustness, not a new OCR head or a graph-validity loss.  This receipt does
**not** authorize training; any selector/atom-head trial requires a separately
fixed CPU protocol and gates.

Artifact:

`outputs/audits/chemvector-visible-on-frozen-reachability-20261017-r4/report.json`
