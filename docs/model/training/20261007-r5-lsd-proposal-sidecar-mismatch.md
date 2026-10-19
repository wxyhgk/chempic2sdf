# R5 LSD proposal / sidecar mismatch

## Scope

This note records a CPU-only investigation of a new raster proposal front-end.
The fixed production checkpoint (SHA256
`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`) and the
sealed split were not touched.  All reports below are train/holdout audit
artifacts; no model output was merged into production.

## Proposal ceiling

An LSD replay using `minimum_length=2`, `STD`, raster scale `0.75`, single-link
endpoint clustering at 10 px improved the target-free proposal ceiling on the
fixed 474-image holdout:

| route | edge precision | edge recall |
| --- | ---: | ---: |
| previous LSD | 0.96379 | 0.90586 |
| new LSD proposal ceiling | 0.96569 | 0.93247 |

All six profile recalls improved (0.913–0.947).  This is only a provisional
edge reachability result; it does not imply that the downstream candidate
ownership contract is compatible.

## Sidecar recompute

The frozen 17-feature edge/type sidecar was recomputed on the new proposal with
800 CPU steps and a train-only threshold.  Its full holdout sidecar metrics were
precision `0.74124`, edge recall `0.67409`, and type accuracy `0.98073`.
With Hungarian ownership (24 px, score lambda 8), merge-20 and the topology
guard, graph metrics were precision `0.97639`, recall `0.70346`, dangling rate
`0.02313`, and oracle sanitize `0.96203`.  Profile recalls were
`0.638/.688/.692/.755/.744/.703`; the required profile floor was not met.

This demonstrates a distribution/contract mismatch: the proposal ceiling is
high, while the frozen candidate alignment and feature sidecar cannot retain
the new endpoint/node identities.

## Threshold and ownership diagnostics

Train-only threshold values (`0.001` through `0.5`, including calibrated
`0.006219`) never reached the required aggregate recall `0.85034` or all
profile floors.  The calibrated threshold gave graph P/R/sanitize
`0.97639/.70346/.96203`; lowering it to `0.001` gave `.97583/.72418/.95570`.

An all-edge replay (no edge MLP, synthetic SINGLE type) isolated ownership:
holdout P/R/sanitize were `.97471/.77458/.99789`, but dangling rate was
`.29285`, with recalls `.716/.724/.783/.824/.842/.759` by profile.  A minimal
ownership A/B confirmed that widening to radius 40 and removing the score prior
made things worse (P/R `.97904/.59215`, dangling `.40287`).  Therefore radius or
threshold tuning alone is not a viable fix.

## Graph-hypothesis ceiling

As a final CPU-only check, 72 combinations of merge radius (12/16/20/24 px),
guard tau (0.2/0.4/0.6), aromatic cycle demotion, and train-derived edge
thresholds (0.004/0.006219/0.01) were replayed with fixed ownership.  The
train-selected hypothesis (merge 24, tau 0.2, cycle demotion on, threshold
0.004) reached holdout P/R/sanitize `.97528/.73191/.95570`; profile recalls
were `.685/.718/.731/.774/.766/.718`.  No hypothesis met aggregate recall
`0.85034` and all profile floors.

## Candidate-contract probes

The corrected new-LSD candidate-coordinate projection replay also failed: the
best train-selected projection had holdout P/R/sanitize
`0.99683/0.58318/.97468`, with dangling rate `0.74492`. A synthetic
center-only candidate ceiling was worse (P/R/sanitize
`0.40525/.13358/.97257`). These are CPU-only diagnostics, not deployable
models, and neither justifies training.

A candidate-conditioned line-support ownership cost was also tested on six
train-selected beta/lambda combinations. Its best holdout result was
P/R/sanitize `0.99211/.42764/.98945`, with dangling rate `0.86972`; the
incident-line penalty caused widespread unmatched endpoints. This route is
also NO-GO for GPU.

A target-free local raster bridge for the new LSD seam added 705 provisional
edges, but reached only holdout P/R/sanitize
`0.96935/.74335/.97890` with dangling rate `0.62270`; profile recall floors
also failed. The bridge therefore cannot justify a learned bridge scorer.

## Decision

**NO-GO for GPU or production.**  Keep production, OCR, selector, and the old
Hungarian/topology guard frozen.  The next meaningful experiment must rebuild
candidate ownership/proposals in the same coordinate/identity space as the
new LSD detector (or replace this proposal route); repeating edge/type training
or widening the ownership radius is not justified by the CPU evidence.

The older many-to-one route was also rechecked. A 144-combination CPU sweep
(ownership radius, score lambda, candidate score floor, and merge cap) found no
configuration satisfying both the frozen precision floor `0.96592` and recall
floor `0.85034`. The best recall was about `0.85993` at precision `0.95797`.
It remains diagnostic only and is not admitted to GPU shadow.
