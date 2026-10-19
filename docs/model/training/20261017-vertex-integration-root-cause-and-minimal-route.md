# r5 vertex integration: root cause and minimal next route

## What failed

The three reported precisions measure different objects and must not be
collapsed into one number:

| boundary | holdout observation | interpretation |
|---|---:|---|
| vertex candidates | about 0.51 | the dense head has high recall but emits many extra peaks |
| vertex -> complete frozen bond | about 0.67 | a clean candidate can often complete a bond, but candidate ownership is noisy |
| residual merge candidates | about 0.195 | the merge admitted a broad residual field, not a graph-ready atom set |

The 0.195 result is partly an artifact bug: the first `v5` JSONL export wrote
`score: 0.0` for every candidate.  Thus the merge had no ranking signal.  A
scored artifact must be exported from the reviewed state before using this
number for a training decision.  The old file is retained and must not be
overwritten.

The structural failure remains after that correction.  On the reviewed v1
integration holdout there are 140,362 active frozen bond endpoints for only
6,468 target endpoints (about 21.7x).  The geometry-only attachment routine can
therefore attach almost every extra vertex to some port.  In the all-candidate
merge, 8,279 residual attachments produce only 1,403 complete residual edges;
6,876 are dangling because the opposite endpoint is not owned by the frozen
baseline.  This is endpoint/partner ownership failure, not a bond-type failure:
conditional bond-type accuracy is about 0.964, while an oracle candidate set
reaches about 0.978 complete-bond precision.

## Route decision

Do not train the bond head or change the production selector.  Do not run a
threshold sweep.  The smallest useful experiment is a frozen-candidate
objectness/ownership sidecar:

1. export scored vertex candidates from the reviewed state (inference-only);
2. cache candidate features from frozen inference: heatmap score, role
   probabilities and margin, local stride-4 feature, coordinate/offset,
   nearest frozen-port distance, frozen port score/type, free-port count, and
   partner-support count;
3. train only a small MLP (at most 10k parameters) on the fixed identity split;
4. keep every frozen production candidate protected; sidecar rows can only
   claim residual atoms and only as an in-memory graph shadow.

Training labels are used only while fitting/auditing: strict unique atom
ownership within 6 px is positive, baseline-owned atoms are protected, and
ambiguous/unknown candidates are masked rather than treated as negatives.
The sidecar must be trained with a weighted BCE plus per-image hard-negative
pairwise ranking loss; a post-hoc score threshold alone is not an allowed
experiment.

The endpoint/partner sidecar is a second stage only.  It is not useful before
candidate purity is fixed, because the frozen endpoint field has a 21.7x
port-capacity surplus and geometry-only attachment will label arbitrary peaks
as ports.  If the candidate sidecar passes, train a separate endpoint-pair
objectness head on frozen bond-query features and require both endpoints of a
new edge; never attach a one-sided residual row.

## Fixed one-shot gate

The run is in-memory only, at most 800 optimizer steps, and uses the existing
696/474 molecule-disjoint r5 split.  No production checkpoint, selector, graph
input, or sealed identity may be read or written.

The candidate sidecar is admissible only if all checks pass:

- scored artifact has exactly 1,170 rows, non-constant finite scores, and no
  target fields;
- initial checkpoint SHA and production state digest are unchanged;
- baseline prefix retention is exactly 1.0 and baseline graph/bond tensors are
  byte-identical;
- fixed calibrated posterior (chosen once from train, no sweep) gives holdout
  residual candidate precision at least 0.85 and residual recall at least
  0.50;
- holdout complete-edge precision is at least 0.80, dangling residual rate is
  at most 0.10, and every accepted new edge has two distinct endpoint owners;
- no profile has residual precision below 0.75, and no baseline bond/graph/SDF
  metric regresses.

Any one failure is `NO-GO`; there is no fallback threshold or production
integration.  Only after this gate passes should the endpoint-pair sidecar be
considered.

## Corrected replay addendum

The reviewed scored export was replayed with `vertex_heatmap_logits` as the
existence source (`role_aware=false`).  On the molecule-disjoint holdout the
new candidate path reaches 0.9369 bond recovery, but only 0.6988 complete-bond
precision and 0.3585 attached-endpoint precision.  The scored residual merge
has 1,445 attachments, 197 complete edges, and 1,247 dangling endpoints
(86.3%).  The oracle candidate path is about 0.978 complete-edge precision and
conditional bond-type accuracy is about 0.964, so the remaining failure is
partner/edge ownership rather than bond-type recognition.

The CPU endpoint-pair separability preflight (fixed 160/80 image slice) has
holdout AUPRC 0.939, but its predeclared operating point gives precision 0.681;
it therefore does not authorize a GPU sidecar.  The next bounded experiment,
if resumed, must score complete candidate–port pairs with explicit partner
support and require two distinct endpoint owners.  No production state or
checkpoint is admissible until that sidecar passes every fixed gate above.

## Pair sidecar shadow result

The revised baseline-owner plus residual-owner CPU preflight passed pair
separability (holdout precision `0.8688`, recall `0.7846`, AUPRC `0.8835`), so
one isolated 800-step GPU shadow was run. Its independent edge audit reached
holdout complete-edge precision `0.8663` with zero dangling endpoints, but edge
recall was only `0.3873` on the fixed 80-image holdout slice. The sidecar is
therefore `NO-GO`: it improves ownership precision but does not recover enough
edges, and no production graph, selector, or checkpoint was touched.

An edge-level train-only threshold calibration raised holdout recall only to
`0.4000` at precision `0.8109`, still failing the `0.50` recall gate. A
capacity audit with `max_per_side` 24/48/96 produced identical reachability
(holdout target-pair reach `0.4164`), so increasing candidate slots is not the
repair. The missing recall is structural candidate/port compatibility, not a
threshold or top-k capacity issue.
