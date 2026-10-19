# O/N raw proposal head: bounded replacement-source pilot (2026-10-17)

## Diagnosis

The r4 frozen replay is an upstream proposal failure, not a selector failure:

| metric | r4 |
|---|---:|
| visible O/N targets | 3,179 |
| raw atom candidates | 127,018 |
| unique raw O/N ownership | 73 (2.30%) |
| strict <=6 px | 71 |
| refinable 6--16 px | 2 |
| ambiguous/zero ownership | 3,106 |

Changing a threshold, adding an offset refiner, or training a selector cannot
recover targets that are not present in the raw candidate field.  The existing
text-objectness, raw-atomness, and high-resolution residual trials are therefore
not the next route: they all score or add candidates after the failed proposal
field.

## Reusable implementation

Use an independent dense proposal source made from existing components:

* `ChemVisionEncoder.forward_stroke_resolution` supplies the frozen 192x192
  stride-4 feature map (96 channels);
* `ChemVertexDetector` supplies a small heatmap/offset/aromatic head;
* `VertexCollator`, `build_vertex_heatmap`, and `vertex_detection_loss` already
  implement anchor and sub-cell offset targets;
* `decode_vertices(..., role_aware=True)` provides a deterministic top-K dense
  proposal decode;
* `atom_center_global_peak_rank_loss` may be added to the heatmap loss so true
  anchors can enter the top-K boundary.

This head predicts the complete atom candidate set directly.  It does not read
the old atom proposals, selected slots, set logits, nearest distances, target
IDs, graph, or OCR.  The two role channels remain `visible-text` versus
`implicit`; O/N is an audit stratum, not an input feature or a separate OCR
head.

The prior virtual-stride-2 and high-resolution center-field experiments are
useful capacity evidence but are not reused as the proposal source: the broad
20k center-field route was `NO_GO`, while the r4 raw replay now requires an
independent complete proposal field.

## Fixed 1k GPU trial (design only; do not start yet)

Use the accepted r4 panel only, split by molecule identity (715 train images,
472 holdout images).  Exclude an image if any atom-anchor cells collide on the
192x192 grid.  No sealed test and no production checkpoint are read or written.

Model and optimizer are fixed before launch:

* new `ChemVertexDetector` only; zero/random head initialization is recorded;
* freeze all existing atom, bond, set, pair, OCR, and decoder modules;
* freeze the vision trunk except the final stride-4 block;
* head learning rate `2e-4`, final stride-4 block learning rate `1e-6`;
* batch size 8, exactly 800 optimizer steps, deterministic seed;
* focal anchor loss + offset loss + role heatmap loss (role weight `0.5`) +
  global top-128 rank loss (weight `0.5`, margin `0.5`);
* decode exactly 128 regional maxima from the new field; do not merge with or
  alter the protected production proposal prefix.

The run is in-memory/shadow only.  It may write a metrics receipt, never a
model checkpoint.

## Read-only preflight before GPU

The preflight must verify, on the exact r4 manifest:

1. molecule-disjoint train/holdout and zero loader/target roundtrip errors;
2. no cell-collision images in either split, with the excluded denominator
   reported separately;
3. visible O and visible N targets are both present in every core profile;
4. existing initial model digest and state are unchanged;
5. new head parameter count and optimizer scope match the fixed recipe;
6. model inputs contain pixels only; target anchors/roles are used only by the
   loss and audit.

If any item fails, no GPU trial is authorized.

## Locked evaluation gates

Metrics are reported independently for `oa13` and `oa33`, O and N, degree-1/2/3,
font/size/rotation strata, and core versus stress profiles.  The candidate
source is evaluated by one-to-one matching at 6 px and by candidate precision;
there is no selector or graph score in this gate.

The trial passes only if all holdout gates pass:

* visible-O recall at 6 px >= 0.50;
* visible-N recall at 6 px >= 0.50;
* no core O/N stratum below 0.35 recall;
* all-atom recall >= 0.80 and implicit-C recall >= 0.70;
* top-128 candidate precision >= 0.30;
* train and holdout both improve over a locked same-decoder baseline computed
  from the old raw field (same top-K, NMS, and 6 px one-to-one matching).  The
  published 2.30% r4 unique-ownership number is a diagnostic lower bound, not
  the comparison metric; no train-only pass is accepted;
* stress is reported separately and cannot compensate for a core failure.

Failure is terminal for this recipe: do not tune K, thresholds, NMS radius,
loss weights, or retry with another head under the same receipt.  A pass only
authorizes a CPU integration shadow; it does not authorize replacing the
production atom head or adding graph edges.

## Why this is the smallest meaningful experiment

It tests the missing capability at its source: can a compact spatial head learn
visible atom anchors from pixels when the current raw field cannot reach them?
It reuses target construction and decoding already covered by unit tests, keeps
the production path byte-for-byte frozen, and has a direct stop condition.  If
this fails, the evidence points to the vision/data contract rather than another
proposal or selector tweak.
