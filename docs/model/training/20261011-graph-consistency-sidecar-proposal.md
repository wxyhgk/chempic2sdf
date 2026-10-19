# Graph-consistency sidecar proposal (direction C)

The next materially different architecture should be a frozen-baseline graph
consistency sidecar, not another MoE expert or pointwise selector. The
repository already contains reusable implementations:

- `GroundedCandidateGraphRefiner` provides node projection, line-band visual
  evidence, endpoint interaction, blocker features, edge messages, and node
  residuals.
- `FullAdjacencyMessageLayer` provides bounded candidate-pair message passing
  from edge logits, geometry, and optional endpoint evidence.

## Proposed training contract

Freeze the initial vision/atom/bond/set outputs. Build a graph whose nodes are
raw atom candidates and whose edges are frozen bond proposals or fixed
candidate pairs. Train a small (<300k parameter) sidecar with node occupancy,
edge occupancy, endpoint incidence, and valence-risk heads. Use strict and
protected nodes as atomness positives, strict-only nodes as rescue positives,
hard negatives as negatives, and mask near-unknown/selected-unmatched rows.
Use a fixed hard-negative pair margin and node/edge/incidence multi-task loss.

At inference, protected baseline nodes are immutable. A deterministic matching
decoder enforces one candidate per endpoint, incident-bond support, no
isolated components, and valence/aromaticity vetoes. There is no learned
router and no threshold sweep.

## CPU gate before GPU

Use identity-disjoint train-core/dev-core with core styles reported separately
and stress held out separately. Require, for every core style:

- conservative graph precision ≥ 0.85;
- strict node recovery ≥ 0.50;
- hard-negative survival ≤ 0.15;
- protected retention exactly 1.0;
- valid-graph rate no worse than baseline by >1 percentage point.

Only after a complete CPU gate should a single bounded GPU shadow be
considered. Existing endpoint and pairwise audits make a failure likely, so a
failed CPU gate closes the route without increasing data, steps, or capacity.
