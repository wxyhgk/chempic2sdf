# r5 route B: graph→render residual critic feasibility

This is a read-only feasibility note for a possible graph-level render
critic.  No production checkpoint, selector, OCR, or graph head was changed;
the sealed test was not read and no training was run.

## Question

Can a critic compare the input raster with a rendering of a proposed graph and
repair missing bond endpoints or bond type without changing the proposal
stage?

## Existing evidence

The strict line-first graph/chemistry roundtrip is the closest available
measurement.  On the 80-image slice, frozen candidate endpoints give edge
precision `0.9893`, edge recall `0.8400`, conditional type accuracy `0.9892`,
raw dangling rate `0.0274`, candidate-only sanitize `0.875`, and
oracle-element sanitize `0.950`.

The complete 474-image replay is the decision measurement:

| metric | value |
|---|---:|
| mapped edge precision | 0.9598 |
| edge recall | 0.7981 (2581/3234) |
| conditional bond-type accuracy | 0.9667 |
| raw dangling rate | 0.0522 (153/2929) |
| candidate-only sanitize | 0.7595 |
| oracle-element sanitize | 0.8038 |
| images with all bonds recovered | 142/474 |

The authoritative replay is
`outputs/audits/chemvector-identity-aligned-line-first-graph-chemistry-roundtrip-r5-full/report.json`;
the same values are also in the v2 report.  The missing 653 GT bonds are
unreachable from the frozen candidate endpoint graph.  A residual image can
expose their pixels, but a critic that only scores existing graph edits cannot
create the missing endpoints.  Adding endpoint search changes the component
into a new proposal route and must be audited separately.

## Decision

Do not fit or run a GPU graph-render critic now.  It can still be useful later
as a post-hoc edge/type validator after endpoint reachability is repaired, but
it cannot address the current bottleneck.

If revisited, the CPU gate is:

* candidate endpoint set and prefix edges are immutable;
* full 474-image replay, with every profile reported;
* edge precision `>= .965917`, recall `> .850031`, dangling `<= .05`;
* every profile recall `>= .75` and sanitize `>= .90`;
* predicted-type sanitize `>= .95` and conditional type accuracy `>= .90`;
* any residual-based endpoint proposal is a separate route and cannot inherit
  this critic gate.

Current full replay fails recall, dangling, and both sanitize requirements.

