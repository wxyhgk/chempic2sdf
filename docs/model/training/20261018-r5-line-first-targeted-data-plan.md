# r5 line-first targeted data plan after full round-trip audit

## Evidence

The 80-image panel was too optimistic. On all 474 holdout images, the fixed
line-first edge/type state and target-free scored-candidate alignment give:

- edge precision `0.9598`, edge recall `0.7981`
- bond-type accuracy `0.9667`
- dangling before rejection `153/2929 = 0.0522`
- oracle-element RDKit sanitize `381/474 = 0.8038`
- candidate-only C sanitize `360/474 = 0.7595`

The weakest profiles are `cov13_base48` and `cov13_stroke16_48`:

- edge recall `0.7161` / `0.7180`
- oracle sanitize `0.6962` / `0.7468`
- dangling `0.0807` / `0.0722`

The stronger `cov33_*` profiles have edge recall roughly `0.86`--`0.94`.
The profile plan binds `cov13` to DejaVu Serif, 180-degree rotation, and the
shorter bond-length cap; the weak stroke profile additionally uses 1.6px
lines. This points to raster/typography coverage and endpoint clustering, not
bond-order classification, as the next data problem.

## Next bounded data action

Create a metadata-only targeted release before any GPU run:

1. oversample `cov13` short-cap / 180-degree cases;
2. split line width 1.6 and 3.2 explicitly, with extra short single bonds and
   close junctions;
3. include aromatic rings, visible O/N adjacent to short bonds, and triple/double
   strokes in each weak profile;
4. add hard negatives for text glyph edges and line fragments near atom labels;
5. keep molecule/scaffold identity disjoint from the fixed r5 holdout and replay
   the same 696/474 identities only for diagnostics.

The release must first pass a CPU renderer/identity/coverage gate. Required
minimums before another GPU shadow: weak-profile line-edge reachability >=0.90,
candidate-aligned edge precision >=0.90, dangling <=0.05, and oracle-element
sanitize >=0.90 on a fresh targeted CPU panel. Until then, do not train another
node or residual head and do not promote the current line-first sidecar.

