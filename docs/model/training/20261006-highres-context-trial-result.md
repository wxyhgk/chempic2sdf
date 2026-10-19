# Dilated-context high-resolution branch trial (2026-10-06)

To test whether the first pixel rescue branch was limited by its small
receptive field, a second zero-start branch added depthwise dilation-2 and
dilation-4 context.  The old 192x192 field and primary candidates stayed
frozen.  The branch was trained in memory on 120 raw-gap images for 200 steps
and evaluated on the 1,080-image holdout.

## Holdout result

| rescue K | 7x7 branch | dilated-context branch |
|---:|---:|---:|
| K=8 | 3 new gaps | 1 new gap |
| K=16 | **14 new gaps** | 6 new gaps |
| K=32 | 26 new gaps | 26 new gaps |

The context branch did not improve the useful high-precision K=16 operating
point.  It is therefore a **NO-GO** as the next architecture; do not spend GPU
budget scaling this variant.

The original 7x7 branch remains a capacity signal only.  It still requires a
dedicated capacity-aware selector and full downstream retention audit before
any training continuation or deployment.

Machine report:
`outputs/audits/chemvector-atom-highres-context-trial-20261006-r1/report.json`.
