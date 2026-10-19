# r11 type-sidecar sampling review

The isolated `r11-profile` CPU trial used 474 train images (all 186 serif30
images and 96 images from each other profile), and capped each profile × bond
type bucket at 64 examples.  The type rows were approximately balanced:

```text
SINGLE 256   DOUBLE 220   TRIPLE 114   AROMATIC 256
```

On the 95-image fresh holdout, the frozen r10 presence sidecar was unchanged.
The type sidecar improved oracle sanitize from `.7579` to `.8316` and candidate
sanitize from `.7158` to `.8000`.  Serif30 r30 remained the limiting profile
(`.5652 → .6522` oracle); all-profile `.90` and production gates were false.

The trial's remaining methodological issue is important: it minimizes
`CE(type_residual(features), target)` and then adds those absolute class logits
to the frozen chemical logits.  This can inject a global class prior (notably
AROMATIC) and is not a true residual objective.  The next CPU-only A/B should
instead minimize `CE(base_chemical_logits + residual, target)` with profile ×
type inverse-frequency weights (softened, e.g. square root) and a conservative
residual scale (≤ `.5`).  No corrected trial was merged or promoted here;
production checkpoint and sealed test remain untouched.
