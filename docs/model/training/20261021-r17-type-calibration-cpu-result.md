# r17 low-parameter bond-type calibration (CPU)

This is a quarantine-only probe. It learns only four additive bond-class logit
biases and one temperature on the frozen chemical logits. The frozen presence
sidecar supplies the candidate edge set; no edge is added or removed, the
production checkpoint is not written, and the sealed test is not read.

## Protocol

The cross-source calibration parameters are selected on the t3e train split
(`oa13`, `oa33`) using a 1pp per-profile typed-precision floor. The fixed
parameters are then evaluated on the disjoint t3e holdout and on the newer t4b
holdout. For t4b profiles not represented in t3e, the cross-source report
explicitly falls back to the frozen baseline rather than extrapolating a
profile policy. A separate t4b-train fit is reported only as an in-source
headroom diagnostic.

Command:

```bash
R17_CALIB_OUT=outputs/audits/chemvector-r17-type-calibration-cpu-20261021-v5 \
  python tools/train_r17_type_calibration_cpu.py
```

## Result

The selected t3e parameters were:

```text
bias = [0.15, 0.30, 0.30, -0.30]
temperature = 0.80
```

On t3e holdout the typed metrics were unchanged (both profile deltas 0). On
t4b, applying the fixed calibration only to the overlapping `oa13`/`oa33`
profiles gave:

```text
oa13: typed precision +2.50 pp, typed recall +2.44 pp
oa33: typed precision  0.00 pp, typed recall  0.00 pp
```

The independent t4b-train fit is useful as an upper-bound diagnostic: it only
improved `oa00` and `oa02` on t4b holdout (+1.25/+1.35 pp typed precision) and
left all other profiles unchanged. This is not evidence for a global
calibrator because it was selected in-source.

## Decision

`production_go=false`. The cross-source probe is safe but too narrow to justify
a GPU run or a production sidecar: t3e does not cover six t4b profiles, and the
t3e holdout has no gain. The evidence points away from a global class-prior
problem and toward profile/style-conditioned visual features (or a more
targeted aromatic representation). Keep the production checkpoint frozen and
use the next CPU experiment to test that upstream hypothesis.

Authoritative report:

`outputs/audits/chemvector-r17-type-calibration-cpu-20261021-v5/report.json`
