# Style-factorial geometry + local-pixel late fusion

After the scalar candidate gate failed, a fixed CPU-only late-fusion probe was
run on the same frozen-initial candidate cache.  The input contained only
deployable features: raw proposal score, set-presence logit, normalized
candidate coordinates, and a candidate-centred 32 px raster crop reduced to
16×16.  Ground-truth nearest distance, atom identity, and protected members
were not inputs.  Training used strict positives and confirmed hard negatives;
near-unknown rows were masked, not negatives.  The gate was fixed at zero with
no threshold sweep.

The probe recovered most strict positives but admitted many line/ink hard
negatives:

| dev style | conservative precision | strict recall |
| --- | ---: | ---: |
| coordgen | 0.091 | 0.909 |
| original | 0.098 | 0.886 |
| rdkit | 0.095 | 0.853 |
| stress (held out) | 0.195 | 0.852 |

Report: `outputs/audits/chemvector-style-factorial-late-fusion-cpu-preflight-20261010-r1/report.json`.

## Decision

Adding a small local crop to frozen geometry improves recall relative to the
scalar gate, but does not solve the atom-versus-line ambiguity: precision is an
order of magnitude below the required 0.85.  No GPU adaptation is authorized.
The next design must model candidate competition/graph context or use a
stronger pretrained visual encoder; another pointwise MLP, threshold sweep, or
more hard negatives would not address this failure.
