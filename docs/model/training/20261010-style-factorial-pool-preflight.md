# Style-factorial target pool preflight (2026-10-10)

The 100k committed style dataset was checked as a possible next source after
the strict selector and local-patch probes failed on the narrow 480-image
release. The check read only `samples.csv`, `molecules.csv`, and the source
commitment; it did not read image payloads, sealed data, or a checkpoint.

## Result

- source: 25,000 molecules / 100,000 four-variant samples;
- overlap with the current 480-image pool and v0.2 proposal: zero molecule,
  InChIKey, and scaffold overlaps;
- deterministic proposal: 1,600 train molecules (6,400 samples) and 400 dev
  molecules (1,600 samples), four variants per molecule;
- train/dev molecule overlap: zero;
- the proposal covers N/O single, double, triple, and aromatic cells except
  the chemically absent O-triple cell; N-triple has 186 train molecules and
  O-aromatic 285.

The proposal is **not training-authorized**. It is only an identity-bound
manifest. A frozen-initial endpoint reachability replay and exact payload
audit must run before any optimizer step. The next model experiment should
use a multi-scale crop/endpoint branch only if that replay shows sufficient
raw and selected endpoint coverage.

Artifact:

`outputs/audits/chemvector-style-factorial-target-pool-20261007-r1/`
