# Dense feature evidence audit (2026-10-06)

The fixed 1,200-image replay identified 232 raw-gap atoms.  For each gap, the
current stride-4 feature at the target cell was compared with the feature at
the nearest predicted dense peak from the read-only peak audit.  This is an
observation only: no optimizer, label, checkpoint, or decoder state was
changed.

| gap role | cases | median cosine | median relative L2 |
|---|---:|---:|---:|
| visible text | 193 | 0.6215 | 0.9560 |
| implicit | 39 | 0.4462 | 1.0753 |
| all | 232 | 0.6038 | 0.9919 |

The target-cell representation is therefore not merely a duplicate of the
nearest false peak.  It contains substantially different information, while
the existing one-channel heatmap readout assigns it a low score.  This makes
a new readout/high-resolution field technically plausible, but does **not**
justify changing the production head yet.

The failed 192x192 residual shadow means the next experiment must change the
field representation or its readout capacity.  The recommended order is:

1. construct a zero-start continuous 384x384 rescue field that preserves the
   existing 192x192 heatmap as an immutable base;
2. prove step-zero proposal parity and run a tiny in-memory shadow on the
   fixed raw-gap slice;
3. only if every selected ID preserves old raw/top-128 matches and at least
   one gap is recovered, run a bounded GPU trial; otherwise retain
   `initial.pt`.

The rescue field must not silently replace the current downstream feature
contract.  Candidate fusion and graph/chemical evaluation remain separate
gates.
