# Coverage-panel replay → next route (2026-10-17)

This is a conditional protocol, not a run authorization.  The production
`initial.pt` remains immutable (`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`).
The selector, bond-port, graph-veto, OCR, and production decoder are not
updated by either branch below.

## Admission before any GPU

The fixed metadata plan is
`outputs/audits/chemvector-oa13-oa33-coverage-panel-plan-20261017-r3/`:
256 train identities + 64 holdout identities, eight profiles, and 2,560
planned rows.  It must first pass renderer-quality checks and one frozen
initial replay.  The replay must report each `cov13_*` and `cov33_*` cell
separately, retain `cov13_base`/`cov33_base`, and include visible O/N,
implicit-C, multiple-bond, and junction strata.

Required replay gate:

1. every cell has raw candidate reachability >= 0.98 on holdout;
2. exact SDF/connection identity, no text or attributed-ink clipping, and a
   complete eight-profile denominator;
3. train/holdout molecule, connection, and scaffold overlap is zero;
4. the six unaffected historical profiles remain within 0.5 percentage points
   of their 10k frozen replay values;
5. initial SHA, old replay, and sealed-test isolation remain unchanged.

The gate is per-cell.  Aggregate reachability cannot admit the release.

## Branch A: replay passes — one bounded raw-atomness sidecar shadow

Only this new sidecar is trainable.  It consumes four frozen candidate
features: `raw_score`, `presence_logit`, `x_norm`, and `y_norm`.  Profile IDs,
target distances, target atom IDs, graph/layer labels, and oracle assignments
are not inputs.  Labels are formed only after the frozen forward:

- positive: `protected` + `strict_positive` raw candidates;
- negative: confirmed `hard_negative` candidates;
- masked: `near_unknown` + `selected_unmatched`.

The protected production prefix is copied unchanged and is never rescored.
The sidecar may only mark a non-protected raw candidate as selected or
abstained for an offline downstream proposal; it cannot create an atom,
assign a bond endpoint, or alter the deployed graph.

Run exactly once after a separate receipt binds the panel manifest SHA,
profile-plan SHA, cache/replay SHA, initial SHA, and GPU index:

| item | fixed value |
|---|---:|
| GPUs | 1 |
| optimizer | AdamW, lr 0.002, weight decay 0.001 |
| seed | manifest-bound, fixed |
| batch size | 2,048 candidate rows |
| optimizer steps | 100 maximum |
| wall limit | 300 seconds |
| gate | logit 0.0, no sweep |
| retry/resume | forbidden |
| checkpoint | independent sidecar artifact only; never `initial.pt` |

Evaluate at step 0 and step 100 (or wall cutoff), per core profile and on a
separate stress holdout.  The minimum informational gate is conservative raw
precision >= 0.85 and raw recall >= 0.50 for each core profile, with masked,
hard-negative, protected-retention, and selected-count metrics reported
separately.  A pass does not claim strict selector rescue or graph recovery;
graph metrics remain unavailable until the separately specified frozen
bond/pair artifact exists.

The saved artifact, if the shadow is informative, must be named independently
(for example `raw_atomness_sidecar.pt`) and bind architecture, feature
normalization, seed, panel/replay hashes, and initial SHA.  It is an
experimental artifact, not a production checkpoint or deployment candidate.

## Branch B: replay fails — no GPU, no selector training

Do not retry the same oa13/oa33 render, lower the 0.98 gate, or train on a
failed panel.  Use the observed failing cell and its visible O/N versus
implicit-C breakdown to select a second metadata-only panel:

- preserve the failing base profile;
- change exactly one `bond_length_cap` (28 ↔ 52) or one
  `bond_line_width` (1.6 ↔ 3.2) at a time;
- keep font family, 32 px size, and rotation fixed;
- use at most 128 train + 32 holdout fresh identities per failing parent;
- run renderer quality and one frozen replay only.

If no cap/stroke control reaches 0.98, classify the issue as renderer/model
representation coverage rather than a trainable selector problem.  The next
work is a pixel/geometry diagnostic of glyph ink and bond proximity; it still
does not authorize model training.  If one control reaches 0.98, admit only
that profile family for future data release and repeat the coverage gate—do
not use the failed cells as training examples.

The metadata-only preparation command is:

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python \
  tools/prepare_oa13_oa33_cap_stroke_panel.py
```

This produces the minimal cov13-only panel under
`outputs/audits/chemvector-oa13-oa33-cap-stroke-panel-plan-20261017-r1/`
(128 train + 32 holdout identities, 3 profiles, 480 rows). If cov33 must be
diagnosed in the same paired batch, add `--include-cov33`; the r2 plan has 6
profiles and 960 rows. Both plans are manifest-only and mark every row
`render_status=not_rendered`.

## Explicit exclusions

This protocol does not revive pointwise selector gates, pairwise rescue,
coordinate refinement, predicted-port attachment, graph consistency, or MoE
routing.  It also does not use Molgrapher or sealed test identities.
