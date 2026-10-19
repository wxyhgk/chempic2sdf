# oa13/oa33 raw-reachability coverage repair (metadata-only plan)

## Diagnosis

The 10k frozen replay is not training-admissible because the failure is
concentrated in two render cells:

| profile | renderer factors | raw reachability | raw gaps |
|---|---|---:|---:|
| oa13 | DejaVu Serif, 32 px, 28 px bond cap, 3.2 px stroke, 180° | 93.412% | 1,012 / 15,362 |
| oa33 | Nimbus Roman, 32 px, 52 px bond cap, 1.6 px stroke, 0° | 91.922% | 1,241 / 15,362 |

The other six cells are 99.52--99.90% raw reachable.  Therefore a global
selector retrain or a larger undifferentiated image pool is not the minimum
experiment.  The shared suspected factor is the large serif glyph regime;
rotation, bond length, and stroke width must be isolated rather than changed
simultaneously.

## Fixed diagnostic panel

Do not change the renderer implementation.  Add only profile metadata entries;
each entry must include `parent_profile_id` and exactly one `changed_factor`.
The base profile remains unchanged and is rendered in the same batch as its
controls.

| profile id | parent | changed factor | fixed values |
|---|---|---|---|
| `cov13_base` | oa13 | none | exact oa13 tuple |
| `cov13_size24` | oa13 | `font_size` | 24 px; all other oa13 values fixed |
| `cov13_rot0` | oa13 | `rotation_degrees` | 0°; all other oa13 values fixed |
| `cov13_sans32` | oa13 | `font_family` | DejaVu Sans; 32 px; all other oa13 values fixed |
| `cov33_base` | oa33 | none | exact oa33 tuple |
| `cov33_size24` | oa33 | `font_size` | 24 px; all other oa33 values fixed |
| `cov33_rot180` | oa33 | `rotation_degrees` | 180°; all other oa33 values fixed |
| `cov33_serif32` | oa33 | `font_family` | DejaVu Serif; 32 px; all other oa33 values fixed |

The panel intentionally does not alter bond cap or line width.  Those factors
are not the first repair target because oa13 and oa33 fail in opposite cap and
stroke regimes.  If all three large-serif controls fail, only then prepare a
second panel for cap/stroke isolation.

## Small identity-disjoint batch

Select 320 fresh molecule identities after the normal complete prior-use,
connection, and scaffold ledger: 256 train identities and 64 holdout
identities.  Render all eight panel cells for every identity (2,560 images).
Use the same canonical structure for all paired cells; no image-level split.
Stratify identities before rendering so each split contains visible N/O,
S/P/halogen/Si/B, aromatic/double/triple, branching/ring junction, and
implicit-carbon cases.  Keep at least 25% visible-text and 10% multiple-bond
cases in each split.  This is a coverage diagnostic and remains quarantine
until the gates below pass.

Required metadata per row:

```json
{
  "sample_key": "<molecule_id>__<profile_id>",
  "parent_profile_id": "oa13",
  "changed_factor": "font_size",
  "profile_id": "cov13_size24",
  "molecule_id": "...",
  "split": "train|holdout"
}
```

The profile tuple, renderer/font version, source SDF hash, PNG hash, and
canonical connection identity are committed before replay.  A failed render
cannot be replaced by another factor or silently moved between splits.

## Frozen replay and stop gates

Run the production `initial.pt` exactly once, with no gradients, selector
updates, threshold sweep, or checkpoint write.  Report each cell separately
and retain the original oa13/oa33 controls.

1. Renderer integrity: exact SDF identity, atom/connection roundtrip, no glyph
   or attributed-ink clipping, and complete profile denominator.
2. Raw coverage: every control and candidate cell must reach at least 0.98 on
   the identity-disjoint holdout; report target counts and Wilson intervals,
   not only aggregate percentages.
3. Factor attribution: select the smallest changed-factor cell whose holdout
   raw reachability is at least 0.98 and whose paired base does not regress.
4. Safety: six unaffected historical cells must remain within 0.5 percentage
   points of the 10k replay; initial checkpoint SHA and old replay remain
   unchanged.

If no single-factor cell reaches 0.98, stop.  Do not train the selector and do
not lower the profile gate; prepare the second cap/stroke panel only.  If one
cell passes, admit only that renderer profile for a later data release.  This
experiment itself still does not authorize GPU training.

## Next implementation action

The metadata-only preparation is now executable:

```bash
CUDA_VISIBLE_DEVICES='' .venv-train/bin/python \
  tools/prepare_oa13_oa33_coverage_panel.py
```

It writes `outputs/audits/chemvector-oa13-oa33-coverage-panel-plan-20261017-r3/`
with `profiles.json`, `manifest.json`, and `report.json`.  The generated plan
contains 256 train identities + 64 holdout identities (2,560 rows), with zero
molecule, connection, or scaffold overlap.  All rows are marked
`render_status=not_rendered`; this command does not create images or run a
model.  Run renderer-quality validation and then the one frozen replay only
after reviewing this manifest.  Keep `initial.pt`, selector/OCR, and all
production profile IDs immutable.
