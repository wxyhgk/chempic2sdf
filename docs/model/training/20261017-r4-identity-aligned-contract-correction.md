# r4 identity-aligned O/N panel: contract correction

The r4 panel is diagnostic only and must not be admitted:

* the plan declares 48 px minimum anchor separation, but the rendered minimum
  is 42 px;
* 13/1,200 rows are rejected (seven molecule identities; all are O/N nitro or
  hydroxylamine contexts);
* `connection_identity` is `null` in the plan, so scaffold disjointness alone
  is not an identity-aligned split;
* the `cov33_*` rows use **DejaVu Sans**, not the target oa33 **Nimbus Roman**
  profile, so they cannot be used as an oa33 replay;
* strict raw unique O/N ownership is only 73/3,179 targets.  This is a model
  reachability failure, not evidence that more of the same clean images should
  be trained.

The rejected identities are fixed and must be blocked in the replacement
ledger: `curr_8bf888531e8e07ea`, `curr_a211e5da433f81c4`,
`curr_c7daf7874ec5a766`, `curr_49ecad6c8fdb31a3`, `curr_687aa0ab3a8015a1`,
`curr_7877bb11569934d6`, and `curr_a2f745ad7c6f5dc4`.

## CPU-only replacement contract

1. Build an immutable candidate ledger from v0.2 entries containing canonical
   isomeric SMILES, InChIKey, stereo/isotope-stripped connection identity,
   Murcko scaffold, O/N degree/bond/context strata, and source record IDs.
   `connection_identity` may not be null.
2. Keep every r4 rejected molecule in a blocked ledger.  Do not move it to the
   other split or silently replace only its failed profile row.
3. Render a CPU preflight for an oversampled pool under the exact intended
   profiles.  Measure post-layout atom-anchor separation and quality outcomes.
   A molecule enters a complete matrix only when **every** profile row passes;
   otherwise the entire molecule is excluded and the next deterministic
   fallback identity is considered.
4. Separate the geometry contract from the profile choice.  The cap28 cells
   systematically produce a 42 px minimum in this renderer, while cap52 cells
   produce 78 px.  Therefore either (a) use a newly named cap52 profile for a
   complete matrix and enforce `observed_min_anchor_separation_px >= 48`, or
   (b) keep cap28 only as a clearly labelled difficult diagnostic with an
   observed ≥40/42 px contract.  A declared 48 px number without a measured
   post-layout check is invalid; do not relabel the 42 px cap28 data as a 48 px
   panel.
5. Use exact profile IDs and tuples: oa13 control is DejaVu Serif/32/cap28/
   stroke3.2/180°; oa33 control is Nimbus Roman/32/cap52/stroke1.6/0°.
   Any cap40/cap52 repair is a new profile ID, never a relabeled control.
6. Select train/holdout by molecule, connection identity, and scaffold with
   zero cross-split overlap.  Reserve at least 3x the requested identities so
   renderer failures do not cause silent profile imbalance.

## Admission gates

The CPU artifact gate requires complete profile denominators, zero quality
rejections in the selected matrix, exact SDF/graph/layers round-trip, unique
PNG/RGB bytes, non-null identity keys, and zero molecule/connection/scaffold
overlap.  The panel remains quarantine until a frozen initial replay shows
profile-level raw visible-text O/N recall >=0.98 on holdout.  Report strict
unique ownership separately; do not convert ambiguous targets to negatives.

The current 73/3,179 strict unique ownership result and the known oa13/oa33
raw text ceilings fail this replay gate.  Therefore r4 should be used only to
repair the data contract and select a clean, identity-aligned diagnostic
matrix; it is not a training release and does not justify GPU training.
