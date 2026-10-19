# r17 oa13/oa33 known-clean graph stress: 16 px owner + NMS

## Scope

This is a frozen, read-only graph-downstream stress audit.  It uses 16
holdout molecules (32 images: oa13 and oa33), selected from the public release
manifest with zero molecule overlap with every identity in the known-clean
r17 m128 sidecar cache.  No sealed-test payload is read and the production
checkpoint is not written.

Artifacts:

- metadata slice (superseded pilot): `outputs/audits/chemvector-r17-oa13-oa33-known-clean-stress-manifest-v1/`
- no-NMS owner replay: `outputs/audits/chemvector-r17-oa13-oa33-known-clean-stress-graph-replay-v1/report.json`
- 8 px NMS replay: `outputs/audits/chemvector-r17-oa13-oa33-known-clean-stress-graph-replay-nms8-v2/report.json`
- target-perturbation sentinel: `outputs/audits/chemvector-r17-oa13-oa33-known-clean-stress-graph-replay-nms8-v2-perturb/report.json`

The v1 slice was cache-disjoint but overlapped earlier slice/artifact
identities.  The stricter authoritative replay is v2 of the manifest and v3
of the replay, which are disjoint from the m128 cache, known-clean graph
artifact, and slice80:

- `outputs/audits/chemvector-r17-oa13-oa33-known-clean-stress-manifest-v2/`
- `outputs/audits/chemvector-r17-oa13-oa33-known-clean-stress-graph-replay-nms8-v3/report.json`
- `outputs/audits/chemvector-r17-oa13-oa33-known-clean-stress-graph-replay-nms8-v3-perturb/report.json`

An independent target-free artifact receipt was also exported and validated:

- artifact: `outputs/audits/chemvector-r17-oa13-oa33-stress-graph-artifact-nms8-v1/`
- validator: `outputs/audits/chemvector-r17-oa13-oa33-stress-artifact-bound-v2/report.json`

The artifact binding gate passed (exact SHA, manifest, schema, and no target
fields).  Its endpoint mapping rates were `0.9913` (oa13) and `0.9957`
(oa33); unmapped bonds were `1` and `0`, respectively.  This artifact export
does not apply the safe compiler transaction, so it is an interface receipt,
not a production graph result.

The compiler uses presence-admitted edge context, with a fixed 16 px owner
exclusion around the selected prefix.  NMS is score-only and target-free.

## Result

On the stricter 32-image holdout, the 16 px owner removed 5 of 13 sidecar
candidates; 8 px NMS kept 8 candidates.  Safe compile accepted 4.  The
baseline and augmented graph had 5 unmapped edges in both cases.  Augmentation
changed strict typed recall from `0.743802` to `0.750000`, edge precision from
`0.988506` to `0.988584`, and sanitize rate remained `0.90625`.

Normal and target-perturbation decision digests are identical:

`4b1d2a097a289b8c96f5b69d09d49ad3c19e7d84de19657307a7c3fa9531a844`

This proves owner/NMS and safe-compile routing are target-independent on the
stress slice.  It does not prove full molecular identity.

## Gate

- Shadow gate: **PASS** — prefix retained, recall/precision/sanitize do not
  regress, and target perturbation leaves decisions unchanged.
- Production/merge gate: **NO-GO** — owner/NMS does not reduce existing
  baseline unmapped edges; only 32 images were replayed; compiler context
  scope still requires the separate contract fix and a larger clean panel.

Next action is a larger identity-clean replay after the context-scope fix;
do not train or modify the production decoder from this audit.
