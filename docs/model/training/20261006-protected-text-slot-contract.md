# Protected text-endpoint slot contract (2026-10-06)

The current AtomObject output can technically support an append-only slot
interface without changing the production 128 selected prefix.  The audit
`tools/audit_protected_text_slots.py` ran on all 1,200 public cohort images and
produced
`outputs/audits/chemvector-protected-text-slots-20261006-r4/report.json`.

Contract used by the audit:

1. Run the unchanged current-initial forward and copy `selected` rows exactly
   as prefix `B`.
2. Set residual capacity to `128 - len(B)`; no image in the cohort filled all
   128 selected rows (selected count range 1--65, mean 30.15).
3. Select dense cells target-independently from
   `heatmap_probability * TEXT_role_probability`, using a 4,096-cell pool,
   excluding cells within 12 px of `B` and suppressing sidecar duplicates within
   6 px.
4. Append at most K residual rows (`K ∈ {1,2,4,8,16}`), sampling the existing
   object feature, anchor, role, text center and extent at that cell.
5. Assign baseline target IDs first.  Residual candidates may claim only target
   IDs outside the immutable baseline set, at the existing 6 px matching radius.

Shape/ownership results:

- prefix tensors were bitwise identical in 1,200/1,200 images for every K;
- maximum prefix geometry delta was 0;
- the existing set head accepted the appended tensors in 1,200/1,200 images;
- all set-head tensors were finite in every appended replay;
- production capacity/checkpoint remained unchanged.

One additional compatibility caveat is material: adding residual rows changes
the set-head atom-presence context because its incidence softmax is normalized
over *all* atom rows.  In the replay, the appended tensors were accepted and
finite, but the old-prefix atom-presence logits changed (maximum absolute
delta 9,999.75; one threshold flip across the K grid).  Therefore simply
passing `B + R` through the existing selector is not ownership-safe.  A real
protected replacement must either evaluate the baseline prefix in a separate
selector call, or mask/freeze the baseline decision before residual rows enter
the incidence normalization.

Quality results were decisively negative.  There were 505 residual targets in
the cohort, but text-ranked sidecar positives were only 9 total:

| K | candidates | positives | precision | residual recall |
|---:|---:|---:|---:|---:|
| 1 | 1,200 | 9 | 0.75% | 1.78% |
| 2 | 2,400 | 9 | 0.38% | 1.78% |
| 4 | 4,800 | 9 | 0.19% | 1.78% |
| 8 | 9,600 | 9 | 0.09% | 1.78% |
| 16 | 19,200 | 9 | 0.05% | 1.78% |

Thus the append-only interface is compatible, but the current text role score is
not a usable residual candidate selector.  Do not train a scorer on these
labels or deploy this sidecar.

Required gates for a future trained multi-slot text endpoint:

- prefix retention 100% (bitwise and selected-ID ownership);
- `set_head` shape/finite checks 100%;
- at least 50 train positives;
- train AUPRC >= 0.50 and precision >= 0.85;
- holdout locked residual precision >= 0.85 and at least 14 new matches;
- zero regression in exact/valid/fallback metrics;
- current-initial checkpoint and sealed identities unchanged.

The next useful work is to train or redesign a *new* text-endpoint objectness
signal while preserving this prefix contract.  Increasing K, changing the
global rank, or reusing the current role score is not justified.
