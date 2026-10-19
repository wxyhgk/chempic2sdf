# 2026-10-06 valence guard public replay (parity-fix)

This is a read-only replay on the fixed public validation cohort. It reads
the initial checkpoint and the rejected `joint-000100.pt` delta only; it does
not read sealed data, train, write checkpoints, or alter deployment.

## Why r2 was invalid

`tools/audit_valence_guard_public.py` r2 compiled the edited scene directly
with `mol_from_chemscene_v0_2`. Production `FrozenOCRPredictor.from_structure`
uses the same first compile, but on a sanitization/kekulization failure it
calls `compile_carbon_placeholder`. Omitting that fallback made unchanged
baseline scenes look like new failures (`690 -> 639` valid). The audit now
replays the fallback after the in-memory guard edit.

## Fixed public result (r3)

| arm | images | original exact | original valid | guarded exact | guarded valid | removed edges | new fallback | broken exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 944 | 233 | 690 | 233 | 690 | 0 | 0 | 0 |
| joint-100 | 944 | 233 | 689 | 234 | 690 | 1 | 0 | 0 |

The only edit is in `fresh_dev:fresh_dev_001966_coordgen`: remove aromatic
edge `(62,40)`. Atom 40 is neutral explicit-H oxygen (`OH`) and already has
one heavy-atom edge; accepting a second aromatic heavy edge gives explicit O
valence 3. After removal, the prediction is valid and canonical-exact. The
guard changes no initial result and rescues the single joint regression.

Command:

```bash
PYTHONPATH=. CHEMVECTOR_VALENCE_GUARD_OUTPUT=outputs/audits/chemvector-valence-guard-public-20261006-r3 \
  python tools/audit_valence_guard_public.py
```

Machine-readable output:
`outputs/audits/chemvector-valence-guard-public-20261006-r3/report.json`.

This is evidence for an audit-only deterministic safety gate, not permission
to deploy it or resume pair-head training. Broader element/valence pruning
was not evaluated here and should not be inferred from this one-case rescue.
