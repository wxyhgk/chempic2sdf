# Aromatic specialist data plan (r17)

This is a quarantine-only data plan.  It does not admit data to training, read
the sealed test, alter the production checkpoint, or start a GPU job.

## Why a new split is needed

The existing r16 probe used the frozen AtomObject features and had no aromatic
positive in `fresh_holdout` (95 images).  Its dev aromatic accuracy was only
`0.79932` and positive recall `0.79412`; consequently a perfect chemistry
score on the previous fresh set cannot establish aromatic generalization.

The current structure-expansion quarantine source is a better source than
adding style copies to the old split:

| source statistic | count |
|---|---:|
| molecules / accepted images | 1,534 / 12,184 |
| molecules containing aromatic bonds | 849 |
| Murcko scaffold groups containing aromatic bonds | 412 |
| groups with aromatic C / N / S | 412 / 385 / 171 |
| fused-ring groups | 262 |
| non-aromatic control groups with DOUBLE / TRIPLE | 216 / 81 |

The source is still quarantine and its external identity status is unchecked;
these counts are capacity, not admission.

## Fixed selection proposal

Use one representative per group.  A group is the stereo/isotope-stripped
Murcko scaffold for cyclic molecules, and the stereo/isotope-stripped
connectivity for acyclic molecules.  All render profiles of one representative
remain in the same split.

Select 448 representatives with a deterministic SHA-256 tie-break:

| split | groups | profiles | planned images |
|---|---:|---:|---:|
| train | 256 | 4 | 1,024 |
| dev | 64 | 4 | 256 |
| fresh_holdout | 128 | 4 | 512 |

The four profiles should be the already implemented clean profiles `oa00`,
`oa11`, `oa20`, and `oa33`; no new renderer or post-render resize is allowed.
Feature quotas are minima and categories may overlap:

| feature | train | dev | fresh |
|---|---:|---:|---:|
| aromatic C | 64 | 16 | 32 |
| aromatic N | 64 | 16 | 32 |
| aromatic S | 32 | 8 | 16 |
| fused ring | 64 | 16 | 32 |
| heteroaromatic | 64 | 16 | 32 |
| non-aromatic SINGLE control | 48 | 12 | 24 |
| non-aromatic DOUBLE control | 48 | 12 | 24 |
| non-aromatic TRIPLE control | 32 | 8 | 16 |

Controls are complete molecules, never isolated line glyphs.  For each split,
keep at least 25% control representatives and at least 2 different heavy-atom
size buckets.  Do not balance by image count alone.

## Identity and leakage checks

Before rendering, the planner must fail closed if any candidate has a canonical
or connectivity match with the existing r6 targeted data or v0.2 curriculum.
After selection it must assert all of the following:

1. train/dev/fresh have zero representative-group overlap;
2. no molecule appears in more than one split;
3. all four profiles exist for every selected molecule;
4. no duplicate PNG/RGB bytes within a split or across different identities;
5. graph/SDF canonical isomeric SMILES and atom/bond index identity roundtrip;
6. aromatic labels are graph-derived (`GetIsAromatic` / `AROMATIC`), while
   non-aromatic DOUBLE/TRIPLE controls are explicitly counted;
7. no source image, sealed artifact, or production checkpoint is opened.

The metadata planner currently materializes this proposal at
`outputs/audits/chemvector-aromatic-specialist-plan-r17-v2/` (the source
payloads remain untouched).  The final release output should contain
`plan.json`, `rows.jsonl`, `source-sha256.json`,
`coverage.json`, and a receipt with `training_authorized:false` and
`gpu_used:false`.  Until external identity admission is separately authorized,
the rows must remain quarantined and must not be visible to a training loader.

## CPU gate before any training

Using only the frozen initial checkpoint (`09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1`),
run a small shared aromatic-attribute sidecar on train rows.  Keep AtomObject,
OCR, and pair presence frozen.  Thresholds are selected from train only.  The
sidecar is admissible for a later shadow only if all are true on both dev and
fresh:

- aromatic C/N/S macro recall >= 0.90;
- non-aromatic specificity >= 0.95;
- per-profile aromatic recall spread <= 0.05;
- edge-set signature is bitwise unchanged from the frozen route;
- checkpoint SHA is unchanged and target-free sentinel passes.

If the CPU gate fails, increase motif coverage or change the head objective;
do not spend GPU on the same data.  If it passes, a separate authorization is
still required for one bounded sidecar-only GPU shadow; this plan never grants
that authorization.
