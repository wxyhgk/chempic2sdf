# cov13_rot0 renderer-only repair probe

The coverage-panel render produced ten `cov13_rot0` quality rejections (eight
typography-clearance failures and two degenerate auxiliary junctions).  The
same ten fixed identities were retried in isolation without writing samples or
changing the production profile.

| isolated variant | changed factor | accepted / 10 | decision |
|---|---|---:|---|
| base control | none | 0 | NO-GO |
| cap 40 px | bond length cap | 10 | PASS |
| cap 52 px | bond length cap | 10 | PASS |
| cap 64 px | bond length cap | 10 | PASS |
| stroke 1.6 px | line width | 2 | NO-GO |
| stroke 2.4 px | line width | 0 | NO-GO |

This provides two classes of repair options:

1. A separately named cap-isolated profile (40 px is the smallest tested
   passing cap) restores the complete denominator for these ten rows while
   preserving font, size, rotation, and stroke.
2. A rotation fallback can use the existing `cov13_base` tuple (180 degrees),
   which is already complete in the panel.  This is a factor substitution,
   not a repair of `cov13_rot0`; it must retain its distinct profile ID and be
   evaluated as such.  The size24/sans32 controls are useful diagnostics but
   are not complete repairs for these ten rows.

Changing stroke width is not a viable repair: both tested values leave
   quality failures.  The original `cov13_rot0` control remains NO-GO and must
   not be relabeled with the passing cap.  A full-panel rerender is required
   before any admission decision; this probe alone authorizes no training.

Machine-readable output:
`outputs/audits/chemvector-cov13-rot0-renderer-repair-20261017/report.json`.
