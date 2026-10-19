# Bond-supported selector-drop attribution (2026-10-06)

After the independent selector sidecar failed its precision gate, a read-only
replay checked whether selector-dropped implicit carbons are supported by the
frozen bond endpoint detector.  The replay used the same immutable `initial.pt`
and the identity-disjoint 480/720 split; it did not train or alter decoding.

Among raw-matched but selected-dropped implicit carbons:

| split | drops | any predicted endpoint within 12px | strict bond identity support |
|---|---:|---:|---:|
| train | 76 | 74 | 24 |
| holdout | 101 | 100 | 36 |

Strict support requires both predicted endpoints within 12px of a ground-truth
bond and the predicted bond type to match.  Most candidates have geometric
endpoint evidence, but only 24/76 train cases have strict identity support.
The pre-registered minimum of 50 train positives is therefore not met, so a
new selector trial is **not admissible**.  The route is frozen; no threshold or
step sweep is justified.

Machine report:
`outputs/audits/chemvector-bond-supported-selector-drops-20261006-r1/report.json`.

The next useful work is data/rendering coverage, targeted to `font_large` and
`serif` visible O/N labels and implicit-C placement around single/double/triple
and aromatic bonds.  Any future GPU trial must start with a fresh read-only
identity-disjoint preflight.
