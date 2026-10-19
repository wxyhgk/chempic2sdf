# r17 m128 endpoint reachability result

The m128 cache was audited after the local-edge and transaction shadows. This
audit uses the frozen candidate mapping only; it does not train or alter any
model. Target bonds are joined post-hoc.

| split | target bonds | reachable | unreachable | unreachable rate |
|---|---:|---:|---:|---:|
| train | 7,620 | 7,176 | 444 | 5.83% |
| dev | 3,752 | 3,510 | 242 | 6.45% |
| fresh | 7,540 | 7,100 | 440 | 5.84% |

The failure is strongly style-dependent on fresh: oa00 1.38%, oa11 4.51%,
oa20 2.12%, and oa33 15.33%. This confirms that the remaining ceiling is
endpoint visibility/selection, not another aromatic or compiler head.

The current pair/local-edge sidecars cannot recover these bonds because their
candidate edge set already excludes them. The next admissible training
direction is therefore a raw/selected endpoint coverage experiment stratified
by oa33/serif and visible O/N, with protected-candidate retention and a frozen
graph replay. No pair/type/rollback head should be expanded before that gate.

Artifact:

`outputs/audits/chemvector-r17-m128-reachability-v1/report.json`
