# r17 raw-vs-selected endpoint attribution

After correcting the audit to the production 16 px post-hoc match radius, a
frozen-initial replay was run on the m128 panel (GPU inference only; no
training, checkpoint write, or sealed-test read). CPU/GPU atom masks matched
on the parity sample (mean Jaccard `1.0`, mean nearest-point delta `0.00065`
px).

The result changes the diagnosis:

| split | target bonds | reachable with raw atoms | reachable with selected atoms |
|---|---:|---:|---:|
| train | 7,620 | 7,620 | 7,443 |
| dev | 3,752 | 3,752 | 3,641 |
| fresh | 7,540 | 7,540 | 7,351 |

At the 16 px contract, raw proposal matching reaches every target atom in
this panel. The missing endpoint reachability is therefore selector rejection,
not raw pixel proposal absence. Fresh selected-atom bond reachability is
`7,351/7,540 = 97.49%`; the remaining candidate-edge gate lowers it further
to 7,100/7,540 because the pair head does not include every reachable bond.

Selector losses are concentrated in the difficult style families (fresh
selected bond reachability: oa00 98.89%, oa11 96.66%, oa20 97.93%, oa33
96.50%). Visible O/N raw candidates are retained almost completely at the 16
px contract; the dominant loss is selected-vs-raw atom retention and then
pair candidate coverage.

Decision: the next bounded training experiment should target selector
retention with protected initial candidates and hard-negative controls. Do not
train another bond-type/rollback head until selector retention is measured on
the same clean dev/fresh panel.

Artifact:

`outputs/audits/chemvector-r17-raw-selected-endpoint-m128-r16/report.json`
