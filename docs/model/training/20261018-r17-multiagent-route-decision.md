# r17 multi-agent route decision (2026-10-18)

Three independent reviews were run before the next CPU experiment.

| route | evidence | decision |
|---|---|---|
| Continue element-context training | m64 element accuracy `.98984/.98750` (dev/fresh), compile `.8711/.9219`; oa20 remains weak | Freeze the head; no blind expansion or GPU |
| Graph-risk/abstention head | target-free overflow gate raised sanitize to `.9653/.9779`, but train conditional sanitize was only `.9006` with ~70% acceptance | NO-GO for training; keep only as fail-closed diagnostic |
| One-edge rollback transaction | dev `.8711→.9492`, fresh `.9219→.9531`; fresh acceptance/abstain gates pass | CPU shadow candidate, not production |

The rollback route uses frozen AtomObject/AtomPair candidates, aromatic/type
sidecars, and the frozen element-context head. It only runs on an invalid graph,
tries a single low-confidence edge removal on a deep copy, and commits only if
the compiler succeeds. Aromatic-invalid graphs abstain; accepted transactions
in the current m64 replay were valence-only. The 768-row target perturbation
sentinel produced byte-identical routing, compiler outcomes, attempts, and
transaction journals.

Artifacts:

```text
outputs/audits/chemvector-r17-target-free-rollback-m64-v2/report.json
outputs/audits/chemvector-r17-target-free-rollback-m64-v3-perturb/report.json
outputs/audits/chemvector-graph-risk-abstain-cpu-r17-m64-v3/report.json
```

The next bounded action is a read-only CPU shadow report exposing the rollback
journal and exact per-profile graph deltas. It must preserve the production
checkpoint, keep sealed data unread, and pass the existing candidate/edge
contracts. No GPU launch or checkpoint merge is authorized by this receipt.
