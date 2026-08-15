# Owner Candidate Domain v0.5 R0

本目录实现 commit `5a8bdd9` 冻结的 known-intersection R0 profile：

```text
annotation-free observation -> factorized owner candidate CSP
```

R0 的 concrete owner 只有两个 geometry intersection。Atomic fragment 只作为每个 carrier 内唯一的 evidence
membership 输出，不升级为 owner；`residual_reasons=[]`，任何 unowned overlap 都整份拒绝。Constructor 不读取两份
stored annotation，也不计算 constrained solution count。

当前 artifact 有 58 个 pixel variable、48 个 singleton、10 个双域、188 条 8-neighbor edge 和 7 条全局约束。
raw Cartesian product 为 1024。`873` 不出现在 constructor 或 artifact 中，必须由独立 `LegalR0` verifier 得到。

`verify_r0.py` 不 import constructor、observation builder 或 shared common helper；它从冻结 renderer 重建 observation、
domain、edge、constraint 和 `LegalR0` truth table，再分别展开 pointwise product 与 candidate CSP。冻结结果为
`1024 raw / 873 legal / 873 candidate`，sound、complete、Nodup 均通过，false singleton/multi-option 都为 0。

`run_replay.py` 已完整执行 1000 次顺序 constructor 重建和 32 次 8-worker 逆序重建，唯一 artifact digest 数为 1。

运行：

```bash
.venv-train/bin/python -m \
  formal.chemgraph.audit.owner_candidate_domain_v0_5.build_artifacts

.venv-train/bin/python -m unittest \
  formal.chemgraph.audit.owner_candidate_domain_v0_5.test_candidate_domain -v

.venv-train/bin/python -m \
  formal.chemgraph.audit.owner_candidate_domain_v0_5.verify_r0 --write

.venv-train/bin/python -m \
  formal.chemgraph.audit.owner_candidate_domain_v0_5.run_replay --write
```

本目录尚未授权 fragment owner、residual concretization、UniqueGraph、SDF 或训练。
