# Symbolic Candidate Family v0.4 容量实验结果

日期：2026-08-15

## 裁决

本周期只授权提交研究结论：

```text
v0.3 Boundary matching completeness: GO
v0.3 interpretation completeness:     NO-GO
v0.4 symbolic capacity hypothesis:     GO
v0.4 production symbolic verifier:     NO-GO
public proposer / UniqueGraph / SDF:    NO-GO
training:                              NO-GO
```

这里的 `symbolic capacity hypothesis: GO` 只表示 factorized finite-domain CSP 能表示超过 128 项的有限候选族，
不表示 root constructor、CSP 语义或真实图片候选已经 sound 且 complete。

## 可执行结果

| 检查 | 结果 |
|---|---:|
| v0.3 completeness 专项测试 | 13/13 |
| v0.4 symbolic family 专项测试 | 13/13 |
| `formal/chemgraph/audit` 全量测试 | 183/183 |
| Lean 构建 | 28/28 jobs |
| matching 独立暴力复核 | `n <= 5` 的 1,099 个简单图全部一致 |
| counterexample 重放 | 5/5 逐字节一致 |

8 个独立二元变量的实验结果为：

```text
7 variables -> 128 solutions
8 variables -> 256 solutions
materialized capacity 128 -> overflow
factorized representation -> 8 Boolean domains
```

Lean 同时证明 8 个二元域展开为 256 个无重复 assignment，完整显式列表无法满足 128 项容量，而 factorized
family 可展开为同一解族。该证明没有假设真实 PNG 中存在 8 个独立歧义，也没有证明 256 个 assignment 会编译成
256 个不同的化学图。

## v0.3 完整性边界

独立 Python verifier 不信任 `complete=true` 或 `all_candidates_emitted=true`，而是从 root 的 Boundary 和
compatible pair 重建全部端点互斥 matching。它可以区分：

- missing/extra matching：`BOUNDARY_MATCHINGS_INCOMPLETE`；
- matching 集合相同但 index 或顺序非规范：`BOUNDARY_MATCHINGS_NONCANONICAL`；
- matching 完整但 interpretation 域不可证明：
  `BOUNDARY_MATCHINGS_COMPLETE_INTERPRETATIONS_UNPROVABLE`。

`full_collision.json` 与删掉一个 alternative 后的 `omitted_alternative.json` 具有完全相同的 root projection，
且都通过 v0.3 结构 Validator，但 interpretation 数分别为 2 和 1。因此只读 v0.3 root 的 verifier 不可能对
真实 interpretation 域同时 sound 且 complete。

Lean matching exact 定理的前提是 root 已提供完整 compatible-pair list；它证明的是“给定该有限域后的枚举完整”，
不证明 pair proposer 对真实 PNG 已经完整。Lean 中的通用 same-root NO-GO 定理与抽象 witness证明逻辑不可能性；
上面的真实 v0.3 schema 反例证据来自 Python 构造器、冻结结构 Validator 和逐字节重放，不能把两类证据混写。

## v0.4 探针边界

当前 `symbolic_family.py` 是容量探针，不是 production verifier。它严格拒绝 `bool`、浮点数和字符串冒充整数
assignment，也会拒绝非 closed、非 canonical 的变量和 table constraint 文档。

但是它仍接受调用者自报的 domain 和 constraint。红队确认以下两种变异都会被研究探针当作另一个合法 family：

```text
domain_size: 2 -> 1          256 solutions -> 128 solutions
add unit constraint x0 = 0   256 solutions -> 128 solutions
```

因此未来 production verifier 必须从 root 逐字重建 variables、domains 和 hard constraints，并把上述两种变异
判为 `INVALID`。不能将当前 Python 文件改名后直接接入生产流程。

## 证明与实现哈希

```text
CarrierCompletenessVerifierV03.lean:
39275cb6458bb87f650768735ad2f402da4c82945cd72e2d869eb072bc552208

SymbolicCandidateFamilyV04.lean:
4ab359397aafb142971689f60ac0fa472f7c31d8c8424964261105240a5669ba

symbolic_family.py:
c0eb844e6fb3c6449210fa6dcee28224296d9fed186acff7c0987ea43f256276
```

## 下一门槛

下一周期只做 v0.4 root constructor 微场景，按低层到高层确定性重建：fragment split、intersection、Atom
instance、endpoint attachment、canonical port partition、ownership、lane flow/rank 和 Bond grouping。每层先冻结
closed root universe，再证明：

```text
Legal(root, I)
<-> exists total assignment a,
      Sat(CompileCSP(root), a)
      and CanonicalCompile(root, a) = Canonicalize(I)
```

双向证明和独立红队都通过之前，不进入训练。
