# Owner Candidate Domain v0.5 预注册

日期：2026-08-15

## 研究问题

`Intersection Ownership v0.4` 已证明：同一份 `geometry root + parent supports + overlap universe` 可以允许两份
合法、但 bridge pixel owner 不同的 world。因此本周期不再寻找唯一 owner，而是验证一个有限、可审计的集合值逆
映射：

```text
observation y -> candidate fiber F(y)
```

这里的 `F(y)` 不是“模型最喜欢的答案”，而是当前证据尚未排除的解释集合。只有 fiber 为 singleton 时，后续阶段
才有资格讨论单答案；多解必须保持 `Ambiguous`，空集必须 closed reject。

## 两层对象

必须严格区分：

1. `evidence membership`：一个 pixel 可以同时属于多个 parent carrier，并在每个 carrier 内确定性属于一个 atomic
   fragment。它是多值 provenance relation，不要求全局唯一。
2. `semantic consumer hypothesis`：intersection、atomic fragment 或 residual/reject 对该 pixel 的有限候选域。它是
   observation-relative hypothesis，不是真实 PNG owner 声明。

禁止再次把 raster 合成的多 contributor pixel 强制压成一个视觉真 owner。

## 冻结构造规则

对每个 cross-carrier overlap pixel：

1. 保留其完整 carrier fiber 和全部确定性 fragment memberships；
2. 对 intersection `I` 构造
   `eligible(I) = {p | incident_carriers(I) subseteq carriers(p)}`；
3. 从 `I` 的 exact anchor 出发，只保留 `eligible(I)` 中的 8-connected reachable component；
4. `p` 属于该 reachable component 时，`I` 才进入 `p` 的 intersection option；
5. fragment option 逐项来自已经验证的 child support membership；
6. residual 只能作为明确的保守 over-approximation 轴，不得冒充已经观测到的对象；
7. exact-one、seed ownership、8-connectivity 和 incidence 是声明式 CSP constraint，不用最近 seed、Voronoi、ID
   precedence、top-k、score 或 RDKit sanitize 偷选 winner。

若两个 intersection 同时可达 bridge pixel，两个 option 必须同时保留。ID 和排序只服务 canonical serialization。

## 输出边界

研究 schema 暂定：

```text
chemgraph-owner-candidate-domain-v0.5-research
```

顶层状态只能是：

```text
candidate_domain
reject
```

始终 `training_authorized=false`。禁止出现 `winner`、`unique`、`accepted`、`preferred_option` 或任何 SDF/Graph。
输出使用 factorized CSP：owner option、去重后的 domain class、pixel variable、8-neighbor adjacency 和 constraints 分开
存储，不物化指数个 world。

## 非平凡性门

仅仅给每个 pixel 放入所有 intersection、所有 fragment 和 residual，可以平凡得到 coverage，但没有信息价值。必须同时
报告 universal-domain baseline，并冻结以下指标：

```text
oracle coverage
mean / max domain size
eliminated-option ratio vs universal domain
singleton-variable ratio
multi-intersection-variable ratio
residual-only / residual-containing ratio
factorized assignment count or certified lower bound
```

若 coverage 为 100%，但 option elimination 为 0，或所有变量都因无条件 residual 保持多解，则裁决为
`SAFE_BUT_VACUOUS`，不是 GO。

## 预注册反例

- bridge witness：中间 pixel 必须同时含左右两个 intersection option，World A/B 都在解域；
- connected far snake：不得被强制归入唯一 intersection；
- disconnected far overlap：不能凭 carrier incidence 假装已被 intersection 解释；
- double-seed component：保持多候选或 reject，不按 seed/ID 拆分；
- partial three-carrier fiber：逐 pixel carrier fiber 不得被 region union 掩盖；
- diagonal corner：明确记录 8-connectivity 的合并边界；
- glyph-like overlap：没有独立 text evidence 时保持 residual/ambiguity；
- shallow angle、border、anchor collision、T/X、三/四 carrier 同点：分别冻结预期状态；
- carrier/intersection 重命名、列表 shuffle：输出集合 up-to-bijection 不变；
- 平移、90 度旋转、反射：结构计数与 domain 关系等变。

## GO / NO-GO

构造器和独立 verifier 必须满足：

1. verifier 不 import constructor 的 domain helper，从 source 重算全部 section；
2. finite micro-scene 穷举中，CSP expansion 对预注册 legal-world predicate sound、complete、Nodup 均为 100%；
3. bridge 两份 world 都被覆盖，任何单 winner 输出均失败；
4. hidden renderer contributor/owner 仅用于 coverage 审计，不能用于 constructor 选答案；
5. 所有容量 exact 成功、`+1` closed reject，禁止 truncate；
6. 1000 次重放 byte exact，并发顺序不改变 canonical bytes；
7. source、section hash、stage trace 和冻结 artifact trust anchor 全部可独立重建；
8. 非平凡性指标必须优于 universal baseline，否则只能得到 `SAFE_BUT_VACUOUS`。

本周期即使 GO，也只授权 research candidate-domain constructor。以下继续 NO-GO：

- PNG -> parent support completeness；
- candidate domain 与真实 PNG 语义 fiber 完全相等；
- UniqueGraph、SDF；
- 模型训练。

## 与现有 Lean 结果的关系

`StructuredRecovery.lean` 已证明 exact candidate set 必须保留 observation fiber 中的多个非等价解释；
`SymbolicCandidateFamilyV04.lean` 已证明 8 个独立二元歧义显式展开为 256 个 assignment，而 factorized domain 可用
8 个变量域表达。本周期不重复这些抽象定理，只对 intersection/fragment/residual 的有限实例补充 coverage、singleton
降级和非平凡性结论。

这仍是条件证明：如果 legal-world predicate 或 renderer oracle 与真实化学图片不一致，形式化通过不能自动升级为
真实 PNG 可用。
