# Carrier Candidate Set v0.3 独立完整性 NO-GO

日期：2026-08-15

## 结论

冻结的 v0.3 能验证候选文档内部结构，但不能从根级记录独立重建全部 interpretation 候选域。因此：

```text
v0.3 boundary-matching completeness: 可验证
v0.3 interpretation completeness:     NO-GO
v0.3 public proposer execution:        NO-GO
v0.3 unique graph / SDF:               NO-GO
training:                              NO-GO
```

不能用外挂代码偷偷补充缺失 universe；那等于修改协议，应显式升级 v0.4。

## 可重建矩阵

| 域 | 从 v0.3 root 独立重建 | 证据 |
|---|---|---|
| Boundary matching | 可以 | 对 compatible-pair 子集枚举，并过滤端点互斥；unmatched 是唯一补集 |
| Direct pair | 可以 | active Atom proposal 的全部无序二元组合 |
| Fragment | 不可以 | split、support、width、primitive 只在 interpretation 内出现 |
| Intersection | 不可以 | region、support、center、kind 没有根级 universe |
| Ownership | 不可以 | owner kind/ID 由 interpretation 自报，部分 owner ID 没有根对象 |
| Atom 实例 | 不可以 | proposal 可重算 active，但实例子集、角色和 glyph fiber 自报 |
| Endpoint attachment | 不可以 | 给定端点可验几何，无法发现被省略的 alternative |
| Port | 不可以 | endpoint partition、center、radius 没有 canonical root constructor |
| Lane | 不可以 | fragment chain、方向和 gap 组合是局部自报 |
| Bond | 不可以 | 给定 lane 可验 order，不能重建全部 lane grouping |
| Interpretation | 不可以 | hash/count 只覆盖提交者已经保留的列表 |

## 决定性反例

### 同 root、不同图类

从公开 glyph/Bond collision fixture 构造两份文档：

1. `full_collision.json` 保留 glyph 空图与 Bond 图；
2. `omitted_alternative.json` 删除 glyph alternative，随后重算 ID、graph class、enumeration hash、空 taxonomy 和
   12-stage trace。

两份文档：

- PNG、mask、Boundary、compatible pair、Atom proposal、Direct pair 和固定阈值完全相同；
- 都通过 frozen v0.3 结构 Validator；
- root projection SHA256 相同；
- interpretation 数分别为 2 和 1。

因此任何只读取 v0.3 root 的 verifier 对两者输入完全相同，却必须对不同的“真实完整域”给出相反答案。
它不可能同时做到 sound 和 complete。

这个论证不否定“输出所有 schema-legal interpretation”的超集 completeness；但该集合包含大量无观测依据项，
不 sound，而且立即超过 128 容量。NO-GO 针对的是可用于唯一图决策的 exact sound+complete certificate。

### 漏掉 all-unmatched matching

`paired_only_matching.json` 公布两条 compatible parallel Boundary，只提交 paired matching，删除合法的
all-unmatched matching，并刷新全部自签字段。结构 Validator 仍通过；独立 matching verifier 重建 2 个
matching，报告：

```text
BOUNDARY_MATCHINGS_INCOMPLETE
missing_matching_count=1
```

这证明自签 `complete=true` 和 hash 只能证明内部一致，不能证明枚举完整。

## 数学边界

Lean v0.3 的 complete 定理依赖预先给定的 `RootedInterpretationUniverse.complete`。新 completeness 模块需要证明：

1. compatible-pair 的全部端点互斥 matching 可由有限枚举器 sound/complete 地生成；
2. unmatched Boundary 是选中 pair 端点集合在 root Boundary 中的唯一补集；
3. 若两个 world 投影到相同 root，但某 interpretation 只属于一个 world，则不存在 root-only verifier 同时对所有
   world sound 且 complete；
4. 该定理不把“只要 complete、允许任意超集”错误说成不可能。

## v0.4 必须增加

v0.4 应把下列对象变成根级、稳定 ID、closed schema 的有限 universe：

- fragment candidate 与 canonical split/support/width/primitive；
- intersection candidate；
- Atom instance candidate；
- endpoint attachment candidate；
- canonical port partition candidate；
- ownership option；
- lane chain candidate；
- Bond grouping candidate。

interpretation 只能引用这些根对象，独立 verifier 再用 exact-cover/CSP 枚举全部可行组合。任一阶段超过冻结容量
必须 `Reject(outside_capacity)`，不能 top-k 或静默剪枝。

## 当前权限

```text
structural_validator_frozen=true
completeness_verifier_frozen=false
proposer_public_execution_authorized=false
blind_execution_authorized=false
unique_graph_authorized=false
training_authorized=false
```
