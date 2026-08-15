# Carrier Candidate Set v0.3 结构验证结果

日期：2026-08-15

## 结论

v0.3 已完成并冻结 closed schema、数据无关结构 Validator、确定性 fixture、mutation tests 和 Lean 条件定理。
当前成功状态只能是：

```text
STRUCTURALLY_VALID_NOT_COMPLETE
```

它表示一份候选文档内部的 raster、Boundary、fragment、ownership、Atom、port、lane、Bond、
AtomPair provenance、graph class 和 stage trace 关系能够被独立重算；不表示 proposer 已经从 PNG 找全候选，
也不表示某个 interpretation 是正确答案。

独立复核结论：红队 `GO`；Oracle 对 288 个 schema-valid 坏引用 mutation 为 0 traceback，并同意只冻结
`STRUCTURALLY_VALID_NOT_COMPLETE`。该 GO 不传递给 completeness、真实 proposer、blind、唯一图或训练。

权限保持：

```text
blind_sample_selection_authorized=false
unique_graph_authorized=false
proposer_public_execution_authorized=false
training_authorized=false
```

## 本轮实际实现

- 根级完整 `boundary_universe`、`compatible_boundary_pairs`、`atom_proposal_universe` 和
  `direct_pair_universe`；
- final foreground、bond mask、text mask 三份完整 RLE、独立 hash 和 PNG 字节绑定；
- Boundary 到 fragment、fragment 到 owner、fragment 到 lane、lane 到 Bond 的关系门；
- Bond 的无序 AtomPair 到根级 Direct pair 的 provenance 绑定；
- endpoint 到 Atom 的平方距离门、Q8 有向 dot/cross 证据和最多 8 个候选限制；
- 1/2/3 lane 与 Bond order 的精确关系；
- white-gap 和 lane-separation 的 final-raster、span、距离与 support 邻域绑定；
- 固定 12 阶段顺序、组件 hash、输入输出链、根级产物 hash 和 record count；
- 多图类 ambiguity taxonomy 的 code-kind、witness 和 evidence 引用；
- 正常序列化只有 `candidate_set` 或 closed `reject`，没有 graph/winner/accept。

## 可执行 fixture

fixture 由 `build_fixtures.py` 确定性生成：

- `valid_single_bond.json`：两 Atom、一 lane、一个单键；
- `valid_glyph_only.json`：笔画归 text glyph，空间化学图为空；
- `valid_glyph_bond_collision.json`：相同像素保留 glyph 与 Bond 两个非同构图类；
- `valid_crossing_junction_collision.json`：相同十字像素保留 through crossing 与中心 junction 两个非同构图类；
- `valid_reject.json`：`unsupported_primitive` closed reject，不携带异常文本或图结果。

这些 fixture 证明协议能表达歧义，不证明真实图片上的 proposer 能发现这些解释。

## Mutation 覆盖

当前 v0.3 专项测试为 45 项，覆盖：

- duplicate key、非规范 JSON、非规范 PNG、额外 reject 字段；
- RLE 重叠、计数/hash 伪造、错误 PNG 绑定；
- Boundary 消失、fragment 改线、ownership 重复或错误 owner；
- 错误 AtomPair、错误 Direct pair ID、缺失 Direct universe 成员；
- inactive Atom proposal、重复空间 anchor、port/relation 不双射；
- `isqrt` 下取整越界反例和端点内向 attachment 反例；
- Bond order 与 lane 数不一致；
- 阈值看似通过但与 Boundary 几何不符的 compatible pair；
- 远离 span 的 white-gap support 和伪 lane-separation；
- stage 顺序、组件、输入输出、计数与 hash 伪造；
- glyph/Bond 与 crossing/junction 被错误压成 false singleton；junction region 的 kind 与中心 port 必须由实际 endpoint 重算；
- taxonomy code-kind、witness、graph class hash 伪造；
- schema-valid 坏引用（包括未知 white-gap ID）必须返回 `INVALID`，不能抛未捕获异常。

## 验证结果

```text
v0.3 Validator tests: 45 / 45
formal audit tests:    157 / 157
Lean build jobs:        26 / 26
Lean version:           v4.33.0
```

当前冻结候选字节 hash：

```text
protocol_v0_3.json:
f4aeb651e531087eea18bc3b96eebd38bdc2956f5dd635bd0baa6b677824e37c

input.schema.json:
d2552f04fe91438cb0d2ff6d88d88ed9fa7f59d5752d2766e9f901f1d0277bf7

candidate_set.schema.json:
bb69ee2b054ecc93e1c563dfb6638a533f4559f787af96645d494d615da7752b

CarrierCandidateSetV03.lean:
46f827184073c7e7b4d791c334eaa1c391964bdfb5d74d90568c827ab5d6f471

validate_candidate_set.py:
c508f32bfd85d836cbb939d31ad0c4cf3feee9e013c11edb6aeb0ba2606dda43
```

hash 只绑定本次被审计字节；任何文件修改后必须重新计算、重跑测试和重新红队，不能沿用本节结果。

## Lean 证明边界

Lean 已证明有限域内的 pair/candidate 枚举 sound/complete、所有权互斥、lane/Bond 精确 fiber、空图解释不遗漏、
12-stage trace 结构和序列化永不直接输出 graph。

这些 complete 定理仍以 `RootedInterpretationUniverse.complete` 为前提。Lean 尚未从真实 PNG 和四个根级
universe 独立构造全部 interpretation，也没有证明神经网络 Boundary/Atom proposal 的召回率。因此这里的
数学结果是“若有限根域和 predicate 完整，则序列化不会漏掉域内可行项”，不是“真实图像已被完整理解”。

## 尚未完成

1. 独立 completeness verifier：不能信 proposer 自报 `complete=true` 和 universe hash；
2. 冻结组件 manifest：多数 preprocess/enumerator/algorithm SHA 还没有绑定未来真实实现；
3. compatible pair 的 raster darkness/occupancy 仍需从 grayscale 和 mask 独立重算；
4. wedge、aromatic circle、curve 等当前只能 closed reject，仍需公开反例验证不能伪装为 straight；
5. 公开合成图和已暴露图片上的 truth candidate coverage、false singleton、overflow 和 exception 统计；
6. 在上述门通过前，不实现 winner，不读取 sealed/blind，不训练。

## 下一步

下一阶段只实现独立 completeness verifier 和公开 fixture coverage runner。它必须从冻结根域和 predicate
重新枚举，而不是读取 proposer 的 `complete` 布尔值作为证据。通过后才能评审一个不训练的公开 proposer；
训练仍是更后面的独立决策。
