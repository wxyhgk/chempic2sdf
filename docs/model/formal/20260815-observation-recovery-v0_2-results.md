# Observation Recovery v0.2 正式结果

日期：2026-08-15

冻结实现提交：`df6f83fbfbb6aabf56df30d8cb32b0c27fbf043b`

总判定：

```text
Strategy A finite specification: PASS
Strategy B global prior:          NO-GO
Observable raster extractor:      NO-GO / not implemented
Semantic field serialization:     PASS on field-strict only
Training authorized:              false
```

## Lean 结果

`ObservationRecovery.lean` SHA256：

```text
db27a74a6c45a401e40ca07df38ac28782daf8efdb70c4c9577ba42689343b6e
```

Lean 4 build 为 13/13，无 `sorry`、`admit` 或新增 `axiom`。本轮证明：

- sound abstaining decoder 在同 observation 异图时不能输出 Graph；
- conservative decoder 不能靠全 Reject 真空通过，必须完整输出 strict Graph 和 honest ambiguity；
- render-invariant `Scene -> Prop` prior 加 coverage 不可能消除一个已有碰撞类；
- production prior 必须建模为 `extract -> exact candidate set -> choose`；
- 同 observation 的确定性 chooser 至多匹配碰撞真值中的一个；
- Oracle evidence 若不能写成 `extract(render(scene))`，就不是可观察字段；
- observable field 加 `FieldStrict` 才能推出 observation strict，左逆只在该条件域成立。

Lean 证明的是接口命题，不会阻止 `extract` 的 Python 实现内部按 hash 查表；因此仍需要隔离执行、
未知输入和独立代码审计。

## A/B Commitment

两次 full 均从冻结提交运行，覆盖：

```text
simple domain observations   19,823
crossing domain observations 36,649
manifest rows                56,472
```

A/B 四个正式产物逐字节一致：

```text
summary.json
0ed7f0367630a3a7b7dbde046846133c070a58c111481dd04da2c2d96fb562cf

validation.json
80d3023d4669143bdc2b43b18f7bc5db71a6fd16fa0d5ff2d62d0c37f58070e2

observation_recovery_manifest.jsonl
ae60a39da34751ac59a64ce71501fa9803dcc7950fe52eeed7c282dcb6ddf32d

recovery_certificates.json
6f1f360404efcbdac2726d3f63f6fb1b73fb2d2ad686af7ee853954e284e39a7
```

stdlib-only 独立 verifier 的 A/B 输出也逐字节一致：

```text
independent_review.json
f109494085130b5de404b31324cb37d8db2be5f8e4fab632bb6f65488a2d928f

reviewer source
fc6a20427ce3ddc60e84e09351f9b15954be1cc961dbca5a60a36cc92b94aeb2
```

verifier 不导入 recovery、runner、Scene generator 或 graph canonicalizer。它独立重组 56,472
个 domain-observation，得到 ambiguity taxonomy：crossing/junction 7，parallel 2（同一个
parallel observation 分别出现在 simple/crossing 累计域）。Strategy B 保留 44,281 个真值
Scene，排除 12,200 个；其中 12,191 个 observation 为零 survivor。

## Strategy A

保守策略精确复现预注册：

| 域 | Graph | Ambiguous | 类型 |
|---|---:|---:|---|
| simple | 19,822 | 1 | parallel multiplicity 1 |
| crossing | 36,641 | 8 | crossing/junction 7；parallel 1 |

错误 Graph 输出为 0，歧义类 Graph 输出为 0。它只证明冻结有限 observation table 上的规格函数，
不是 PNG 推理器；ambiguity kind 仍由完整有限 candidate class 审计得到。

## Strategy B

Evidence-only chooser 不读取 observation hash、Scene ID、GT graph signature 或 RDKit 结果。固定规则
把相同纵向投影的 2/3 条平行 lane 解释为 multiple Bond，并把无标记内部 crossing 解释为
disconnected。

正式结果：

| 域 | Graph | Reject / zero survivor | coverage |
|---|---:|---:|---:|
| simple | 14,650 | 5,173 | 73.90% |
| crossing | 29,631 | 7,018 | 80.85% |

两个域合计 12,191 个 observation 的 protocol choice 不属于该 observation 的合法图候选集合。
survivor graph collision 虽为 0，但这是“只选一个解释”的自然结果，不能抵消 coverage 失败。

这证明两个 prior 不能作为全局化学绘图规则。若只在 8 个已知 collision hash 上启用，则会变成
GT/hash 查表，同样禁止。

## Crossing Prior 外部风险

对 `Data/MR-TADF_SDF` 的 1,120 个 V2000 SDF 使用 `Fraction` 精确读取坐标。对每个 degree-4
Atom 枚举三个 perfect matching，严格对向条件为：

```text
cross(vi, vj) = 0 and dot(vi, vj) < 0
```

结果：

- 16 个严格十字 degree-4 Atom，分布于 13 个分子；
- 16 个全部是 C，且四条 Bond 全为 single；
- 1° 桶为 145 个 Atom / 94 个分子；
- 5° 桶为 216 个 Atom / 138 个分子。

代表反例 `GDG1136.sdf` 的 0-based Atom 83 是合法隐式 C 十字 junction。因此
`unmarked_transversal_crossing_is_disconnected_v1` 会确定性误杀真实合法结构，只能绑定受限
renderer profile，不能作通用默认。

风险审计 SHA256：

```text
ecbf28654ba600f7fb739e7f7fb58ab95eb2d05320222aeb0f49e1a773dba7ca
```

## 当前表示边界

本轮 lane evidence 来自有限 Scene `PhysicalMember` 的 canonical merge。它验证了同一冻结
observation 的 witness 得到相同 lane evidence，但没有实现：

```text
PNG/tensor bytes -> neutral lane/endpoint/intersection/text-mask Evidence
```

Strategy A Semantic Oracle Field 的 56,463 个 field-strict observation 左逆全部通过，只说明
`semantic field -> graph` 序列化未丢图；不能写成 image extractor GO。

## 下一门

下一周期只做隔离的 raster-neutral evidence extractor：

1. 子进程只允许读取完整 PNG/tensor bytes，禁止导入 Scene、manifest、SDF、digest 表和 graph
   canonicalizer；
2. 输出未解释 lane、terminal event、intersection 和 text mask，不输出 Bundle/Junction GT；
3. exact collision pair 必须输出相同 Evidence 和完整的两个 candidate Graph；
4. strict motif 必须生成完整唯一 candidate；
5. 独立 agent 在实现冻结后生成未知坐标、旋转、反射、缩放、线宽和文字干扰；
6. 独立枚举器验证 candidate soundness/completeness；
7. A/B commitment 一致，并验证稳定性条件 `2 epsilon < Delta`。

在 raster factorization、candidate completeness 和 conservative classification 三项全部达到
100% 前，保持 `training_authorized=false`。
