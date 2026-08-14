# Observation Recovery v0.2 可执行审计

日期：2026-08-15

状态：实现与 smoke 已完成；尚未形成 A/B 全量 commitment。训练未授权。

## 目的

本审计回答两个不同问题，禁止把它们混在一起：

1. 在冻结有限域中，保守拒识和两个全局绘图 prior 是否满足 observation-class 正确性；
2. 是否已经有一个从任意 opaque raster/tensor 提取 `Lane -> BondBundle -> Port ->
   Junction/Crossing` 的可部署算法。

第一个是有限关系审计，第二个才是视觉恢复算法。目前只实现了第一个问题的一部分，第二个问题
仍是 `NO-GO`。

## 冻结输入

runner 只接受 Grammar v0.1 `final_a` 的两个承诺输入，并在解析前验证 SHA256：

```text
scene_manifest.jsonl
f7519924b13b685f235a90dbd837772ced626cdffc9bbbacd4ff423d22c406f7

collision_certificates.json
f97262008a50edf26b263745b9058533820be8bb72fdffcc3f9c38e6f9839485
```

硬门还要求：

- manifest 恰好 66,716 行；
- Scene ID 全部存在且唯一；
- 3,518 份 collision certificate 与 manifest 的全域冲突 observation 完全一致；
- 每个 certificate witness 的 observation 和 graph signature 与 manifest 一致；
- 冻结生成器重建的 Scene graph signature 与 manifest 逐行一致。

证书只能用于审计，不能成为 forward chooser 的查找表。

## 三层对象

### 1. Opaque observation

冻结输入只给出 production tensor 的 SHA256。代码没有从这个 tensor 恢复 lane，更没有用 SHA256
查 GT 图。

### 2. Canonical lane evidence

审计从有限 Scene 的 `PhysicalMember` 构造 lane evidence：同一直线上相接或重叠的 member 合并为
最大 carrier。这样 crossing witness 的两根穿越线与 junction witness 的四个半线段得到相同
evidence。

这个构造读取的是 Scene physical member，不是 raster/tensor。对每个冻结 observation 检查
“所有 Scene witness 得到完全相同 evidence”，只能证明在该有限表上存在外延因子化：

```text
same opaque observation -> same canonical lane evidence
```

它不提供可计算的 `opaque tensor -> lane evidence` extractor。因此所有产物固定写入：

```json
{
  "audit_oracle_only": true,
  "deployable_from_opaque_observation": false,
  "observation_extractor_implemented": false
}
```

### 3. Semantic Oracle Field

Oracle Field 包含：

- `visible_observation_primitives.lanes`：有限 Scene member 归一化后的 lane evidence；
- `protocol_interpretation.bond_bundles`：lane owner、共享 terminal envelope 和 order candidate；
- `protocol_interpretation.terminal_ports`；
- `protocol_interpretation.junctions`；
- `protocol_interpretation.crossings`：owner bundle、精确有理数交点和 marker state；
- `ambiguity`。

Strategy A 的 strict field 中，bundle ownership 来自唯一有限 Scene witness；Strategy B 的 field
来自固定 prior chooser。两者都不是当前图像模型可直接监督的“可见真值”。字段中不包含 Scene
ID、grammar layer、collision class 或 GT graph signature。

## Strategy A：保守拒识

对域 `D` 中同一 observation 的全部 Scene，先取 graph signature 集合：

```text
|GraphSet_D(o)| = 1 -> Graph(unique graph)
|GraphSet_D(o)| > 1 -> Ambiguous(kind)
```

Ambiguous 输出不含 bundle/port owner，只保留 lane evidence 和 observation-class 歧义类型。整个
class 得到同一个非 Graph 结果，不能只拒绝其中一个 GT。

全量回归目标仍是：

| 域 | Graph | Ambiguous | 类型 |
|---|---:|---:|---|
| simple | 19,822 | 1 | `parallel_multiplicity=1` |
| crossing | 36,641 | 8 | `crossing_junction=7`，`parallel_multiplicity=1` |

Strategy A 的 candidate set 只相对冻结有限域完整，不代表现实世界所有合法解释都已枚举。

## Strategy B：Evidence-first 全局 chooser

早期实现曾在同一 observation 的 semantic Scene candidates 中比较 bundle/crossing score，再保留
“更符合 prior”的 Scene。该方法会读取待证明的 semantic ownership，不能作为前向算法，已经
废弃。

当前 forward 路径固定为：

```text
canonical lane evidence
    -> 按平行方向和完全相同的纵向投影区间分组
    -> 2/3 条一组解释为 order=2/3 BondBundle
    -> 其他 lane 解释为 order=1 BondBundle
    -> 无标记内部 transversal crossing 一律不建 Junction
    -> Graph
```

chooser 不读取 observation hash、Scene ID、grammar layer、GT graph signature、RDKit sanitize 或
验证集答案。前向 Graph 生成后，审计才比较：

```text
survivors(o) = {s in SceneClass_D(o) | meaning(s) = chosenGraph(evidence(o))}
```

这里的 graph signature 只用于统计 consistency，不能影响 chooser。

激活 prior 固定为：

```text
terminal_aligned_parallel_strokes_are_one_multiple_bond_v1
unmarked_transversal_crossing_is_disconnected_v1
```

“terminal aligned”在本版不使用线宽、像素距离或角度容差，只表示平行 lane 具有完全相同的
有理数纵向投影区间。横向距离不参与判断。这是对预注册全局规则最直接、无调参的执行，也会
暴露该规则对合法平行独立单键的风险。

## 当前发现的反例

在代码完成后的非 commitment 全量诊断中：

| 域 | observation | Strategy B Graph | 零 survivor |
|---|---:|---:|---:|
| simple | 19,823 | 14,650 | 5,173 |
| crossing | 36,649 | 29,631 | 7,018 |

原因不是模型训练不足。冻结域含大量两条或三条端点投影对齐、但化学上彼此独立的单键；全局
parallel prior 把它们重解释为双键/三键，chosen graph 不属于原 observation class。crossing
prior 也会在较窄域中排除合法 junction 解释。

因此，两个 prior 能解开 8 个已知碰撞类，并不意味着它们能作为全局绘图协议保持 coverage。
按预注册门，`strategy_b_coverage_is_100_percent=false` 必须得到 `finite_policy_consistency=NO-GO`。
不能把规则改成“仅对已知 collision hash 生效”，也不能查看 GT 后逐图启用 prior。

## Left-inverse 的严格边界

`decode_oracle_field(field) = canonical graph` 只在 `field_strict=true` 时检查。通过意味着
bundle/port/junction 的**语义序列化**没有丢图结构：

```text
semantic witness/prior field -> graph
```

它不证明：

```text
opaque image -> Oracle Field
```

validation 因此把门拆开：

- `semantic_serialization_left_inverse`：只对 field-strict denominator 报 numerator；
- `finite_policy_consistency`：Strategy A 正确性、Strategy B nonempty/unique survivor；
- `observable_extractor_factorization`：当前固定为 `no_go_not_implemented`；
- `overall_gate_status=no_go`；
- `training_authorized=false`。

即使 semantic serialization 为 100%，也禁止写成“Oracle extractor 已通过”。

## 产物

runner 输出：

```text
observation_recovery_manifest.jsonl
summary.json
validation.json
recovery_certificates.json
```

manifest 每行对应 `(domain, observation)`，同时记录 A/B 结果、coverage、survivor、field-strict、
left-inverse 和完整边界字段。certificate 记录 Strategy A ambiguity 和 Strategy B 非 Graph 类，
其中 Scene ID/graph signature 只用于审计反例，不属于 Oracle Field。

每个 decision 另有 `protocol_choice_graph_signature`。Strategy B 无论最后是 Graph、Ambiguous 还是
Reject，都记录 evidence-only chooser 产生的图；Reject 的正式 `graph_signature` 仍为 `null`。
独立 verifier 因而可以检查“chosen graph 不在候选意义集合”是否为真正拒识原因，而不必导入
chooser。Strategy A Graph 的 protocol choice 等于唯一图，Strategy A Ambiguous 为 `null`。

## 执行

smoke：

```bash
python -m formal.chemgraph.audit.run_observation_recovery_v0_2 \
  --output /tmp/observation_recovery_v0_2_smoke \
  --smoke-limit 8
```

测试：

```bash
python -m unittest formal.chemgraph.audit.test_observation_recovery_v0_2 -v
```

最终 commitment 必须从同一提交运行两次 full：

```bash
python -m formal.chemgraph.audit.run_observation_recovery_v0_2 \
  --output formal/chemgraph/audit/results/observation_recovery_v0_2/final_a

python -m formal.chemgraph.audit.run_observation_recovery_v0_2 \
  --output formal/chemgraph/audit/results/observation_recovery_v0_2/final_b
```

四个产物必须逐字节一致，随后再由不导入本 generator/chooser/graph canonicalizer 的独立 verifier
复核。当前尚未执行这两步，不能把本文件当最终结果报告。
