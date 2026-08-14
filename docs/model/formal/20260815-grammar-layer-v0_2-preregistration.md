# Grammar Layer v0.2 预注册协议

日期：2026-08-15

状态：`PREREGISTERED`，训练未授权。

输入基线：Grammar v0.1 最终 manifest：

```text
f7519924b13b685f235a90dbd837772ced626cdffc9bbbacd4ff423d22c406f7
```

v0.1 结果永久保留，不允许覆盖。v0.2 不通过调线宽、角度或距离阈值消除反例，而是把
“物理 stroke 如何组成语义 Bond”和“内部 crossing 是否为 Junction”提升为显式协议。

## 数学对象

对允许域 `D` 和 production observation `o`：

```text
SceneClass_D(o) = { s in D | Render(s) = o }
GraphSet_D(o)    = { canonicalGraph(s) | s in SceneClass_D(o) }
Ambiguous_D(o)   := |GraphSet_D(o)| > 1
Strict_D(o)      := |GraphSet_D(o)| = 1
```

总解码器的输出类型不再强制为 Graph，而是：

```text
DecodeResult = Graph canonicalGraph | Ambiguous ambiguityKind | Reject reason
```

基础正确性命题：

```text
Strict_D(o) -> decode(o) = Graph(g) -> g 是 SceneClass_D(o) 的唯一图
Ambiguous_D(o) -> decode(o) 不得无 prior 输出 Graph
```

允许拒识不是降低标准。它把逻辑上不可观测的输入从“被迫猜一个图”改成显式不确定性；若业务
必须单答案，只能激活公开、全局一致的绘图先验。

## Oracle Field v0.2

先验证表示，不训练预测器。每个 observation 的 canonical Oracle field 包含：

```text
Lane        = 两个可见 terminal port + carrier pixel/segment
BondBundle  = 1/2/3 条 lane + 共享 terminal envelope + order candidate
Junction    = 聚合后的 terminal port 集合
Crossing    = 两个 carrier 的内部相交点 + owner IDs + marker state
Ambiguity   = parallel_multiplicity | crossing_junction | none
```

解码顺序冻结为：

1. physical lane；
2. lane bundle / Bond primitive；
3. terminal port；
4. Junction 或 Crossing；
5. Atom/Bond graph；
6. RDKit/SDF 编译。

不能先生成 Atom 顶点再靠全点对补边；也不能把 double/triple 的每条 lane 端点当独立 Atom。

Oracle field 必须只描述 observation 中可见或协议声明的信息。GT Atom ID、SDF Atom 顺序、
exclusive grammar layer 和 collision class 禁止进入模型输入字段。

## 两种冻结策略

### 策略 A：保守拒识

`prior_id=null`。若 `Ambiguous_D(o)`，输出确定的 `Ambiguous(kind)`，整个 observation class
一起拒识，禁止只删除其中一个 GT。

有限 v0.1 域的复现目标：

- simple domain：19,822 个 strict observation，1 个 `parallel_multiplicity` ambiguity。
- crossing domain：36,641 个 strict observation，7 个 `crossing_junction` ambiguity，加 1 个
  `parallel_multiplicity` ambiguity。

这些是对冻结 manifest 的回归值，不是新规则的调参目标。

### 策略 B：绘图约定消歧

只有显式激活以下全局 prior 才允许单答案：

```text
terminal_aligned_parallel_strokes_are_one_multiple_bond_v1
unmarked_transversal_crossing_is_disconnected_v1
```

第一个 prior 对同 terminal envelope 的 2/3 条平行 lane 只保留一根 order=2/3 Bond 解释；
第二个 prior 对无 marker 的内部 transversal crossing 只保留 disconnected crossing 解释。

两者都是绘图约定，不是从 opaque raster 可证明的视觉事实。发布时必须记录完整 prior set；
切换 prior 必须产生不同协议版本。不能按单张图片、验证集正确答案或 RDKit sanitize 结果动态选择。

策略 B 只做数学可行性验证，不因有限域达到 0 collision 就自动成为默认生产策略。它还必须在
公开绘图风格上统计被排除的合法 degree-4 Junction 和近邻平行独立单键风险。

## Lean 门

必须在 Lean 中新增并证明：

1. `DecodeResult` 的 abstaining decoder 在 strict/ambiguous 域上的 soundness。
2. 同 observation 异图时，无 prior 的 decoder 不得同时正确。
3. `ResolvedByPrior` 在每个支持 observation 上非空且图签名唯一。
4. prior 是 observation-class 级函数，不依赖 Scene ID 或 GT graph。
5. Oracle field 左逆只能陈述为：

```text
fieldStrict(s) -> decodeField(oracleField(s)) ~= canonicalGraph(s)
```

## 可执行门

- v0.1 manifest SHA 必须匹配，输入行 66,716，scene ID 唯一。
- 策略 A 对所有 ambiguity class 输出同一个非 Graph 结果；错误 Graph 输出 0。
- 策略 B 在声明域内 coverage 为 100%，每 observation 保留至少一个 Scene。
- 策略 B 在声明域内不同 graph signature collision 为 0。
- Oracle field 对全部 field-strict Scene 的左逆为 100%。
- Atom relabel、Bond reorder、endpoint swap、平移、90° 旋转、反射和整数缩放不改变结果。
- A/B 同提交重放的 manifest、summary、validation、certificate SHA 全部一致。
- 独立 verifier 不导入 generator 的 prior/规范签名实现。
- 任一项失败：`training_authorized=false`。

## 禁止捷径

- 不允许用 production observation hash 查 GT 图或 ambiguity 标签作为真实推理算法。
- 不允许把 grammar layer、prior survivor 或 collision class 当模型监督字段而不提供可见 Oracle
  primitive。
- 不允许用 RDKit sanitize 事后选择碰撞类中的正确 GT；sanitize 只能拒绝化学无效输出。
- 不允许因为 MR-TADF 或 USPTO benchmark 分数上升而修改本版符号规则。
- 不允许在 Oracle field 左逆通过前训练 Capacity64、扩大数据或恢复主模型训练。
