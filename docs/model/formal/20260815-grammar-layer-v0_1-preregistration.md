# Grammar Layer v0.1 预注册协议

日期：2026-08-15

状态：`PREREGISTERED`。本文件先于正式 classifier 全量结果提交；结果不允许反向修改本版规则。

## 目标

上一周期完整枚举任意 0–4 边，证明宽松场景集合中存在大量同 raster 异图。这个周期不训练
模型，而是先回答：哪些场景属于当前化学骨架绘图约定，哪些场景只有在显式 crossing 先验下
可接受，哪些场景不得进入单答案任务。

## 精确几何

v0.1 使用整数坐标，它是有理坐标的有限子集。所有判定使用整数叉积、点积和区间比较，不使用
浮点 tolerance、像素 NMS 半径或角度近似。

对线段 `s=[a,b]`、点 `p` 定义：

```text
strictInterior(p,s)
  := cross(b-a, p-a) = 0
     and dot(p-a, p-b) < 0

positiveOverlap(s,t)
  := s,t 共线
     and 在主轴投影上的开区间交集长度 > 0

properCross(s,t)
  := s,t 不共线
     and 唯一交点同时位于两条线段的相对内部

implicitStraightDegree2(v)
  := v 为 IMPLICIT、语义 degree=2
     and 两个邻接方向 cross=0、dot<0
```

`179°` 不等于 `180°`。近似共线属于后续稳定性 shell，不得加入本版符号规则。

## 固定失败信号

按以下字段输出，命中任一即进入 `adversarial_invalid`：

- `zero_length_member`：任一 physical segment 长度为 0。
- `duplicate_atom_coordinate`：不同 Atom 占据同一位置。
- `passes_nonincident_atom`：Bond 的相对内部穿过非语义端点 Atom。
- `positive_collinear_overlap`：两条不同 Bond 的中心线存在正长度共线重叠。
- `implicit_straight_degree2`：无可见 marker 的隐式二价 Atom 严格 180°。
- `nonsemantic_t_touch`：一条线端点落在另一条非 incident 线的相对内部。
- `multiway_internal_crossing`：三条或更多非 incident segment 在同一内部点相交。
- `crossing_mixed_with_atom_or_attachment`：crossing 位置同时是 Atom、visible port、文字 attachment 或 marker。
- `ambiguous_multiplicity`：physical lane 无法唯一归入语义 Bond multiplicity。
- `no_carrier_after_ambiguity_mask`：去除歧义区域后 Bond 没有可监督 carrier。

当前 3×3 core 没有 Text、marker、attachment、dash 和 carrier mask；相应字段必须显式记录为
`not_in_current_grammar`，不能伪造通过。结构性 self-loop、未知 Atom 和重复语义 endpoint pair
继续由 Scene AST validator 拒绝，不进入几何层统计。

## 互斥场景层

分类优先级冻结为：

1. 命中任一固定失败信号：`adversarial_invalid`。
2. 未命中失败信号，且存在合规 `properCross`：`crossing_allowed`。
3. 其余：`simple_line_angle_valid`。

合规 crossing 必须满足：两条 Bond 无共同语义端点，交点不是 Atom/port/text/marker，没有第三条
segment 同点穿过，也不伴随 overlap 或 T-touch。

每个 Scene 必须恰好属于一个互斥层。`adversarial_invalid` 的含义是“当前版本不得进入单答案
grammar”，不是化学上永远禁止；未来支持需新建 grammar 版本。

## 累积允许域

互斥层标签不等于可辨识域。必须同时计算：

```text
simple_domain   = {simple_line_angle_valid}
crossing_domain = {simple_line_angle_valid, crossing_allowed}
all_domain      = 全部三层
```

`IdentifiableOnUpTo`、collision class 和 strict 都在上述域内分别计算。尤其 crossing scene 与
simple junction scene 可以同像；只在 exclusive crossing 层内部统计会漏报，禁止这样做。

`strict` 一律写成 `strict_within_<domain>_v0_1`，不得声称全局 strict。

## Crossing 先验

预注册先验：

```text
prior_id = unmarked_transversal_crossing_is_disconnected_v1
```

这个先验是绘图约定，不是视觉事实。未激活先验时，crossing/junction 同 observation 异图必须
标记 ambiguity。激活先验时必须对 observation 等价类整体选择 crossing 解释，不能同时保留
同像 junction 单标签；反向 junction 先验必须使用另一个 `prior_id`。同一数据发布只能激活一套
互斥先验。

## 完整 Manifest

每个 Scene 必须输出一条确定性记录：

```text
scene_id
source
raw_raster_sha256
production_observation_sha256
canonical_graph_signature_full
canonical_graph_signature_structure
exclusive_grammar_layer
grammar_reasons
global_collision_class
simple_domain_class
crossing_domain_class
active_prior_id | null
```

manifest 必须覆盖 66,712 个 core 和 4 个 fixture，scene_id 唯一。碰撞证书只保留代表，不能替代
完整 manifest。

## 冻结门

- 66,716 个 Scene 恰好分类一次，未分类为 0，层间重叠为 0。
- Atom 重命名、Bond 重排、端点交换后层标签 100% 不变。
- 平移、90° 旋转、反射和正比例整数缩放后层标签 100% 不变。
- 长边穿点、共线分段、overlap、180° degree-2、T-touch、crossing/junction、double/parallel
  固定反例分类 100% 正确。
- `all_domain` 必须复现上一周期 3,518 个 collision class、17,028 个场景和最大 9 个图签名。
- 未应用 prior 的跨层同像异图全部进入 ambiguity。
- `adversarial_invalid` 进入单答案候选数为 0。
- 同提交 A/B 的 manifest、summary、validation、certificate SHA 全部一致。
- `training_authorized=false` 固定不变。

红队在预实现诊断中曾得到“排除穿点/重叠/180° 后 crossing_domain 约 8 个碰撞类，进一步禁止
crossing 后 simple_domain 约 1 个”。这些是已披露的预期检查值，不是本版规则的调参目标；正式
结果不一致时必须报告原因，禁止改规则追数。

## 下一门

本门通过后才建立 canonical Oracle field，并验证：

```text
decode(oracleField(scene)) ~= canonicalGraph(scene)
```

其中 `~=` 是忽略 Atom ID、Bond 顺序和无向端点顺序的图等价。只有全部 `field_strict` 场景
达到 100% 左逆，才进入样式距离与拒识半径门；仍不授权 Capacity64 或大规模训练。
