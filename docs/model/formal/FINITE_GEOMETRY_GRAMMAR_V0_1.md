# 有限微场景几何 Grammar v0.1

状态：实现草稿，训练未授权。

这组规则把 `finite_micro_scene_v0_1` 的场景分成三个互斥层，并同时定义三个累计允许域。
所有判断只读取 Scene AST 中的节点坐标、节点可见角色和语义 Bond 端点；模型推理仍只读取图片。

## 互斥层

按以下优先级只分配一个 `grammar_layer`：

1. `adversarial_invalid`
   - 任一语义 Bond 的开区间穿过 active、nonincident Atom；
   - 两条不同语义 Bond 的中心线共线且有正长度重叠；
   - 一个没有文字标记的 degree-2 Atom，其两条 incident Bond 严格反向共线。
   - 同一点有三根以上 Bond 做 proper crossing，超过当前每像素两个 Bond owner 的容量；
   - 未参与任何 Bond 的隐式节点、重复 Node 坐标、零长度或越界 physical member；
   - 任意 `TEXT:*` 节点；v0.1 renderer 尚未绘制 glyph，不能把 GT 文字角色当侧信道；
   - physical member 与 semantic centerline/terminal envelope 不一致；
   - `order/display/member-count` 不满足 `1/SINGLE/1`、`2/DOUBLE/2`、`3/TRIPLE/3`。
2. `crossing_allowed`
   - 没有上述退化；
   - 至少两条不共享端点的语义 Bond 在双方开区间 proper crossing。
3. `simple_line_angle_valid`
   - 没有退化，也没有 nonincident proper crossing。

正长度重叠包括 shared-endpoint 同射线的嵌套线段。只在一个端点接触不算重叠。共线分段
`A-B-C` 在 B 为无标记 degree-2 Atom 时由第三条规则拒绝。若 crossing 恰好落在一个未参与
当前 Scene 的 grid 位置，该位置不是 active Atom，因此仍属于 `crossing_allowed`。

双键、三键的多条 physical lane 属于同一个语义 Bond，不彼此触发重叠规则。每条 lane 必须
与 semantic centerline 平行，并在轴向投影上覆盖同一 terminal envelope；垂直于轴的 lane
offset 合法。化学 graph signature 只包含 `order`，不包含任意 renderer 字符串；
`display_type` 只用于检查冻结映射，避免人为构造 `FOO` 造成假 collision。

## 累计允许域

互斥层统计回答“错误类型在哪里”，累计域统计回答 Lean 的 `IdentifiableOn`：

```text
simple_domain      = {simple_line_angle_valid}
crossing_domain    = {simple_line_angle_valid, crossing_allowed}
all_domain         = 全部三层
```

crossing/junction 两个 Scene 分属不同互斥层；只有在累计 `crossing_domain` 中按 observation
重新分组，才会看到它们构成同输入异图 collision。

## Manifest

每个 Scene 固定输出一条 canonical JSONL：

```text
scene_id
source
raw_raster_sha256
production_observation_sha256
canonical_graph_signature_full
canonical_graph_signature_structure
exclusive_grammar_layer
grammar_reasons
collision_origin
global_collision_class
simple_domain_collision_class
crossing_domain_collision_class
all_domain_collision_class
active_prior_id
single_answer_candidate
```

域外 Scene 写 `excluded`。其余取 `unique_graph`、`same_layer_graph_collision`、
`cross_layer_graph_collision` 或 `mixed_layer_graph_collision`。summary 分别报告互斥层和累计域
的 Scene 数、observation 数、异图 collision class 数，并报告从全域缩到 crossing/simple 后
变为唯一的 `prior_resolved` observation 数。

所有 layer、reason 和 collision class 都由 GT Scene 及替代图集合导出，只允许离线审计和过滤，
严禁作为 model forward 输入、teacher target 或逐样本动态 prior。发布域 prior 必须是全局配置。

`active_prior_id` 在本轮固定为 `null`：本轮只审计未施加 crossing prior 的累计域，不能把
`crossing_allowed` 场景偷偷当作已经消歧。`single_answer_candidate` 只对 simple domain 中
observation class 唯一的场景为真；它是审计输出，不是模型输入。

## 边界

以上规则在查看枚举计数前预注册，后续不得按层数量或准确率调整。该 grammar 是显式绘图先验，
不是从图片学出的事实。它只能证明有限 3x3、固定黑色 3px 样式
上的 `IdentifiableOn` 统计；不能证明真实化学图片可唯一恢复，也不授权训练。

Lean 当前证明的是精确整数几何谓词、Bool signal dispatcher 的优先级、互斥/穷尽和累计域
包含关系。Python 从 Scene AST 提取 signal 的正确性以及 66,716 个有限场景计数由可执行审计
与独立 verifier 验证，尚未被 Lean 内核证明。
