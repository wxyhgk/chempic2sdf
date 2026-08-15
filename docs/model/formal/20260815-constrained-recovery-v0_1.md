# 受约束恢复 v0.1：训练前数学方案与有限验证

日期：2026-08-15

状态：`FINITE_SPECIFICATION_PASSED`
训练授权：`false`

## 1. 结论

下一版不应继续把任务表述为“找出所有点，再判断所有点对是否有键”，也不应直接生成一段 SVG 后相信它
天然具有化学语义。更合适的数学对象是一个有限的场景解释集合：每个解释同时给出可见图元的语义归属、
键 lane 分组、端点 incidence、文字区域、化学图和可重渲染参数。

视觉模型只提供候选图元和带误差界的局部分数。确定性解码器执行以下流程：

```text
最终图片 y
  -> 可观察图元 P 与分数区间
  -> 满足精确归属和结构约束的解释集合 H(y)
  -> 按图同构合并为候选图类 C(y)
  -> Graph | Ambiguous | Reject
```

本轮 Lean 定理和 12 个有限 fixture 验证了这个接口的核心性质，但没有证明真实图片上的图元提取覆盖率，
因此不授权训练。

## 2. 数学定义

一个解释 `h` 至少包含：

- 可见 primitive 到 `bond lane / glyph stroke / marker` 的归属；
- 一条语义 Bond 拥有的 1--3 条 physical lane；
- Bond 的两个语义端点、键级和显示类型；
- 文字节点及其连接 anchor；
- 化学图 `G(h)` 与渲染 `R(h)`。

冻结约束如下。

### 2.1 精确笔画归属

对每个最终可见 primitive `p`，必须恰好存在一个 owner：

```text
forall p, exists! owner, owns(owner, p)
```

因此同一笔画不能同时作为 `N/H` 的字形和化学键。这个约束能阻止重复使用证据，但不能凭空判断 owner；
证据不足时必须保留两种解释。

### 2.2 多键是一个对象

双键、三键分别是拥有 2、3 条 physical lane 的一个 Bond，而不是 4、6 个端点：

```text
BondBundle = (beginNode, endNode, order, lanes)
len(lanes) = order, 1 <= order <= 3
```

所有 lane 共享同一对语义端点。这样从表示层消除“每条平行线各自产生两个原子”的错误。

### 2.3 边必须有可见载体

每条输出边必须由一个候选 carrier 支撑：

```text
edge(a, b) -> supportedCarrier(a, b)
```

所以 `A-B-C` 不得补出 `A-C`，三臂 fan 的两个外围节点也不得直接相连。化学价态可以进一步删除候选，
但不能创造没有可见载体的边。

### 2.4 鲁棒唯一性

令干净分数为 `E(h)`，实际图像分数为 `E'(h)`，并已审计：

```text
|E'(h) - E(h)| <= epsilon
```

若真实解释与所有非同构可行解释满足：

```text
E(truth) + 2 * epsilon < E(alternative)
```

则实际分数下真实解释仍严格获胜。Lean 已证明这个充分条件；有限枚举同时验证边界是紧的：等于
`2 * epsilon` 时可以打平，不能输出唯一图。

完全同像的非同构解释具有相同纯视觉重建分数。此时不得用候选顺序、文件名、数据频率或 RDKit sanitize
冒充视觉证据，只能输出 `Ambiguous`，或者公开引入一个会改变任务定义的先验。

## 3. 文献依据

- [MolScribe](https://doi.org/10.1021/acs.jcim.2c01480) 直接预测原子、键和几何布局，并将符号化学约束
  用于后处理，说明图结构和确定性规则可以结合。
- [MolGrapher](https://doi.org/10.1109/ICCV51070.2023.01791) 使用 keypoint、supergraph 和 GNN；其候选边
  构造仍按估计键长半径和最多邻居数裁剪，说明候选覆盖必须单独审计。
- [LETR](https://doi.org/10.1109/CVPR46437.2021.00424) 用集合预测和匈牙利匹配直接检测完整线段，支持把
  Bond carrier 当作一等对象，而不是从局部边缘或拐点拼装。
- [HAWP](https://doi.org/10.1109/CVPR42600.2020.00286) 用 attraction field 联合线段与 junction，支持从
  稠密视觉证据提出结构对象，但不提供化学图唯一性的保证。
- [Relationformer](https://doi.org/10.1007/978-3-031-19836-6_24) 联合预测对象和关系，说明节点与边应共享
  上下文；其成对关系头仍不能代替候选覆盖和全局认证。
- [MarkushGrapher-2](https://arxiv.org/abs/2603.28550) 报告 PaddleOCR 会把化学键误作减号或等号；专用
  ChemicalOCR 使用约 23.5 万合成和 7 千真实 OCR 标注，支持文字/键归属应作为专门证据，而非通用 OCR
  教师的硬标签。
- [StarVector](https://doi.org/10.1109/CVPR52734.2025.01508) 说明图片到矢量程序生成可行，但 SVG 语法正确
  不等于化学语义唯一，因此这里只借鉴图元程序表示，不借用“单序列即唯一答案”的假设。
- [数据驱动逆问题综述](https://doi.org/10.1017/S0962492919000059) 强调逆问题中的可辨识性、稳定性和先验
  边界；本方案的候选类与 `2 epsilon` 间隔门正对应这两个条件。

## 4. 已完成验证

Lean 模块：`formal/chemgraph/ChemGraph/ConstrainedRecovery.lean`

已证明且不含 `sorry/admit/axiom`：

- 精确归属下 owner 唯一，同一 primitive 不得同时归属键和字形；
- 多条 lane 共享同一对 Bond 端点；
- 无 carrier 支撑的边不能进入 sound graph；
- 干净分数间隔严格大于 `2 epsilon` 时，扰动后赢家保持；
- 同渲染解释的纯视觉分数相同，不能形成严格视觉间隔；
- `A-B-C` 和三臂 fan 的假边反例。

Python 有限验证共 9 项，覆盖现有 12 个 EvidenceCore fixture：

```text
Graph      5
Ambiguous  5
Reject     2
```

重点结果：

- chain 无首尾 shortcut；
- fan 无外围节点互连；
- 双键、三键和带小扰动三键的 bundle 候选均为 2 节点、1 条语义边；
- crossing、double/triple-vs-parallel-singles、glyph-vs-bond 保持多候选；
- 严格分离的分数区间可认证唯一图，接触或重叠区间保持歧义；
- 对自然数分数的全部允许扰动枚举验证了 `2 epsilon` 边界。

机器结果：

`formal/chemgraph/audit/results/constrained_recovery_v0_1/validation.json`

## 5. 下一道训练门

训练前还需一次真实输入候选覆盖审计，而不是立刻扩大训练量：

1. 冻结一个不读 SDF/Render JSON 的 primitive proposer；
2. 在未参与调参的图片上测 GT Bond carrier recall，目标先设为 99.9% 以上；
3. 分桶统计文字附近、单双三键、交叉、长短键和颜色条件下的候选图类；
4. 用重复渲染和真实扰动估计 score interval，不以 softmax 置信度直接代替误差界；
5. 只有 GT 图始终在候选集合内，且唯一输出均满足严格区间分离，才训练视觉 scorer；
6. scorer 训练目标是给候选解释排序和校准区间，不再让一个序列模型同时发明顶点、文字和全部边。

当前结论是：数学路线可行，有限规格通过；真实候选覆盖尚未验证，所以仍是 `NO-GO` for training。
