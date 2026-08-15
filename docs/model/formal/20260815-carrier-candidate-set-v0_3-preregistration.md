# Carrier Candidate Set v0.3 预注册

日期：2026-08-15

实施结果已单独记录在 [Carrier Candidate Set v0.3 结构验证结果](20260815-carrier-candidate-set-v0_3-structural-validation.md)。
本文保留为实施前假设和门禁，不用后验结果覆盖预注册内容。

## 结论先行

v0.2 只能验证已经给定的物理 segment、terminal 和 Bond lane 是否自洽，不能验证这些 terminal
属于哪个预测 Atom。把真实 `A-B` 的载体错误接到 `A-C`，仍可能得到几何合法的 v0.2 输出。因此
v0.2 纯适配器继续保留为编译器和 Oracle 上界，但不能作为完整 PNG proposer。

v0.3 正式路线采用：

```text
有限 Q4 候选域
  + 冻结可判定 feasibility predicate
  + 完整枚举证书
  + 多 interpretation candidate set
  + 超容量或超表达域 closed reject
```

本阶段没有 `UniqueGraph`、没有 SDF、没有 winner scorer，也不授权训练。

```text
blind_sample_selection_authorized=false
unique_graph_authorized=false
proposer_public_execution_authorized=false
training_authorized=false
```

## 为什么必须改协议

### AtomPair provenance 丢失

v0.2 的 `TerminalHint.id` 最终只用于端点等价类，输出 terminal center 由物理 endpoint 重算，
prediction 不保存 Atom hypothesis、port 或 unordered AtomPair。独立 trace 即使记录 AtomPair，也只是旁证，
冻结 validator 并未约束它。

v0.3 必须把下列关系放进被验证对象：

```text
stroke endpoint -> port -> atom
lane -> begin/end port
Bond -> canonical unordered AtomPair
```

### 贪心配对会删除反事实

粗单键的两条 boundary、双键的四条 boundary、X crossing、文字 `H/N/A/I/-/=` 的笔画都可能有
多个局部合法 pairing。只保存 greedy winner 会删除证明 `Ambiguous` 所需的反事实，错误 winner 仍可能
通过几何 validator。

v0.3 只在冻结的有限 boundary 域和可判定 `FeasiblePair` 上谈“全部”。连续空间中的所有 centerline
无法有限枚举，也不得伪称完整。

## 四层对象

### 1. Boundary

Boundary 是从最终 768x768 canonical raster 提出的有限物理边界候选。必须保存：

- 稳定 ID；
- Q4 两点几何；
- 非空、规范排序的最终 raster foreground support RLE；
- 非空、规范排序的 P1 bond-mask support RLE；
- 输入、预处理、P1 checkpoint、阈值和 extractor digest 绑定。

### 2. Stroke 与 fragment

`StrokeHypothesis` 把一个或两个 boundary 解释成一个 primitive stroke，保存 boundary member IDs 和
canonical centerline。`Lane` 与 `fragment` 必须分开：一条长线被 LSD 切成两段时，它们是同一 lane 的
有序 fragments，不能因 fragment 数为 2 就推成双键。

当前 schema 显式区分：

```text
straight
glyph
charge
unknown
```

v0.3 第一版只把 `straight` 编成 Bond lane；`glyph` 和 `charge` 只进入对应 ownership，`unknown` 不能进入
Bond。楔键、虚线整体、芳香圆和曲线必须 closed reject 为 `unsupported_primitive`，不能静默降成普通单键。

### 3. Ownership、Atom 与 port

每个 interpretation 独立保存：

```text
stroke -> bond_lane | text_glyph | marker | charge | residual
```

ownership 不能放在全局，因为同一笔画像是文字还是键本身可能有歧义。一个 interpretation 内，除具名
intersection region 外，foreground support 必须精确单 owner；共享区域必须有 `crossing/junction/touch`
taxonomy，且 validator 重新核对几何交集。

Atom hypothesis 必须区分 `explicit_label` 与 `implicit_vertex`。显式文字原子的 port 不是 bbox 中心；
每个 port 保存 atom ID、terminal region 和对齐的 stroke endpoint refs。

### 4. Lane、Bond 与 interpretation

Lane 保存：

- physical lane ID；
- 1..64 个有序 fragment stroke IDs；
- begin/end port ID；
- 多 lane 时引用 final-raster white-gap certificate。

Bond 保存：

- 1..3 个 physical lane IDs；
- canonical `atom_pair=[min_atom_id,max_atom_id]`；
- displayed order 必须等于 lane 数，不能等于 fragment、boundary 或 dash 数。

每个 interpretation 保存完整 selected atoms/ports/strokes/lanes/bonds、ownership、residual 和具名共享区域。
不同 interpretation 可以复用同一输入 boundary/support，但每个 interpretation 内必须独立满足全部所有权和
引用合同。

## 完整枚举合同

候选域和 predicate 在执行前冻结。枚举器必须输出：

- 全部 feasible boundary pairing；
- 每个 endpoint 门内全部 Atom/port attachment，不使用 nearest argmin；
- 全部合法 lane fragment chain；
- 全部 1..3 lane Bond grouping；
- bond/text/charge/residual 的全部可行 ownership；
- crossing 与 split-junction 两种可行解释；
- 所有组合得到的 interpretation，直到冻结容量。

以下行为一律是 `INVALID`，不是普通低置信度：

- greedy、top-k、beam 或 score-prune；
- 达到容量后静默截断；
- `all_candidates_emitted=true` 但遗漏一个冻结 predicate 判定为 feasible 的候选；
- 用 Direct/P1 分数删除逻辑上可行的反事实；
- 用 RDKit sanitize 删除视觉候选。

超过冻结容量时只能 `Reject(outside_capacity)`。不存在支持 boundary 时只能
`Reject(no_supported_carrier)`；楔键、曲线等当前不能无损表达时只能
`Reject(unsupported_primitive)`。

## 决策边界

v0.3 输出只有：

```text
CandidateSet(1..128 complete interpretations)
Reject(reason)
```

没有 `UniqueGraph`。独立 verifier 后续对 interpretations 做带 Atom/port/Bond 属性的图同构商：

- 存在两个不等价可行图类：必须 abstain；
- 只有一个图类但没有独立 coverage certificate：仍不能授权唯一图；
- coverage certificate 绑定完整候选 bytes，且所有 surviving interpretations 属于一个图类：未来版本才可
  降成 UniqueGraph。

## 必须形式化的命题

Lean v0.3 至少证明：

```text
pairEnumerator_sound
pairEnumerator_complete
boundaryClass_iff_sameStroke
ownership_exclusive
portComponent_iff_sameAtomClass
bondLanes_exactFiber
bondOrder_eq_laneCard
interpretation_exactOwnership
inequivalentFeasibleCandidates_force_abstain
reject_emits_noGraph
```

这些是条件定理：它们证明冻结候选域内的编译与拒绝不会越权，不证明视觉候选等于真实绘图意图。

## 数学不能代替的公开审计

公开数据必须独立报告：

- boundary proposer precision/recall；
- boundary -> stroke pairing precision/recall；
- bond/text/charge ownership confusion；
- Atom/port 定位和 endpoint attachment coverage；
- lane fragment chain exact；
- 1/2/3 lane grouping/order exact；
- GT graph candidate coverage；
- candidate graph class count；
- accept coverage 与 accepted-only error（未来 decision 层）；
- overflow/reject/exception，全部留在分母中。

## 公开奖励和否决门

### 合成反唯一门

以下同像 fixture 必须保留至少两个非同构 graph class，不能靠分数变 singleton：

- crossing vs center junction；
- double vs 两条平行 single；
- triple vs triple/double+single/三条 single 分组；
- glyph/charge `- I l 1 H N A =` vs Bond；
- 长 Bond vs 两条共线 Bond 加不可见二价 C；
- fragment chain vs 多条短 Bond；
- wedge vs ordinary single；
- aromatic circle vs glyph/marker；
- 多字符 label 的多个 port；
- touch vs junction。

每桶 `false singleton = 0` 是硬门。

### 合成正例门

single、double、triple、chain、fan、ring、长键的真实解释必须全部出现在 candidate set；输入顺序、
boundary ID 重命名、无向端点交换后 canonical bytes 不变；同机 CPU 重放逐字节一致。

### 已暴露图片门

已暴露 MR-TADF 只用于 proposer coverage 诊断，不选择 winner：

```text
supported carrier precision >= 0.99
GT carrier recall per image >= 0.95
GT carrier recall micro >= 0.99
GT graph in candidate set = 100% for any future accepted singleton
false singleton = 0 in every difficulty bucket
```

上述平均门不能授权唯一图。一个 100 键分子 95% recall 仍是错误整图，只有 candidate coverage 和
false-singleton 门能约束未来 accept。

## 下一步

1. 冻结 v0.3 input/candidate-set closed schema；
2. 实现数据无关 validator 和 mutation tests；
3. 完成 Lean v0.3；
4. 构造合成正例与反唯一 fixture；
5. 实现不训练的公开 candidate enumerator；
6. 公开 coverage 门通过前，训练保持关闭。
