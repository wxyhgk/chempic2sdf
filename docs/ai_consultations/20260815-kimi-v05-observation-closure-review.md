# 2026-08-15 Kimi：v0.5 Observation Closure 评审

- 咨询对象：Kimi Code
- 会话 ID：`session_c3fcd39b-9e85-4745-bcfd-62b19545bb1f`
- 任务：独立审查 RasterGraphSet/EvidenceGraphSet 闭包、有限域、冻结顺序、反作弊门和 Lean 边界
- 文件修改：Kimi 只读代码和文档，没有修改仓库

## 输入问题

项目已证明 v0.4 存在 S8/C4 exact canonical PNG collision。拟议 v0.5 定义有限 Scene 域 D，先按
Raster fiber 合并图，再按冻结 extractor 的 Evidence fiber 二次合并；Candidate 必须与完整
EvidenceGraphSet 按图同构精确相等。要求 Kimi 从第一性原理回答该定义是否修复矛盾，以及 sampled
closure、完整域、commit/reveal、真空通过和形式化边界。

## Kimi 的主要结论

Kimi 认为路线在定义上正确：v0.4 的 label 是 provenance/生成顺序的函数；v0.5 把 label 改为 Evidence
fiber 的函数。S8/C4 相同 raster 会得到 `{two_lanes, text}` 并标为 Ambiguous，不再由先到 family 决定。

它同时给出四个硬前提：

1. 图同构必须保留节点标签、电荷、键级和断连结构；
2. 真值只相对于完整声明域 D 成立；
3. extractor 必须在构造 Evidence fiber 前冻结且确定；
4. `seen_png` 不能继续从 motif 意图产生 label，label 必须只来自 closure certificate。

Kimi 特别强调 sampled closure 的单向性：sampled Ambiguous 在扩域后仍 Ambiguous，但 sampled Graph 可能
因未采到的冲突变 Ambiguous。因此 sampled closure 不能签发 Graph 标签。

## 与三个独立 agent 的共识

四方一致认为：

- 最终 oracle 是 EvidenceGraphSet，不是 RasterGraphSet；
- Hash 只能索引，class equality 要比较完整 canonical bytes；
- 必须双实现 closure 和图同构审计；
- valid domain 内 Reject 必须为 0；
- Ambiguous 必须输出完整显式候选集合；
- 候选超过 cap、完整枚举失败或版本乱序都应 NO-GO；
- Lean 可证明 fiber/单调性/三态合同，不能证明 Python、域覆盖或操作时序。

## 主审裁决

Kimi 有两项建议没有被设为硬门：

1. 它建议 Evidence 到 family 的预测接近随机。合法视觉几何本身可能让 family 可预测，因此该指标只能
   诊断；硬门应是 Candidate 进程只读 Evidence，显式 family/style 元数据置换不改变 closure。
2. 它建议 Graph class 比例接近旧 grammar 意图。完整闭包可能诚实地把大量旧 strict family 变成
   Ambiguous，不能为了比例修改真值。v0.5 改用预注册 domain adequacy 下限；不足则整个 domain NO-GO。

此外，Kimi 提议对域外输入设置 Reject 率门。该方向对生产有价值，但 v0.5 当前只验证有限域内闭包，
域外分布尚未定义，因此先保留为后续现实桥接，不混入本轮 exact closure 成绩。

## 项目结论

v0.5 不是继续优化线条阈值，而是修正标签的数学来源。优先级是：

```text
完整有限域
-> Raster closure
-> 带标签图同构
-> extractor freeze
-> Evidence closure
-> Candidate
```

在这些门完成前维持 `training_authorized=false`。
