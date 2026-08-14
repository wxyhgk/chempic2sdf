# v1 P2 Semantic BondObject Pilot

> 日期：2026-08-14
>
> 状态：已预注册，等待训练
>
> 模型：`chemvector-v1p2-semantic-bondobject-pilot`

## 目标

冻结已经通过 validation/internal test 的 P1-A 语义前端，只把预测 bond 概率图交给新模型。
模型不读取混合 RGB feature、文字图、原子候选或 GT bond layer，不枚举 AtomPair。每根化学键
只生成一个 BondObject：中心、无向双角、半长度、两个 visible ports、display/chemical type
和 direction。双键、三键内部的两/三条墨线不得生成额外端点或额外 BondObject。

## 架构与标签

- 输入：冻结 P1-A 的连续 bond probability，checkpoint SHA256
  `490b5daae3a2059c3d335785a69e0c2a41eed32f0237ef77fc3556c36d134a74`；
- stride 4/8/16 轻量 CNN 金字塔；
- 空间中心格固定对象身份，每格 3 个无序 slot；
- 多尺度 DenseBondObject 头输出完整对象，不使用自由 learned query；
- ChemScene v0.2 `BondObject.ports` 是几何真值，文字原子的 port 位于文字 bbox 边界，不用
  bbox 中心或 atom anchor 代替；
- 同中心多键只在格内做 Hungarian slot 匹配。

## 数据协议

P0-R safe manifest 保持固定。train 的 2,214 个完整四布局分子按
`SHA256(20260816:molecule_id)` 排序：

- 前 100 个分子、400 张作为 calibration；
- 其余 2,114 个分子、8,456 张训练三轮；
- 每轮只在 calibration 的固定 `0.05..0.95` 网格选择 presence threshold；
- checkpoint 按 predicted typed exact、known typed exact、geometry F1 依次选择；
- checkpoint 和 threshold 冻结后，496 张 validation 只读取一次；
- 已消费的 internal test、MR-TADF、USPTO 均不读取。

训练 batch 4、AdamW、学习率 `2e-4`、bfloat16；P1-A 全部参数冻结。

## GO/NO-GO

固定 validation 同时满足：

- known-count port-pair geometry accuracy 至少 90%；
- known-count geometry + display/chemical typed accuracy 至少 85%；
- known-count typed image exact 至少 20%；
- predicted count accuracy 至少 50%；
- predicted typed image exact 至少 15%；
- predicted geometry F1 至少 88%；
- text-adjacent BondObject geometry accuracy 至少 85%；
- stress known geometry accuracy 至少 85%。

这些门显著高于旧混合图空间 BondObject 的 67.55% object accuracy、3.01% known exact 和
约 2% SDF exact。GO 仍只代表 BondObject 组件成立；下一周期才聚类 ports 生成 AtomObject，
不能用 GT Atom 编译结果冒充端到端识别。
