# v1 P2-D1 Listwise Pair Scorer Pilot

> 日期：2026-08-14
>
> 状态：已预注册，等待训练
>
> 模型：`chemvector-v1p2d1-listwise-pair-scorer-pilot`

## 目标

P2-D0 已证明每端 top8 partner union 在不裁剪长键时具有 95.91% pair recall。P2-D1 冻结
P1-A、P2-B endpoint 和 top8 候选生成协议，只训练一个轻量 scorer 判断候选端点对是否对应
一根真实 BondObject。

本轮只学习键存在性，不预测单双三/芳香键级，不接 OCR、AtomObject 或 SDF。先证明候选排序和
定基数可学，再进入键级头，避免把多个问题混在同一次训练里。

## 输入与模型

每个候选 pair 使用 64 维特征：

- 9x48 bond-probability ribbon 的 48 维纵向最大响应；
- 9 维横向平均响应；
- 整段均值、20% 下分位数和 `probability >= 0.5` 覆盖率；
- 两端 endpoint 分数的最小值和最大值；
- 归一化长度；
- 中间 endpoint blocker 支持。

scorer 是 `LayerNorm -> 64x128 -> 128x128 -> 128x1` MLP，共 25,089 参数。候选生成固定使用：

- P1-A separator SHA256
  `490b5daae3a2059c3d335785a69e0c2a41eed32f0237ef77fc3556c36d134a74`；
- P2-B endpoint SHA256
  `3ec7bf27e96f166e0f30e7c5c8b79f33860701acfeb3e16297015fc7a91511ca`；
- endpoint threshold 0.05、每图最多 256 endpoint；
- `line_score` 每端 top8 partner union；
- 不设置最大键长、空间半径或固定近邻数。

推理分数排序后，以 4px 量化的无序 segment key 合并近重复 pair。该 NMS 不读取 GT、原子价态
或化学规则。

## 监督与损失

候选 pair 的两个端点无序匹配同一 GT BondObject visible ports，且两端误差均不超过 6px 时，
标记该 GT bond ID；否则为负例。同一 GT 键允许有多个几何近邻候选，训练目标是至少选中一个。

总损失为：

- 正负不平衡 BCE，positive weight 上限 16；
- 权重 0.5 的 listwise hard-negative loss：每根可达 GT 键的最高候选 logit 必须超过整图最难
  负例至少 1.0 margin。

候选阶段不可达的 GT 键不伪造负标签，但最终 recall/exact 仍按全部 GT 键计算。

## 数据隔离

P2-B 的 2,014 个 fit 分子重新按 `SHA256(20260820:molecule_id)` 排序：

- pair fit：1,914 个分子、7,656 张；
- pair calibration：100 个分子、400 张；
- P2-B holdout：100 个分子、400 张，训练期间不读；
- P2-B route-select calibration 不用于本轮训练或选参；
- validation/internal test/MR-TADF/USPTO 均不读取。

训练三轮，逐图构造候选，AdamW、学习率 `3e-4`、weight decay `1e-4`、bfloat16。每轮在 pair
calibration 的固定 `0.05..0.95` 阈值网格选择 presence threshold；checkpoint 依次按 known
image exact、known pair recall、predicted pair F1 选择。winner 和阈值冻结后，只读取一次
P2-B holdout。

## 指标

- known pair recall：给定真实键数，从 NMS 后取相同数量最高分候选，计算恢复的不同 GT 键比例；
- known image exact：上述 known-count 结果覆盖整图全部 GT 键；
- predicted precision/recall/F1：使用冻结阈值，重复命中同一 GT 键只计一次正确，其余算额外 pair；
- predicted count accuracy：阈值后 pair 数等于 GT 键数；
- predicted image exact：pair 数正确且覆盖全部 GT 键。

## GO/NO-GO

一次性 400 张 P2-B holdout 同时满足：

- known pair recall 至少 90%；
- known image exact 至少 20%；
- predicted pair F1 至少 80%；
- predicted count accuracy 至少 20%；
- predicted image exact 至少 10%；
- stress known pair recall 至少 85%；
- crossed-bonds known pair recall 至少 75%；
- P1-A、P2-B 和 top8 候选协议全部冻结；
- validation/internal test 读取数均为 0。

GO 仍只是键存在性组件成立。下一周期才能在冻结 pair 上增加 display/chemical type，并最终聚类
ports 生成 AtomObject；不能用 GT 键数或 GT 类型编译结果冒充端到端模型。
