# v1 P2-D1 Listwise Pair Scorer Pilot

> 日期：2026-08-14
>
> 状态：已完成，严格 NO-GO
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

## 实际执行

- pair fit：1,914 个分子、7,656 张图；
- pair calibration：100 个分子、400 张图；
- P2-B holdout：100 个分子、400 张图；
- 三轮共呈现 22,968 个训练样本；
- scorer 参数量：25,089；
- 总耗时：674.79 秒；
- validation/internal test 读取数均为 0。

三轮 calibration 如下：

| Epoch | Train loss | 阈值 | Known recall | Known exact | Predicted F1 | Count exact | Predicted exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5895 | 0.95 | 72.16% | 0% | 66.61% | 0.25% | 0% |
| 2 | 0.5177 | 0.95 | 72.05% | 0% | 63.42% | 0.50% | 0% |
| 3 | 0.4993 | 0.95 | 71.45% | 0% | 63.02% | 0.25% | 0% |

loss 持续下降，但 known-count 排序和 predicted F1 同时回退。按预注册规则，epoch 1 是正式
winner；不能用更低训练 loss 选择后两轮。

## 一次性 Holdout

冻结 epoch 1 和阈值 0.95 后，一次性读取 400 张 P2-B holdout，共 14,016 个 GT BondObject：

| 指标 | 结果 | 门槛 | 判定 |
| --- | ---: | ---: | --- |
| Known pair recall | 73.04% | 90% | 失败 |
| Known image exact | 0% | 20% | 失败 |
| Predicted pair F1 | 66.85% | 80% | 失败 |
| Predicted count accuracy | 0.50% | 20% | 失败 |
| Predicted image exact | 0% | 10% | 失败 |
| Stress known pair recall | 67.61% | 85% | 失败 |
| Crossed-bonds known pair recall | 57.41% | 75% | 失败 |

predicted precision/recall 为 53.91%/87.95%。每图 top8 原始候选均值 1,013.49，经 4px 量化
segment NMS 后仍有 357.15 个。全部预注册能力门失败，最终判定
`listwise_pair_scorer_no_go`。

## 结论

P2-D0 在相同候选协议上的 Oracle recall 为 95.91%，P2-D1 winner 的 known-count recall 只有
73.04%，相差 22.87pp。说明真 pair 大多仍在输入中，失败发生在学习式排序，而不是候选生成。
三轮曝光增加没有收窄差距，因此不支持扩大同一 MLP 的层数、数据或轮数。

根因是每个 pair 被独立编码和打分。即使 listwise loss 让每根 GT 键的最佳候选超过最难负例，
模型仍看不到多个 pair 共享 endpoint、多个近似正候选代表同一 GT 键，以及整图需要同时覆盖
不同键而不能重复占用同一区域。阈值只能在高 recall 和大量额外 pair 之间移动，无法学出稳定
cardinality。

下一周期停止独立 pair MLP。P2-D2 应把 top8 candidate graph 作为一个集合：pair 通过共享
endpoint 做消息传递，显式预测 endpoint keep/incident degree 和 edge existence，并加入整图
cardinality/coverage 目标。先做一个小型 graph-set capacity 实验，验证它能否在固定候选上逼近
P2-D0 Oracle；未达到 known recall 90% 前，不增加键级或接 SDF。

## 产物

- checkpoint：`outputs/checkpoints/chemvector-v1p2d1-listwise-pair-scorer-pilot/best.pt`；
- checkpoint SHA256：`37e0fe8227d0f226c5476cdbec391a814ce634daf1f62c4e4316cd4073567935`；
- result：`outputs/checkpoints/chemvector-v1p2d1-listwise-pair-scorer-pilot/result.json`；
- result SHA256：`c4c85400325fb9dbe4d59c2417cd881e665741e142e9d9b6bcdcc6d822cdaa63`。
