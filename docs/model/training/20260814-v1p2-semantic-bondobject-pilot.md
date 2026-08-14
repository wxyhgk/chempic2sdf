# v1 P2 Semantic BondObject Pilot

> 日期：2026-08-14
>
> 状态：已完成，严格 NO-GO
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

## 实际执行

- fit：2,114 个分子、8,456 张图；
- calibration：100 个分子、400 张图；
- 每轮样本呈现 8,456 次，三轮合计 25,368 次；
- validation：496 张，只在 checkpoint 和阈值冻结后读取一次；
- internal test：读取 0 张，继续保持封存；
- 参数量：1,081,593；
- 总耗时：1,149.59 秒。

三轮 calibration 学习曲线如下。`exact` 均按整张图片全部 BondObject 正确计算。

| Epoch | Train loss | 阈值 | Known geometry | Known typed | Predicted count | Predicted geometry F1 | Known/Predicted typed exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.8363 | 0.30 | 34.23% | 23.76% | 0.25% | 47.28% | 0% / 0% |
| 2 | 2.2677 | 0.30 | 41.27% | 32.03% | 0.75% | 50.92% | 0% / 0% |
| 3 | 1.9354 | 0.35 | 44.59% | 35.44% | 1.25% | 56.33% | 0% / 0% |

Epoch 3 按预注册排序规则成为 winner，随后冻结 checkpoint 和 presence 阈值 0.35。

## Validation 结果

一次性读取 496 张 validation，共 18,244 个 BondObject：

| 指标 | 结果 | 门槛 | 判定 |
| --- | ---: | ---: | --- |
| Known-count geometry accuracy | 45.58% | 90% | 失败 |
| Known-count typed accuracy | 35.43% | 85% | 失败 |
| Known-count typed image exact | 0% | 20% | 失败 |
| Predicted count accuracy | 0.20% | 50% | 失败 |
| Predicted typed image exact | 0% | 15% | 失败 |
| Predicted geometry F1 | 56.48% | 88% | 失败 |
| Text-adjacent geometry accuracy | 45.64% | 85% | 失败 |
| Stress known geometry accuracy | 41.94% | 85% | 失败 |

Predicted geometry precision/recall 为 44.82%/76.35%。四种布局的 predicted geometry F1
分别为 coordgen 58.41%、original 57.28%、RDKit 58.09%、stress 51.60%。stress 中
`crossed_bonds` 最差，known geometry 38.01%、predicted geometry F1 45.35%。所有预注册
能力门均未通过，最终判定为 `semantic_bondobject_no_go`。

## 结论

P1-A 已证明 bond/text 语义分离可行，本轮输入也只使用预测 bond probability，因此当前失败
不能归因于文字笔画污染。三轮 calibration 的 geometry 从 34.23% 稳步升至 44.59%，说明模型
学到了局部几何信号；但 validation 与 90% 门相差 44.42pp，count 几乎完全失败，所有整图
exact 为 0，不能用追加一轮、放宽 6px 匹配或降低门槛来解释。

本轮否定的是“以中心格和固定 slot 同时承担 presence、方向、长度以及两个远端 visible port”
的 Dense center regression 参数化，不是否定 bond-first 路线。尤其是文字边界裁剪会让两个
visible port 关于中心不对称，中心、双角方向和半长度并不是稳定的完整对象坐标。

下一周期停止沿用该头。先在 train-only 新建分子级 calibration/holdout，直接从 P1-A bond
probability 预测两个 endpoint heatmap/向量场，再用沿线连通支持配对为 BondObject；双键、三键
和芳香键仍通过 group/type head 合成一个对象。第一步只做 Oracle bond-count 或 endpoint-pair
上限审计，确认端点表示显著超过本轮 45.58% known geometry 后，才允许训练完整 P2-B。

## 产物

- checkpoint：`outputs/checkpoints/chemvector-v1p2-semantic-bondobject-pilot/best.pt`；
- checkpoint SHA256：`4f7f0f1de5e0e63c926b2bfe49951828d210abe930b54e0d92596178bd7aca8e`；
- result：`outputs/checkpoints/chemvector-v1p2-semantic-bondobject-pilot/result.json`；
- result SHA256：`a6c835b37f65edd76bddeda3f2ab2e37e2ef59afdcb226762891588847eb5ba2`。
