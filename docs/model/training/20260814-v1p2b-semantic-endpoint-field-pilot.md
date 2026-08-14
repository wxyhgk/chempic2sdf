# v1 P2-B 语义键端点场 Pilot

> 日期：2026-08-14
>
> 状态：已完成，严格 NO-GO
>
> 模型：`chemvector-v1p2b-semantic-endpoint-field-pilot`

## 为什么做这个实验

P2 的中心格需要同时预测整根键的中心、方向、长度和两个远端 visible port。三轮后 validation
known geometry 只有 45.58%，说明这种长距离耦合参数化没有成立。P2-B 不退回 Atom-first，
也不枚举 AtomPair，而是先回答一个更小的问题：冻结的语义 bond map 能否可靠产生完整的
visible port 集合。

每个 port 只由所在 stride-4 格和格内二维偏移表示。共享原子处完全相同的 port 合并为一个
集合元素；同格不同 port 用最多 3 个无序 slot 表示。slot 不预测另一端、键长或键级，因此
不会因为长键把监督变成长距离坐标回归。

## 零训练可表示性审计

在 400 张 train-only calibration 上共有 28,720 个 BondObject 端点出现；共享端点合并后为
16,077 个唯一 port。stride-4 网格只有 11 张存在同格不同 port，共 16 个额外 slot，最大同格
数量为 3。因此 3-slot 标签可以无损表示已审计样本，不能把后续错误归因于单格容量不足。

## 模型与损失

- 输入只使用冻结 P1-A 的连续 bond probability；
- P1-A checkpoint SHA256 为
  `490b5daae3a2059c3d335785a69e0c2a41eed32f0237ef77fc3556c36d134a74`；
- stride 4/8/16 轻量 CNN，多尺度特征只在 stride-4 输出前融合；
- 每格 3 个无序 port slot，每个 slot 输出 presence 和格内二维偏移；
- 同格 slot 按局部偏移代价做 Hungarian 匹配；
- 损失由 focal heatmap、整图正负峰 margin、格内 count 和局部 offset 组成；
- 推理对每个 slot 做 3x3 局部极大值，最多保留 256 个 port。

该模型不输出 BondObject 配对、键级、AtomObject 或 SDF。

## 数据隔离

P0-R safe manifest 保持固定，只使用其 train split。2,214 个四布局分子按
`SHA256(20260817:molecule_id)` 固定排序：

- fit：2,014 个分子、8,056 张；
- calibration：100 个分子、400 张，用于逐轮选 checkpoint 和固定 presence 阈值；
- holdout：100 个分子、400 张，只在 winner 冻结后读取一次；
- validation 读取 0 张；
- 已消费的 internal test、MR-TADF 和 USPTO 均不读取。

训练三轮，batch 4、AdamW、学习率 `2e-4`、bfloat16。checkpoint 依次按 calibration 的
known pairable-bond recall、known unique-port recall 和 predicted unique-port F1 选择。

## 指标定义

- known unique-port recall：给定真实唯一 port 数量，取同样数量的最高分预测，与 GT 做一一
  匹配后，6px 内正确 port 的比例；
- known pairable-bond recall：某个 GT BondObject 的两个 port 都在上述一一匹配中恢复，表示
  它具备进入后续配对阶段的几何上限；这不代表模型已经判断出二者属于同一根键；
- known unique-port exact：整张图的 port 数量和位置全部正确；
- predicted unique-port F1/count：使用 calibration 冻结的 presence 阈值，不提供真实数量；
- text-adjacent/stress：分别报告文字邻接键和 stress 布局的 pairable-bond recall。

## GO/NO-GO

一次性 train-only holdout 必须同时满足：

- known unique-port recall 至少 95%；
- known pairable-bond recall 至少 90%；
- known unique-port exact 至少 30%；
- predicted unique-port F1 至少 88%；
- predicted count accuracy 至少 30%；
- text-adjacent pairable-bond recall 至少 85%；
- stress known pairable-bond recall 至少 85%；
- stride-4 同格唯一 port 数不超过 3；
- P1-A 保持冻结，validation/internal test 读取数均为 0。

GO 只证明 endpoint set 是比 P2 center regression 更合适的中间表示。下一周期仍需用沿线连通
支持完成 port 配对，并用 group/type head 把双键、三键和芳香键合成为一个 BondObject。若本轮
NO-GO，则先分析漏 port 是 junction、文字边界还是峰选择问题，不追加 P2 center regression。

## 实际执行

- fit：2,014 个分子、8,056 张图；
- calibration：100 个分子、400 张图；
- holdout：100 个分子、400 张图；
- 三轮共呈现 24,168 个训练样本；
- 模型参数量：356,713；
- 总耗时：500.11 秒；
- validation/internal test 读取数均为 0。

三轮 calibration 如下：

| Epoch | Train loss | 阈值 | Known port recall | Known pairable bond | Predicted port F1 | Predicted pairable bond |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.1315 | 0.35 | 59.99% | 38.54% | 60.93% | 78.02% |
| 2 | 0.9660 | 0.35 | 58.24% | 32.85% | 59.61% | 76.80% |
| 3 | 0.9330 | 0.35 | 57.44% | 36.35% | 57.75% | 81.15% |

按照预注册排序，epoch 1 是正式 winner。loss 持续下降但 known-count 峰排序没有同步改善，不能
用训练损失代替集合恢复指标。

## 一次性 Holdout

冻结 epoch 1 和阈值 0.35 后，一次性读取 400 张 train-only holdout，共 15,616 个唯一 port、
14,016 个 BondObject：

| 指标 | 结果 | 门槛 | 判定 |
| --- | ---: | ---: | --- |
| Known unique-port recall | 59.73% | 95% | 失败 |
| Known pairable-bond recall | 39.85% | 90% | 失败 |
| Known unique-port exact | 0% | 30% | 失败 |
| Predicted unique-port F1 | 61.10% | 88% | 失败 |
| Predicted count accuracy | 0.75% | 30% | 失败 |
| Text-adjacent pairable-bond recall | 32.41% | 85% | 失败 |
| Stress known pairable-bond recall | 35.53% | 85% | 失败 |

holdout 的 predicted port precision/recall 为 47.24%/86.48%，predicted pairable-bond recall
为 79.37%。同格容量审计通过：只有 6 张出现碰撞、额外 slot 11 个，最大 occupancy 为 2。
因此失败不是 stride-4 标签碰撞或 3-slot 容量不足。

## 诊断结论

本轮严格判定 `endpoint_field_no_go`。独立 endpoint heatmap 没有解决全图峰排序和定基数：
known-count 只能找回约 60% port，整图 exact 为 0，count exact 近乎为 0。三轮增加曝光没有改善
这些指标，因此不能沿相同损失继续扩数据或轮数。

但 calibration 中存在一个明确的候选上限信号。正式 winner epoch 1 在未选用的 0.05 阈值下，
predicted port recall 为 98.68%、pairable-bond 上限为 98.15%，precision 只有 20.63%；epoch 3
对应值为 98.88%/98.42%/23.35%。这组 post-hoc 指标不能替换本轮 winner 或改变 NO-GO，
但说明大多数真 port 已存在于宽候选集，断点主要是独立峰排序，而不是坐标场完全看不到端点。

下一周期不再要求 endpoint 头独立决定最终数量。P2-C 应在新的协议中冻结宽 endpoint 候选，
用一根键的两端和沿线 bond 支持联合选择候选对，并显式抑制“跨过中间端点”的 A-C 假键；先做
train-only pair-candidate Oracle，确认真实 BondObject 是否进入可控大小的候选对集合，再决定
是否训练配对头。双键、三键仍是一根语义 BondObject，不能按墨线数量制造端点。

## 产物

- checkpoint：`outputs/checkpoints/chemvector-v1p2b-semantic-endpoint-field-pilot/best.pt`；
- checkpoint SHA256：`3ec7bf27e96f166e0f30e7c5c8b79f33860701acfeb3e16297015fc7a91511ca`；
- result：`outputs/checkpoints/chemvector-v1p2b-semantic-endpoint-field-pilot/result.json`；
- result SHA256：`a24e242ecf4268eebe8a1746a0121d991fb0fc6268fb7e37468c25547598e1cc`。
