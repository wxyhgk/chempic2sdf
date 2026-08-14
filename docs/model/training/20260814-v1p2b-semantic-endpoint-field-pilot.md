# v1 P2-B 语义键端点场 Pilot

> 日期：2026-08-14
>
> 状态：已预注册，等待训练
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
