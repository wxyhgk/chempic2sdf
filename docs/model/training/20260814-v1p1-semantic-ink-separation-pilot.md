# v1 P1 语义墨迹分离 Pilot

> 日期：2026-08-14
>
> 状态：NO-GO；固定 0.5 阈值召回不足
>
> 模型：`chemvector-v1p1-semantic-ink-separation-pilot`

## 目标

旧空间 BondObject 已证明“键作为完整对象”具备容量，但直接读取混合图片时，文字笔画仍进入
Bond 特征，9k scaffold 独立验证的整图 exact 只有约 2%。本轮先解决更前置的问题：让同一个
可微视觉前端从原始 768×768 图片预测两个独立通道：

- bond ink：只保留键、楔键和芳香键等视觉证据；
- text ink：只保留元素、显式氢、电荷和缩写文字。

两个通道允许在抗锯齿边界少量重叠，不做互斥三分类。后续 BondObject 只消费预测 bond feature，
OCR 只消费 text feature；模块仍可联合微调，因此不否定端到端训练。

## 数据边界

- 固定 P0-R safe manifest SHA256：
  `eb9da30c85aaede01b635294f53dca4454825f13db55002cb2df843edb49f11b`；
- 训练只读 8,856 张 train；
- checkpoint 选择只读 496 张 validation；
- 496 张 internal test 不传给训练进程；
- 不读取 MR-TADF、USPTO 或任何模型预测；
- 不新增模糊、低分辨率或线宽增强，保持原生 768×768。

## 模型与损失

使用约百万参数的轻量 U-Net：stride 2/4/8/16 编码，逐级恢复到原始分辨率，输出两个独立
logit。损失为稀疏前景加权 BCE、双通道 soft Dice 和联合前景重建。它不预测 Atom，不枚举
AtomPair，也不按线宽或像素条数决定单双三键。

训练固定两轮、AdamW、学习率 `3e-4`、batch 4、bfloat16；每轮完整读取 8,856 张。最佳
checkpoint 只按 `min(bond IoU, text IoU)` 选择。

## GO/NO-GO

在 496 张 validation 上同时满足：

- bond IoU 至少 90%；
- text IoU 至少 75%；
- bond/text 联合前景 IoU 至少 90%；
- stress 子集 bond IoU 至少 88%；
- GT text-only 像素被误报为 bond 的比例不超过 10%；
- internal test 读取数严格为 0。

GO 只表示文字与键的视觉解耦可泛化。下一周期才把预测 bond feature 接到“每根化学键一个
BondObject”的端点/键级头；NO-GO 时先检查分辨率和损失，不允许退回 Atom-first 或根据
MR-TADF 调阈值。

## 实际运行

预注册提交为 `c943720`。首批运行发现 AMP 不允许 sigmoid 后 BCE，修为等价显式对数；第二次
运行发现 bfloat16 的 `1 - 1e-5` 舍入为 1，loss 从首批即为 NaN，立即停止且没有 checkpoint。
最终修复只将损失计算提升为 float32，并加入非有限 loss fail-fast；模型前向、数据、门槛和
优化参数未改变。修复提交为 `9b6c1ed`、`0457aa3`。

正式训练读取两轮完整 train，共 17,712 次样本呈现，耗时 345.66 秒。模型参数 1,341,922。
最佳 checkpoint 为 epoch 2：

`outputs/checkpoints/chemvector-v1p1-semantic-ink-separation-pilot/best.pt`

SHA256：`e4f726b80f73e46a71619f8f4a42ff3b0042a3156cc86c0a4f584154b574263c`。

## 固定 0.5 结果

| Validation 指标 | Epoch 1 | Epoch 2 |
|---|---:|---:|
| Bond IoU | 72.54% | 71.73% |
| Bond precision / recall | 98.96% / 73.10% | 99.60% / 71.94% |
| Text IoU | 61.81% | 65.09% |
| Text precision / recall | 99.56% / 61.98% | 99.78% / 65.18% |
| Foreground IoU | 72.14% | 71.23% |
| Stress bond IoU | 72.87% | 72.11% |
| Bond-on-text-only | 4.48% | 1.68% |

只有文字污染门、样本数、原生分辨率和 internal-test 未读门通过；四项 IoU 门失败，P1 严格
NO-GO。`result.json` SHA256 为
`eb6f2e4120cda0f508459cdffe821a0dd637d54ed080ade7caa0daeb12929898`。

## 判断与下一步

模型并未表现出“把字母当键”的旧失败：两个通道 precision 都超过 99.5%，文字区域误报为
bond 只有 1.68%。主要残差是两个通道同时低召回。训练目标保留 0--255 抗锯齿 soft alpha，
而固定评测把 GT `>8/255` 视为前景、预测却要求概率 `>=0.5`，两者的边缘定义可能不一致。

下一周期先做零训练 threshold PR 审计：只在 validation 的固定网格上分别选择 bond/text
阈值，锁定后一次性读取 internal test。若阈值扫描也不能过原 IoU 门，才判定模型容量或损失
不足；若 validation 通过但 internal test 失败，则判为校准过拟合。无论结果如何，本 P1 的
固定 0.5 结论保持 NO-GO。
