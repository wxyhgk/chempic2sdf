# v1 P1 语义墨迹分离 Pilot

> 日期：2026-08-14
>
> 状态：已预注册，等待训练
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
