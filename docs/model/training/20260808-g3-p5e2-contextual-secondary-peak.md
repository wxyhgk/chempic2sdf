# G3 P5-E2 Contextual Secondary Peak

> 日期：2026-08-08
>
> 状态：已预注册，等待训练

## P5-E1 复核

P5-E1的输出虽然保持32x32，但每个输出位置只有三层3x3卷积提供定向空间证据，卷积感受野为
7x7。GroupNorm会聚合全图均值和方差，但不会把距目标6--12px的source方位传到目标位置；模型
也没有相对坐标通道。因此P5-E1的NO-GO不能区分“局部证据不足”和“头看不到完整上下文”。

训练域只读探针同时发现：center-field环带argmax的6px命中率只有23.11%，目标cell分数中位
排名约100；这不是一个可以直接阈值化的现成第二峰。P5-E1八种D4前向的全一致命中率仅33.33%，
Oracle-best为74.24%，简单TTA聚合仍只有54.92%。旋转不一致存在，但不是单独解法。

## 唯一变量

模型名固定为`chemvector-g3-p5e2-contextual-secondary-peak-fit264`。数据、目标、loss、D4增强、
训练步数和评测门与P5-E1完全相同，只替换定位头：

- 输入增加以source为原点的x/y相对坐标；
- 三次stride-2编码后在4x4 bottleneck卷积，使定向卷积路径覆盖完整32x32 crop；
- 逐层上采样并融合skip feature，输出同样的32x32 heatmap；
- 参数少于10万，不加入Bond、OCR、32D候选feature或新数据。

## 固定门

fit仍只用264个正动作，batch64、3,000步、AdamW `3e-4`、weight decay `1e-4`、clip 1。
calibration和holdout继续强制正确Oracle source；两域6px恢复率都至少80%且MAE都不超过3px才
GO。任一失败即NO-GO，不读dev、confirm_v2、外部集或封存集，不进入集合动作头。

每个实际训练结果都提交，并创建与模型名完全一致的annotated tag。
