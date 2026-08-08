# G3 P5-E2 Contextual Secondary Peak

> 日期：2026-08-08
>
> 状态：已完成，NO-GO

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

## 实际结果

模型共45,353个参数，训练3,000步、累计暴露192,000个D4动作样本。loss从P5-E1的近均匀状态
明显下降，最终为3.6042；完整上下文路径已经解除训练表示阻塞。

| 域 | 动作数 | 6px内 | 6px恢复率 | MAE | 中位误差 | P90误差 |
|---|---:|---:|---:|---:|---:|---:|
| fit | 264 | 264 | 100.00% | 0.54px | 0.51px | 0.95px |
| calibration | 31 | 14 | 45.16% | 7.61px | 6.22px | 16.22px |
| holdout | 67 | 22 | 32.84% | 9.10px | 10.23px | 15.01px |

四项正式门全部失败，最终决策为`contextual_secondary_peak_no_go`。与P5-E1不同，本模型已经
把fit动作全部学到亚像素量级，但跨scaffold反而下降；继续增加同一小集合上的step或参数只会
强化记忆，不会解决泛化。

checkpoint：
`outputs/checkpoints/chemvector-g3-p5e2-contextual-secondary-peak-fit264/step-003000.pt`，SHA256
`72b14c82c12c36a4664d8c29f71efcf5fc37173e3f1b2beab66217f41a61f631`。audit SHA256：
`3d0adf6c7f49ff5e28bc3c0d093114fafd4050a8c2a31ada0112cd6ac9987f79`。`confirm_v2`、dev、外部集
和封存集均未读取。

## 结论

P5-E2把问题收敛为“264个稀有残差动作的数据覆盖不足”与“局部crop对正确偏移不存在可迁移
判别规则”二选一。下一步先做零训练容量审计：只在未用于评测的full80k训练域统计可生成的
独立refine动作、scaffold和失败类型。若规模仍不足或分布单一，则停止该路线；只有动作容量和
覆盖显著扩大时，才允许预注册一次固定规模曲线，不读取confirm或封存集。
