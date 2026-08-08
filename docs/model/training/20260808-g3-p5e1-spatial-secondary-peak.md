# G3 P5-E1 Spatial Secondary Peak

> 日期：2026-08-08
>
> 状态：已完成，NO-GO

## 目标

P5-E0证明P4-E9B即使强制选择正确source，holdout offset也只有18/67落入6px。当前head在卷积后
使用全局空间池化，再用MLP回归方向；本轮只替换这一表示，不训练NO_ACTION或pointer。

## 固定模型

- 输入为32x32灰度crop、32x32 center-score crop和score/rank；
- 卷积始终保持32x32空间分辨率，score/rank只通过FiLM调制；
- 输出32x32 secondary-peak logits，不再直接回归二维向量；
- 有效支持为距source 4--13px的离散环带，监督目标仍严格来自Oracle `(6,12]px`；
- 推理取有效环带最大logit位置，量化误差上界小于1px；
- 参数少于5万。

## 训练与门

- 数据固定为P4-E9B fit-cache中的264个refine正动作，不加入负例；
- 每步对crop和offset同步应用8种正方形旋转/镜像之一，不加模糊、线宽或风格增强；
- 32D P4-E3 feature没有已知旋转等变协议，本轮不输入定位头，避免D4增强产生错误通道监督；
- batch64、3000步、AdamW、学习率`3e-4`、weight decay `1e-4`、clip 1；
- calibration/holdout只读取各自Oracle动作图，并强制使用正确source；
- 两域6px恢复率均至少80%、MAE均不超过3px才GO；
- 不读dev、confirm_v2、外部集或封存集。

模型名固定为`chemvector-g3-p5e1-spatial-secondary-peak-fit264`。无论GO/NO-GO，实际训练后提交
结果并创建同名annotated tag。GO后才恢复set-conditioned NO_ACTION/pointer；NO-GO则停止。

## 实际结果

模型共16,597个参数，训练3,000步、累计暴露192,000个动作样本。训练loss均值为5.5648，最终
loss为5.4305。给定正确Oracle source后得到：

| 域 | 动作数 | 6px内 | 6px恢复率 | MAE | 中位误差 | P90误差 |
|---|---:|---:|---:|---:|---:|---:|
| fit | 264 | 144 | 54.55% | 7.31px | 5.28px | 16.40px |
| calibration | 31 | 17 | 54.84% | 7.77px | 4.42px | 17.27px |
| holdout | 67 | 36 | 53.73% | 7.12px | 5.56px | 15.27px |

两域恢复率都未达到80%，MAE也都高于3px，四项门全部失败，最终决策为
`spatial_secondary_peak_no_go`。fit与两个跨scaffold域都稳定在约54%，没有出现只在训练集
拟合良好而跨域崩溃的现象；这个结果直接否定P5-E1 head，但后续复核发现其每个输出位置的定向
卷积感受野只有7x7，不能据此否定完整32x32局部证据。它不允许直接叠加
`NO_ACTION + count + pointer`或扩到full80k；下一轮只修复该架构混杂因素。

checkpoint：
`outputs/checkpoints/chemvector-g3-p5e1-spatial-secondary-peak-fit264/step-003000.pt`，SHA256
`aa790e5868e31c6e17920e45edfb616a98401441de8acbae5108f7ddd31f6fa2`。audit SHA256：
`7427276b6334182b657dfd115d308a99e90195cfef5476ddfc98651d22a7a3fd`。`confirm_v2`、dev、外部集
和封存集均未读取。

## 后续边界

只读复核已确认center-field环带argmax在fit仅23.11%命中6px，目标cell分数中位排名约100；
P5-E1的D4全一致命中率为33.33%，简单TTA没有提高原始54.55%。下一步P5-E2保持数据、目标和门
完全不变，只加入相对坐标与完整crop定向上下文；不追加同构卷积层、训练步数或样本量。
