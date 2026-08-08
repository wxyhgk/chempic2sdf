# G3 P6-E1 Structural Context96 Mixed

> 日期：2026-08-08
>
> 状态：已完成，NO-GO

## 模型

模型名`chemvector-g3-p6e1-structural-context96-mixed50k-fit1378`，共150,365参数：

- 96px灰度、center-field和相对坐标经五次stride-2编码到3x3，bottleneck卷积感受野覆盖完整96px；
- 对称解码回96px并截取中心32px结构feature；
- 中心32px另有精细局部分支，二者融合后仍输出32x32目标heatmap；
- 中心采样与P5逐像素一致，batch64 BF16峰值约414MiB、稳定训练步约6.4ms。

## 固定训练

- 构建50k generic与1,378真实动作的96px image/center-field cache；所有真实source边距至少55px；
- 50k generic训练10,000步，batch64、AdamW `3e-4`；
- 再以32 generic+32真实动作混合微调5,000步，学习率`1e-4`；
- D4、weight decay `1e-4`、clip 1和目标环带保持不变；
- 只保存最终checkpoint，两域恢复率均至少80%、MAE均不超过3px才GO；
- 不读dev、`confirm_v2`、外部或封存集。

无论GO/NO-GO，提交结果并创建与模型名一致的annotated tag。

## 训练结果

固定训练于2026-08-08完成。通用预训练10,000步后，再完成5,000步1:1混合微调；
最终checkpoint为：

```text
outputs/checkpoints/chemvector-g3-p6e1-structural-context96-mixed50k-fit1378/step-015000.pt
SHA256 2af36baf611b9a4c899f807742345e48bf905a30705901ca69c29efac9ca987d
```

最终结果：

| 数据域 | 动作数 | 6px恢复率 | MAE | p90 |
| --- | ---: | ---: | ---: | ---: |
| generic fit | 50,000 | 99.592% | 0.68px | 1.07px |
| real fit | 1,378 | 90.058% | 2.67px | 5.97px |
| calibration | 31 | 58.065% | 6.48px | 14.93px |
| holdout | 67 | 56.716% | 5.78px | 14.93px |

四个能力门全部失败。混合微调相对通用预训练把calibration从35.48%提高到58.06%、
holdout从28.36%提高到56.72%，证明真实残差监督有用；但训练内真实动作达到90.06%而两个
冻结域都停在约57%，说明当前逐点heatmap目标仍主要拟合训练动作的局部外观，96px上下文容量
本身没有转化成可泛化的结构条件。

## 决策

正式判定`structural_context96_no_go`，不接部署、不读取`confirm_v2`、dev、外部或封存集。
停止在相同目标上追加step、调学习率或简单扩大真实动作混合比例。P6下一项实验应改变监督单位：
以source、候选目标和邻接键共同构成结构化选择任务，直接比较多个合法目标，而不是继续独立回归
单个32x32热图；96px编码器可作为冻结或热启动的视觉底座。

## 产物审计

- 参数量：150,365；最终state SHA256：
  `9d1f683702e43cfbf5cd71028637346751b6e3d7400f70d59772b0cb9b10c855`；
- audit：`outputs/checkpoints/chemvector-g3-p6e1-structural-context96-mixed50k-fit1378/audit.json`，
  SHA256 `a70a5bf3b7920736b14c7d67dc1e25bce5bf44491aef4c7fe3395e63acc2ca3d`；
- generic cache SHA256：
  `80ffa1919e88605099fa2714571f0b51ef8b9803353ec5c4c002679605efbfba`；
- real cache SHA256：
  `d29c842a9e1be2c4d32ea4406014cde8c7f95ce4a6b989e38f2950fb155c69f9`；
- `confirm_v2_read=false`，`dev_and_external_read=false`。
