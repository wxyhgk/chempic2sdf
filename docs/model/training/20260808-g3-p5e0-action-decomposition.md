# G3 P5-E0 动作分解审计

> 日期：2026-08-08
>
> 状态：已完成，representation/offset NO-GO

## 目标

P4-E9B已经证明refine offset可以学到亚像素级附近，但独立candidate BCE无法安全决定是否执行。
本轮不训练，只在冻结checkpoint上把失败拆成三个互不替代的能力：

1. `image trigger`：整张图是否至少需要一个refine动作；
2. `pointer rank`：已知本图需要动作时，Oracle source在128个candidate中的稳定排名；
3. `forced offset`：强制选择Oracle source后，预测anchor是否在目标6px内。

## 固定协议

- checkpoint固定为P4-E9B `a564510f...cf500`；
- 只读scaffold-disjoint calibration 3k和holdout 3k，不重训、不读confirm_v2；
- image trigger score固定为每图最大refine logit；
- pointer报告top1/2/4/8/16，offset报告6px恢复率和误差；
- image trigger阈值只在calibration的无动作图上选为严格零假阳性的最高边界，再原样重放holdout；
- holdout pointer top4至少50%、强制offset 6px至少80%，才允许进入set-conditioned capacity；
- image trigger零假阳性非零召回只作断点诊断，不单独阻断set-conditioned建模。

若pointer和offset均过门，说明下一步应显式建模`NO_ACTION + action count + candidate pointer`；若任一
失败，则先扩大局部/结构证据，不能只换集合损失。

## 实际结果

| 域 | 动作图/总图 | 动作数 | trigger AUC | source top4 | source top16 | 强制offset 6px | offset MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| calibration | 28/3000 | 31 | 0.7496 | 4 | 11 | 9/31 | 8.71px |
| holdout | 61/3000 | 67 | 0.7892 | 10 | 23 | 18/67 | 8.85px |

三个能力门全部失败。calibration零假阳性阈值仍为`3.09375`，执行0张正图；固定到holdout后
执行3张且全是假阳性。正确source没有稳定进入前列，强制source也只有26.87% offset落在6px。
因此不能直接进入set-conditioned capacity。

代码复核发现当前局部head在三层卷积后使用`AdaptiveAvgPool2d(1)`，把32x32 crop的空间方向
压成单向量，却让MLP回归二维offset。fit的0.72px主要是小样本记忆，跨scaffold后空间方向不
泛化。下一步先训练保留空间的secondary-peak heatmap，只验证强制source的定位能力；定位过门
后才恢复`NO_ACTION + pointer`集合建模。

audit SHA256：
`a5e29323650e7ce32acd7b63928e7573ffbdc63a16885cacceb6b0b4e8613a0c`；耗时226.15秒。
