# G3 P5-E7 Wrong Destination

> 日期：2026-08-08
>
> 状态：已完成，NO-GO

## 目标

P5-E6证明主要失败形态是方向选错，且局部2个GT与degree=2明显更难。本轮只判断错误预测是否
落到或更接近同图另一GT，从而区分“对象竞争”与“方向证据完全丢失”。

## 固定分类

按优先级将每个预测分为：

1. `TARGET_HIT`：距目标不超过6px；
2. `OTHER_GT_HIT`：失败且距另一GT不超过6px；
3. `OTHER_GT_CLOSER`：失败、未命中另一GT，但另一GT比目标更近；
4. `TARGET_STILL_NEAREST`：失败但目标仍是最近GT。

calibration和holdout失败中，`OTHER_GT_HIT + OTHER_GT_CLOSER`占比都至少40%才GO，支持下一步
显式集合条件化；否则不采用该解释。冻结P5-E5和同一98动作，不训练、不读dev、confirm或外部集。

## 实际结果

| split | 失败 | 命中另一GT | 另一GT更近 | 目标仍最近 | 另一GT更近占失败 |
|---|---:|---:|---:|---:|---:|
| calibration | 7 | 0 | 1 | 6 | 14.29% |
| holdout | 31 | 2 | 11 | 18 | 41.94% |

calibration未达到40%门，最终`wrong_destination_competition_no_go`。合计38个失败中只有2个真正
落入另一GT 6px内，另有12个更接近另一GT；24个错误预测仍以目标为最近GT。显式集合条件化
可能改善holdout的一部分对象竞争，但不能解释两个域的主要失败，也不能作为下一主路线。

结合P5-E6的degree=2和局部多GT缺口，下一假设改为“模型没有从足够广泛的一般原子局部几何中
学到source到真实anchor的方向场”。应先用训练域全部普通原子生成通用source→anchor监督，再在
原98个真实残差上冻结评测；不继续只收集稀有Oracle失败动作。

audit SHA256：`24ffc18f24cfd030e652059b2c407301ebe64984beda02d7d037fc70c1cea374`；
records SHA256：`21a626058ba36b8122b28ab51833d6ecedc3eca68394bfd3ddf6177fc109d21a`。没有训练，不创建tag，
dev、`confirm_v2`、外部和封存集均未读取。
