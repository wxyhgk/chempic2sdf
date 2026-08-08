# G3 P5-E7 Wrong Destination

> 日期：2026-08-08
>
> 状态：已预注册，等待审计

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
