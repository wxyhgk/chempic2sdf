# G3 P6-E0 Context Capacity

> 日期：2026-08-08
>
> 状态：已完成，GO；选择96px

## 目标

P5-E11证明32px局部方向头即使Oracle融合也无法通过MAE门。P6不再调局部专家，先验证64/96px
视野是否为失败动作带来新的目标邻接结构，而不是只增加无关背景。

## 固定协议

- 冻结P5-E6的31/67动作与成功状态，不训练；
- 对每个目标从structure labels取得incident bond和相邻原子；
- 以真实base source为中心，比较32/64/96px正方形crop内可见的目标邻接原子和对侧bond port；
- 对P5-E5失败动作，64或96px相对32px新增至少一个incident neighbor的比例必须在calibration和
  holdout都达到40%；
- 选择满足门槛的最小crop；不读dev、`confirm_v2`、外部或封存集。

过门才允许构建对应尺寸的image/center-field context cache并训练P6模型。本轮不创建tag。

## 实际结果

| crop | calibration失败新增邻居 | holdout失败新增邻居 | 结论 |
|---:|---:|---:|---|
| 64px | 0/7（0%） | 9/31（29.03%） | NO-GO |
| 96px | 3/7（42.86%） | 19/31（61.29%） | GO |

选择满足40%门的最小尺寸96px。全部动作可见邻居均值从32px的0.06/0.16增加到96px的
0.48/0.81；能看全目标所有incident neighbor的动作从0/3增加到5/17。64px仍不足以给
calibration失败增加邻接信息，不能作为折中尺寸。

下一P6模型必须用覆盖完整96px的结构上下文编码路径，同时保持中心`(4,13]px`目标环带的精细
输出；不能只把P5的32px三层卷积输入插值放大。

audit SHA256：`4bf4270aa5f8cabb2cff4c6e0352edd5af4aed1160dd1aaf7716ca76cde8fd3d`；
records SHA256：`6b5a9e64e771f94ea0c4c712dfe03630f6d6d9c3c5d9c700d1cdc485b57071c9`。本轮零训练、
不创建tag；dev、`confirm_v2`、外部和封存集均未读取。
