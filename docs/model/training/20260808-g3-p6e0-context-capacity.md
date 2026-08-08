# G3 P6-E0 Context Capacity

> 日期：2026-08-08
>
> 状态：已预注册，等待审计

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
