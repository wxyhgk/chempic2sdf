# G3 P6-E11 OOF Residual Safety

> 日期：2026-08-09
>
> 状态：已完成，NO-GO

冻结P6-E9三臂，以global改选事件为对象，从三臂margin、entropy、residual、agreement及候选相对
坐标构造26维target-free特征。按scaffold哈希五折训练32→16 MLP并生成OOF分数，再在全部train
重训固定模型。calibration只对单一安全概率阈值做零harmful最大rescue校准。

calibration至少100 rescues且零harmful才GO；否则holdout继续000。本模型不读取GT字段作为特征，
不读取holdout/dev/confirm/外部或封存集。正式checkpoint无论GO/NO-GO都提交并打同名tag。

## 结果

- train改选事件6,804，正例5,984；calibration改选810，正例692；
- 五个scaffold OOF fold全部存在，最终模型1,409参数；
- 零harmful阈值`0.892985999584198`执行366次；
- rescue 69、harmful 0、useless 297，未达到100 rescue门；
- 决策为`oof_residual_safety_calibration_no_go`，holdout继续000。

相比固定三臂共识的46 rescues有所提高，但26维连续特征仍无法充分区分“改选无害”与“改选真正
恢复目标”。停止在同一特征上增加MLP参数或继续搜索calibration阈值；下一机制必须直接学习
source、prior候选与替代候选之间的成对结构反事实，而不是只分类汇总标量。

checkpoint SHA256：`5bac5eb39fc58f9a9b309425d3b1c9782e7d6ff13246d27f701bf1ec8ae6066b`；
audit SHA256：`c6865f0211aa88e354ec843e0790771394c47af27fbc71fddd096c5262496d50`。
