# G3 P6-E11 OOF Residual Safety

> 日期：2026-08-09
>
> 状态：已预注册，等待训练

冻结P6-E9三臂，以global改选事件为对象，从三臂margin、entropy、residual、agreement及候选相对
坐标构造26维target-free特征。按scaffold哈希五折训练32→16 MLP并生成OOF分数，再在全部train
重训固定模型。calibration只对单一安全概率阈值做零harmful最大rescue校准。

calibration至少100 rescues且零harmful才GO；否则holdout继续000。本模型不读取GT字段作为特征，
不读取holdout/dev/confirm/外部或封存集。正式checkpoint无论GO/NO-GO都提交并打同名tag。
