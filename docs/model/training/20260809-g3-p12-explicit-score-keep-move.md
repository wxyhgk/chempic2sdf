# G3 P12 Explicit Score KEEP/MOVE

> 日期：2026-08-09
>
> 状态：已预注册，尚未训练

P11-E3将真实MOVE扩大5.28倍后仍无安全动作，否定单纯数据不足。calibration审计显示center-field
精确点差`alternative-source`对rescue/harmful的pair AUC为0.8113；当前CNN的多层卷积与全局池化
没有可靠保留该标量。

P12保持P11-E3的1,464:1,464+2,048数据、patch-only编码器、损失、8,000步和随机隔离，只在每个
candidate特征中显式追加candidate center-field值及其相对source差值。source、alternative和差值
均由已有score crop双线性采样，不重扫数据。

calibration至少40 rescue/0 harmful才GO；否则停止。未过门不读取P11-E2 holdout、confirm或外部集。
正式checkpoint无论结果均提交并创建同名tag。
