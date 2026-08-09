# G3 P11-E3 Predicted Source KEEP/MOVE Full

> 日期：2026-08-09
>
> 状态：已预注册，尚未训练

P11-E2 full80k数据门GO：1,495 MOVE、1,216可达；train/calibration/holdout MOVE为
1,188/149/158，KEEP为8,584/976/1,112，scaffold重叠0且holdout已封存000。

train可构造1,464 rescue pair；按role/variant/degree/距离严格匹配后，所有43个rescue层均有充足
harmful，固定使用1,464:1,464和2,048 equivalent训练8,000步。模型、损失和显式零偏移KEEP协议
与P11-E1不变，只改变数据规模。

calibration每个source最多执行一个MOVE，阈值不得低于0；至少40 rescue且0 harmful才GO。未过门
不得读取holdout；通过才允许使用冻结checkpoint和阈值单次读取P11-E2 holdout。confirm和外部集
仍禁止读取。正式模型无论结果均保存checkpoint并创建同名tag。
