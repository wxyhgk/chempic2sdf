# G3 P11-E3 Predicted Source KEEP/MOVE Full

> 日期：2026-08-09
>
> 状态：已完成，NO-GO

P11-E2 full80k数据门GO：1,495 MOVE、1,216可达；train/calibration/holdout MOVE为
1,188/149/158，KEEP为8,584/976/1,112，scaffold重叠0且holdout已封存000。

train可构造1,464 rescue pair；按role/variant/degree/距离严格匹配后，所有43个rescue层均有充足
harmful，固定使用1,464:1,464和2,048 equivalent训练8,000步。模型、损失和显式零偏移KEEP协议
与P11-E1不变，只改变数据规模。

calibration每个source最多执行一个MOVE，阈值不得低于0；至少40 rescue且0 harmful才GO。未过门
不得读取holdout；通过才允许使用冻结checkpoint和阈值单次读取P11-E2 holdout。confirm和外部集
仍禁止读取。正式模型无论结果均保存checkpoint并创建同名tag。

## 结果

- checkpoint SHA256：`c58e6ba4603ccecc6dc5e1ac1977b3f37f2170191eb0d33c2aa3a1171fbe64f2`；
- calibration 149 MOVE/976 KEEP；
- 零harmful阈值仍只能0执行，正式NO-GO；
- 阈值0为31 rescue/142 harmful；最高分区约2.0为2 rescue/5 harmful。

MOVE扩大5.28倍后安全比率没有实质改善，否定“仅数据不足”。审计center-field标量发现
`alternative_score-source_score`区分rescue/harmful的pair AUC为0.8113；现有CNN经多层卷积和全局
池化未保留该精确点值。下一步固定把source、alternative及差值作为显式特征接入utility，不重扫
数据、不读holdout，也不继续增加当前模型step。
