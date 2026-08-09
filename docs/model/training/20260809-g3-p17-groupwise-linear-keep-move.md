# G3 P17 Groupwise Linear KEEP/MOVE

> 日期：2026-08-09
>
> 状态：已预注册，尚未训练

P16证明8参数线性模型也无法把0.847 pair OOF AUC转化为安全动作。固定阈值0在calibration为
9 rescue/37 harmful；部署对每个source取top8最大值，会放大独立pair训练中的任一假阳性。

P17保持P16完全相同的7维输入和8参数线性模型，只把训练单位改成完整source组。train固定为
191 rescue、1,584 KEEP、34 unsupported组；每组全部有效top8同时进入损失：

- rescue：至少一个正确MOVE超过margin，所有错误MOVE低于负margin，且最佳正确项超过最佳错误项；
- KEEP：所有MOVE的最大分数低于负margin；
- unsupported：所有无效MOVE的最大分数低于负margin。

固定margin1、L2 0.01、AdamW lr 0.01、4,000步。每批24 rescue、32 KEEP、8 unsupported，固定
seed；不增加特征、参数、Bond或top128上下文。train零阈值组准确率至少75%才满足容量门。

最终仍按部署协议在calibration每个source取top8最高分，非负阈值中至少20 rescue且0 harmful才GO。
历史holdout、confirm和外部集不得读取。正式checkpoint无论结果均提交并创建同名annotated tag。
