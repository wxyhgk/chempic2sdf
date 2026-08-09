# G3 P17 Groupwise Linear KEEP/MOVE

> 日期：2026-08-09
>
> 状态：已完成，NO-GO

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

## 结果

- 模型8参数，固定4,000步耗时2.32秒；
- train组为191 rescue、1,584 KEEP、34 unsupported，协议完全一致；
- train零阈值组准确率95.19%，容量门通过；
- checkpoint：`outputs/checkpoints/chemvector-g3-p17-groupwise-linear-keep-move-train1809/step-004000.pt`；
- checkpoint SHA256：`84ddffe49afeba2fb10ed6ef9c8b833657d947ef33d7a1c20ed5a5fa6811e070`；
- calibration非负安全阈值仍只能0执行、0 rescue、0 harmful；
- 历史holdout保持000且未读取。

模型在train/calibration的最高动作logit都为负。即使事后仅作诊断地放宽到-1.5，train为
18 rescue/13 harmful，calibration为0 rescue/1 harmful；放宽到-2.0则分别为67/189和8/31。
因此不是阈值零点偏移。95.19%的train准确率主要来自KEEP占比高，不能表示恢复容量。

## 结论

P17正式NO-GO。按source组直接优化top8最大值仍不能从固定7维信息中获得跨scaffold安全恢复，
P6局部MOVE修正路线至此停止。P13--P17已经依次排除局部峰、完整top128上下文、冻结Bond端点、
低容量线性模型和groupwise max目标。下一路线必须联合训练Atom选择、Bond对象和incidence，使一个
Atom候选是否成立由整图关系共同决定，而不是继续给独立source打修补分数。
