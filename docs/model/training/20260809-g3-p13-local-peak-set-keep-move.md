# G3 P13 Local Peak Set KEEP/MOVE

> 日期：2026-08-09
>
> 状态：已完成，NO-GO

P12证明显式加入source与candidate的center-field点值仍不足以获得安全动作。P13检验一个更具体的
机制假设：KEEP/MOVE不能只由一个source和一个alternative的局部patch决定，还需要知道它们在
局部峰集合中的位置、排名和竞争关系。

## 固定协议

- 模型名：`chemvector-g3-p13-local-peak-set-keep-move-full2928`；
- 数据、1,464 rescue、1,464 harmful、2,048 equivalent、损失、8,000步均与P12相同；
- 保留P12的显式source/candidate center-field点值；
- 从已有96×96 score crop以5×5 regional maximum提取固定top-16峰，不重扫full80k；
- 每个峰token包含相对source坐标、相对candidate坐标、峰值、ink、两侧距离和归一化排名；
- token共享MLP后做masked mean/max DeepSets聚合，集合输入顺序不影响结果；
- candidate utility仍以正反两次有序计算之差输出，交换KEEP/MOVE时严格反对称；
- 参数上限保持300k。

训练只读取train与calibration cache。calibration达到至少40 rescue且0 harmful才允许读取P11-E2
holdout；否则直接NO-GO。confirm、MR-TADF与USPTO均不参与训练、调参或阈值选择。正式checkpoint
无论结果均记录SHA256并创建与模型名相同的annotated tag。

## 预注册判断

本轮只改变局部峰集合上下文。若仍只有零执行安全阈值，则停止继续扩展单source局部
KEEP/MOVE修正器；下一路线应回到完整top128 Atom集合上联合预测占位和基数，而不是继续增加
patch、局部标量、训练步数或同分布pair数量。

## 结果

- 模型参数：49,101；
- 训练：1,464 rescue、1,464 harmful、2,048 equivalent，8,000步，44.50秒；
- checkpoint：`outputs/checkpoints/chemvector-g3-p13-local-peak-set-keep-move-full2928/step-008000.pt`；
- checkpoint SHA256：`af87e139ee676c41ad386b9f8b155b36b786b92c13c05d86deb73ba1c4a0cea5`；
- calibration安全阈值仍为正无穷，只能0执行、0 rescue、0 harmful；
- P11-E2 holdout及历史holdout保持000且未读取。

固定阈值诊断也否定“只差阈值”的解释：阈值0执行716项，为49 rescue/161 harmful；阈值1.5
仍执行262项，为41 rescue/92 harmful。局部峰集合让模型拟合损失下降，但不能判断source在完整Atom
集合中是否已正确占位，高分区仍由有害MOVE主导。

## 结论

P13正式NO-GO。停止继续增加单pair patch、局部峰、显式标量、相同数据量或训练步数。下一周期
必须直接输入完整top128候选集合，在统一的一对一占位和基数约束下联合决定KEEP/MOVE；P13不进入
holdout、整图部署、MR-TADF或USPTO benchmark。
