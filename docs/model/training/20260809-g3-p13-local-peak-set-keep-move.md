# G3 P13 Local Peak Set KEEP/MOVE

> 日期：2026-08-09
>
> 状态：已预注册，尚未训练

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
