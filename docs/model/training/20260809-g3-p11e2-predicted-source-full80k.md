# G3 P11-E2 Predicted Source Full80k

> 日期：2026-08-09
>
> 状态：已完成，GO

P11-E1在225个train MOVE上容量100%但跨scaffold完全不安全，证明pilot正动作不足。P11-E2固定
扫描已有full_train 80,000图，动态重建冻结P4-E3 top128与ADD_REFINE动作。仅对出现MOVE的困难
图片采样最多8个已匹配KEEP，为零偏移KEEP与P6 top8 MOVE生成真实crop cache。

按scaffold哈希80/10/10拆成train/calibration/holdout，holdout写完立即设为000。数据门要求至少
1,000 MOVE、800个top8可达MOVE、train至少6,000 KEEP、calibration和holdout各至少100 MOVE、
各split覆盖四variant且scaffold重叠0。该阶段不训练，不读取confirm、MR-TADF、USPTO或其他封存集。

门通过后才按P11-E1相同KEEP/MOVE协议训练规模曲线；失败则停止P6 top8真实source路线。

## 结果

- 80,000图完整扫描，耗时2,623秒；
- MOVE 1,495，其中1,216可达，reachability 81.34%；
- train/calibration/holdout MOVE：1,188/149/158；
- train/calibration/holdout KEEP：8,584/976/1,112；
- train/calibration/holdout scaffold：946/110/120，重叠0；
- 三个split均覆盖四variant，所有数据门通过；
- holdout cache SHA256：`e7d7d768214e07db40b8cbdaa6c53e26be99d007a5dd2cf1571d554ce5f60821`，权限000。

train严格分层后可得到1,464 rescue和1,464 harmful且无短缺层，允许进入P11-E3固定8,000步训练。
