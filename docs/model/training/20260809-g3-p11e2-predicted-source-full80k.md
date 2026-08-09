# G3 P11-E2 Predicted Source Full80k

> 日期：2026-08-09
>
> 状态：已预注册，尚未运行

P11-E1在225个train MOVE上容量100%但跨scaffold完全不安全，证明pilot正动作不足。P11-E2固定
扫描已有full_train 80,000图，动态重建冻结P4-E3 top128与ADD_REFINE动作。仅对出现MOVE的困难
图片采样最多8个已匹配KEEP，为零偏移KEEP与P6 top8 MOVE生成真实crop cache。

按scaffold哈希80/10/10拆成train/calibration/holdout，holdout写完立即设为000。数据门要求至少
1,000 MOVE、800个top8可达MOVE、train至少6,000 KEEP、calibration和holdout各至少100 MOVE、
各split覆盖四variant且scaffold重叠0。该阶段不训练，不读取confirm、MR-TADF、USPTO或其他封存集。

门通过后才按P11-E1相同KEEP/MOVE协议训练规模曲线；失败则停止P6 top8真实source路线。
