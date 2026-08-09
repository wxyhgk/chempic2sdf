# G3 P11-E1 Predicted Source KEEP/MOVE

> 日期：2026-08-09
>
> 状态：已预注册，尚未训练

P11-E0数据门通过：真实MOVE 224/264可达，真实KEEP 1,888个并提供4,930个harmful alternative。
按scaffold固定切分后，train为225 MOVE/1,584 KEEP，calibration为39/304。

训练pair以零偏移source为prior、P6 top8之一为alternative，不使用peak margin。train有285 rescue；
按role/variant/degree/距离严格分层后得到281个唯一rescue和281个唯一harmful，另取512 equivalent。
先在64 directed pair上训练1,500步并要求99%准确率，再以相同模型训练6,000步。

calibration每个真实source最多执行一个最高utility MOVE，阈值禁止低于0。至少20 rescue且0 harmful
才GO；否则停止。GO也只允许构建旧P4 calibration动作的独立重放，不读取P7 holdout、confirm或
外部集。正式模型无论结果如何保存checkpoint并创建同名tag。
