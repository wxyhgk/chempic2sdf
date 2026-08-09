# G3 P11-E1 Predicted Source KEEP/MOVE

> 日期：2026-08-09
>
> 状态：已完成，NO-GO

P11-E0数据门通过：真实MOVE 224/264可达，真实KEEP 1,888个并提供4,930个harmful alternative。
按scaffold固定切分后，train为225 MOVE/1,584 KEEP，calibration为39/304。

训练pair以零偏移source为prior、P6 top8之一为alternative，不使用peak margin。train有285 rescue；
按role/variant/degree/距离严格分层后得到281个唯一rescue和281个唯一harmful，另取512 equivalent。
先在64 directed pair上训练1,500步并要求99%准确率，再以相同模型训练6,000步。

calibration每个真实source最多执行一个最高utility MOVE，阈值禁止低于0。至少20 rescue且0 harmful
才GO；否则停止。GO也只允许构建旧P4 calibration动作的独立重放，不读取P7 holdout、confirm或
外部集。正式模型无论结果如何保存checkpoint并创建同名tag。

## 结果

- 64-pair容量：64/64，100%；
- 正式训练：281 rescue、281 harmful、512 equivalent，6,000步；
- checkpoint SHA256：`f5f5349654575e1f15bc30bdf82013e58bf2d5f4c9b3809aa55241cdad369d79`；
- calibration source：39 MOVE、304 KEEP；
- 唯一零harmful阈值为不执行，0 rescue/0 harmful，能力门失败。

风险曲线显示不是阈值过严：最高分约2.0的区间已经有12个KEEP harmful，却仅1个MOVE rescue；
阈值0时为10 rescue/54 harmful。模型在64-pair可完全记忆，但225个train MOVE不足以学习跨scaffold
的KEEP/MOVE边界，输出大量饱和到residual cap。

正式决定为`predicted_source_keep_move_calibration_no_go`，不读取旧P4 calibration/holdout、confirm
或外部集。P11-E0已证明候选可达，E1则证明pilot正动作规模不足。下一步应固定协议扫描full80k，
把真实MOVE扩大到预计1,000--1,500；不能在当前562 pair上追加step、调cap或降低安全阈值。
