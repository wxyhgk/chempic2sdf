# G3 P5-E0 动作分解审计

> 日期：2026-08-08
>
> 状态：已预注册，执行中

## 目标

P4-E9B已经证明refine offset可以学到亚像素级附近，但独立candidate BCE无法安全决定是否执行。
本轮不训练，只在冻结checkpoint上把失败拆成三个互不替代的能力：

1. `image trigger`：整张图是否至少需要一个refine动作；
2. `pointer rank`：已知本图需要动作时，Oracle source在128个candidate中的稳定排名；
3. `forced offset`：强制选择Oracle source后，预测anchor是否在目标6px内。

## 固定协议

- checkpoint固定为P4-E9B `a564510f...cf500`；
- 只读scaffold-disjoint calibration 3k和holdout 3k，不重训、不读confirm_v2；
- image trigger score固定为每图最大refine logit；
- pointer报告top1/2/4/8/16，offset报告6px恢复率和误差；
- image trigger阈值只在calibration的无动作图上选为严格零假阳性的最高边界，再原样重放holdout；
- holdout pointer top4至少50%、强制offset 6px至少80%，才允许进入set-conditioned capacity；
- image trigger零假阳性非零召回只作断点诊断，不单独阻断set-conditioned建模。

若pointer和offset均过门，说明下一步应显式建模`NO_ACTION + action count + candidate pointer`；若任一
失败，则先扩大局部/结构证据，不能只换集合损失。
