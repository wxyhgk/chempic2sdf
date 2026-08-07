# G3 P4-E3 全图峰排序损失审计

> 日期：2026-08-07
>
> 状态：已完成，机制与成本审计通过
>
> 范围：零训练、零数据集读取

## 问题

P4-E2 的 `atom_center_peak_margin_loss` 只要求GT中心高于自身3×3邻域内的非GT像素。它没有
比较整图假峰，也没有直接优化真实峰能否进入top128。因此P4-E2训练20,000张唯一图后，
known-Count多匹配71个Atom，但top128只多匹配4个Atom、exact只增加1张。

本轮先验证一个训练目标，不启动模型训练：若一张图有 `G` 个GT中心且候选预算为 `K=128`，
则每个GT中心至少要超过第 `K-G+1` 个假regional maximum，才能让全部GT进入top-K。

## 损失协议

- 用detach后的整图logit通过3×3局部最大条件选择训练期regional maxima位置；
- GT中心从假峰集合中排除，相邻GT永远不会互相成为负样本；
- 从假峰排序的候选边界向上取最多16个峰，避免只依赖一个不稳定次序统计量；
- 选中位置只detach索引，正峰与假峰logit都从原张量gather并保留梯度；
- margin固定为1；当假峰数量加GT数量不超过128时，该图返回可微零损失；
- 不引入固定像素聚类、键长删除、NMS半径或推理后处理规则。

等值plateau在训练期可贡献多个局部最大位置。这是有意的保守近似：训练会压低整个假plateau，
而不是为训练目标新增一条与推理绑定的像素聚类协议。正式评估仍使用已有完整plateau regional
maxima生成器。

## 审计结果

纯函数测试覆盖以下条件并全部通过：

- GT低于top-K边界时loss非零；
- GT logit获得向上的梯度，选中假峰logit获得向下的梯度；
- 未选中的更高假峰不被错误更新；
- margin满足后loss接近0；
- 相邻GT不互相惩罚；
- 所有候选都能容纳时返回可反向传播的零损失。

5090上的生产形状审计使用batch4、768×768、BF16：

| 项目 | 结果 |
|---|---:|
| 单次损失前向+反向 | 1.42 ms |
| CUDA峰值分配 | 103,809,536 bytes（约99 MiB） |
| 正峰梯度 | 全部非零且方向正确 |
| 假峰梯度 | 存在且方向正确 |
| 梯度有限值 | 通过 |

审计文件SHA256：
`41e5ebc942623a637bd29eee4fac44ccf3c999a4c4abfafa1f3f213632e37eb2`。

## 决策

机制和成本均GO。下一步预注册实际模型
`chemvector-g3-p4e3-global-peak-rank-pilot20k`，从与P4-E2相同的train25k checkpoint初始化，
使用相同pilot20k、步数、optimizer和可训练参数，只增加全图峰排序损失，以形成受控A/B。
confirm_v2、fresh、USPTO、MR-TADF和sealed_test继续不得读取。
