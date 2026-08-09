# G3 P10 Pairwise Dev Shadow

> 日期：2026-08-09
>
> 状态：已完成，NO-GO

P7--P9一直用单个source的局部动作做绝对零harmful门，但最终目标是整张图的Atom集合。P9已消除
原文字锚点系统性错误，剩余1个cross-certificate harmful来自折间极值阈值波动。P10不部署、不
修改阈值，直接验证局部净收益能否传导到整图coverage exact。

使用已暴露但与P6/P7 sample及scaffold均零重叠的dev：640图、160分子，四variant各160。冻结
P4-E3 top128 source、P6-E1峰候选和P9 checkpoint。baseline对每个source选择P6第一峰；treatment
仅在P9 corrected margin达到固定全calibration阈值1.6389398574829102时改选alternative。

指标使用6px最大基数匹配，比较matched Atom、top128 coverage exact及paired exact rescue/break。
同时满足有执行动作、matched净增益>0、exact净增益>0且exact rescue>break才GO。该GO仅允许设计
新的独立confirm，不允许切换前端或读取现有confirm、MR-TADF、USPTO及封存集。

## 结果

640图共32,400个GT Atom：

| 路线 | matched | coverage exact |
|---|---:|---:|
| 原始P4-E3 top128 | 32,193 | 550/640 |
| 每个source应用P6第一峰 | 27,046 | 45/640 |
| P6第一峰加P9改选 | 27,010 | 44/640 |

P9在固定阈值执行4,581次，相对P6 prior的matched净变化为-36，exact rescue/break为0/1。所有整图
能力门失败，正式`pairwise_dev_shadow_no_go`，不进入confirm或部署。

该结果暴露了比安全阈值更上游的问题：P6--P9的训练source是从GT确定性平移6--12px得到，默认
source需要移动；真实P4-E3 top128 source则绝大多数已经在GT 6px内。将P6第一峰无条件作为prior
破坏了已正确source。下一协议必须把原source零偏移作为显式KEEP候选，在真实预测source分布上
训练`KEEP vs MOVE`，而不是继续调P9阈值或pair负例。
