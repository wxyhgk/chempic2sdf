# G3 P10 Pairwise Dev Shadow

> 日期：2026-08-09
>
> 状态：已预注册，尚未运行

P7--P9一直用单个source的局部动作做绝对零harmful门，但最终目标是整张图的Atom集合。P9已消除
原文字锚点系统性错误，剩余1个cross-certificate harmful来自折间极值阈值波动。P10不部署、不
修改阈值，直接验证局部净收益能否传导到整图coverage exact。

使用已暴露但与P6/P7 sample及scaffold均零重叠的dev：640图、160分子，四variant各160。冻结
P4-E3 top128 source、P6-E1峰候选和P9 checkpoint。baseline对每个source选择P6第一峰；treatment
仅在P9 corrected margin达到固定全calibration阈值1.6389398574829102时改选alternative。

指标使用6px最大基数匹配，比较matched Atom、top128 coverage exact及paired exact rescue/break。
同时满足有执行动作、matched净增益>0、exact净增益>0且exact rescue>break才GO。该GO仅允许设计
新的独立confirm，不允许切换前端或读取现有confirm、MR-TADF、USPTO及封存集。
