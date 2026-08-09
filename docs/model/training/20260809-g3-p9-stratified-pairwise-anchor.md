# G3 P9 Stratified Pairwise Anchor

> 日期：2026-08-09
>
> 状态：已预注册，尚未训练

P8三成员仍在calibration交叉证书产生1个harmful。合法诊断将其定位为
`CHEMBL5808128 / stress / TEXT / degree3plus`：prior位于文字左侧真实连接锚点，alternative位于
同一字形右侧约12px，所有四个已训练模型都系统性选择alternative。

审计发现P7实现没有兑现预注册的分层harmful匹配，而是全局哈希截取。该失败层在train中实际为
83 rescue对22 harmful，模型因而学会在这类文字上偏向移动。这是训练协议错误，不是模型容量或
ensemble方差。

P9只修正pair选择，不改变模型、损失、步数或calibration协议：

- 排除prior/alternative间距小于8px、缺少对称负例的pair；
- 按`role × variant × degree × {8--16px, >=16px}`严格分层；
- 每层唯一rescue与harmful按较小数量1:1下采样；
- 得到4,960 rescue、4,960 harmful、2,048 equivalent；
- 已知失败层固定为83:83；
- patch-only新seed训练8,000步。

全calibration必须至少80 rescue/0 harmful，五折交叉证书合计至少60 rescue/0 harmful。失败即
NO-GO；通过才允许设计新的独立确认集。历史P7 holdout保持000且不读取。
