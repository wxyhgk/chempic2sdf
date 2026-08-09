# G3 P9 Stratified Pairwise Anchor

> 日期：2026-08-09
>
> 状态：已完成，NO-GO

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

## 结果

- 训练pool严格为4,960 rescue、4,960 harmful、2,048 equivalent；
- 已知TEXT/stress/degree3plus/8--16px层从83:22修正为83:83；
- checkpoint SHA256：`8bc96e42c2229c640967baa1cf4e12ed577230fae8a0465bb815e1c799423eae`；
- 全calibration：136执行、108 rescue、0 harmful；
- 五折certificate：156执行、124 rescue、1 harmful；
- 原fold 1文字锚点harmful已消失，证明分层修复命中该机制；
- 新harmful出现在fold 4，为coordgen/IMPLICIT/degree2，分数1.637464；该折用其余四折选择的阈值
  为1.412567，而全calibration安全阈值为1.638940，相差仅0.001476。

严格交叉证书门仍失败，决定为`stratified_pairwise_anchor_certificate_no_go`，历史holdout未读。
结果表明剩余问题已从文字表示偏差转为极值阈值的小样本波动。下一步不能继续对同一calibration
搜索安全阈值；应冻结P9全calibration阈值，只在已暴露dev做整图shadow，比较最终Atom exact的
paired rescue/break。shadow不等于部署，也不能读取confirm或外部封存集。
