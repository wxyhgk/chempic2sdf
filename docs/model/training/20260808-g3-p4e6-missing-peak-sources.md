# G3 P4-E6 Missing Peak Sources

> 日期：2026-08-08
>
> 状态：已完成，混合来源
>
> 审计：`chemvector-g3-p4e6-missing-peak-sources`

## 目标

P4-E5在dev top1024仍缺58个Atom，768到1024完全无增长。本轮固定同一checkpoint和dev640，
逐个定位58个最大基数匹配下的未匹配GT，确定下一次修改应该针对局部峰生成、深尾排序还是
一对一匹配冲突。

## 固定协议

- 使用与生产指标完全相同的最大基数优先、距离次优Hungarian匹配；
- top1024必须严格重放32,342 matched、58 missing、43张失败图；
- 每张图同时提取全部完整plateau regional maxima，上限200,000，实际不得触顶；
- 每个漏点记录top1024与全部峰的最近距离/排名、GT cell logit与regional-max状态；
- 记录最近GT距离、是否相邻GT cell、3×3局部竞争者是否为另一GT；
- 独占分类顺序为matching conflict、below top1024、GT cell不是峰、峰anchor偏移超6px；
- 为所有失败图生成原图叠图和HTML索引，红圈为漏GT、橙圈为最近完整峰；
- 不训练、不修改checkpoint、不读pilot、confirm_v2、fresh或外部/封存集。

若某一类别占58个漏点至少一半，则报告为primary；否则报告mixed。该结论只决定下一实验方向，
不构成模型GO，不创建tag。

## 预期产物

- `outputs/audits/chemvector-g3-p4e6-missing-peak-sources/audit.json`
- `outputs/audits/chemvector-g3-p4e6-missing-peak-sources/failures.jsonl`
- `outputs/audits/chemvector-g3-p4e6-missing-peak-sources/index.html`
- `outputs/audits/chemvector-g3-p4e6-missing-peak-sources/overlays/`

## 实际结果

top1024严格重放32,342 matched、58 missing、43张失败图；全部峰上限未触发。58个漏点分为：

| 类别 | 数量 | 解释 |
|---|---:|---|
| matching conflict | 28 | top1024内有6px候选，但一对一最大匹配无法同时覆盖邻近GT |
| below top1024 | 15 | 全部峰中有6px候选，但排名在3,254–86,560 |
| GT cell不是regional maximum | 15 | 全部峰中最近候选仍在6px之外 |

没有单一独占类别达到29/58，正式结论`mixed_missing_peak_sources`。但局部状态给出了更强的共同
信号：52/58个GT cell本身不是regional maximum；这不是简单的相邻GT问题，因为只有11/58为
相邻GT cell、7/58的3×3竞争者是另一GT。

matching-conflict漏点的最近另一GT距离中位数仅3.23px，确认它是局部多对象覆盖问题，不是单点
分数阈值。below-top1024的峰排名中位数为51,073，继续把候选从1024扩到几千没有实用价值。
无6px峰的15个漏点距最近完整峰为6.12–10.57px、中位数7.16px，更像局部坐标/offset越界，
而非整块视觉证据消失。

按variant计数为coordgen/original/rdkit/stress `6/18/8/26`；按Count桶为
`4/6/6/13/29`。stress和81–100仍最难，但所有桶、所有variant都有失败，不能做专项数据修补。

本轮生成43张叠图：绿圈为已匹配GT，红圈为漏GT，橙圈与连线为最近完整峰。审计耗时44.51秒。

## 决策

停止继续扩大同一regional-max候选和训练全局selector。下一步拆成两个零训练审计：

1. 对top1024比较预测offset、cell-center坐标及6/8/10/12px曲线，量化15个近距离越界是否来自
   offset定位；
2. 对28个matching conflict构造局部GT/candidate连通分量，确认需要输出几个独立对象，作为
   局部multiplicity/set decoder的监督目标。

不直接针对stress或大分子训练；两项都是一般局部几何机制。

## 产物哈希

- audit：`d689bc0980439a916c3cba02ca4f9c7d57b54694e1f408c0ba6e0cf93fb07db3`；
- failures：`6a1a32937649eacd393bf06be8b32aac755f4a4ebf39cb5c2cc049dfc2b23006`；
- HTML：[本地叠图索引](../../../outputs/audits/chemvector-g3-p4e6-missing-peak-sources/index.html)；
- confirm_v2、fresh、USPTO、MR-TADF和sealed_test均未读取。
