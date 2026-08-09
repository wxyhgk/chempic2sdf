# G3 P19-E4 Presence Threshold Calibration

> 日期：2026-08-10
>
> 状态：已预注册，等待运行
>
> 类型：零训练 calibration/eval 审计，不创建模型 tag

## 目标

P19-E3 best 的 Edge recall 为 99.98%，但固定 threshold 0.5 产生 1,391 FP。旧 v0.10.3 的
固定 threshold 是 0.85，因此必须先区分“P19-E3 logits 可分但 0.5 未校准”和“正负 pair
logits 本身不可分”，再决定是否改 loss 或增加整图 context。

## 固定协议

- 冻结 P19-E3 best step 7,500，不训练、不改权重；
- 从 full80k 删除 pilot20k、P19-E3 old dev640 的全部 molecule 与 scaffold；
- calibration 与 eval 各选 160 个 molecule、640 图，split 内 scaffold 唯一，两个 split
  molecule/scaffold overlap 均为 0；
- 每图的全局 exact 可行区间定义为 `max_negative < threshold <= min_positive`；
- calibration 枚举所有区间边界，只选一个全局 threshold；先最大化 skeleton exact，再按
  Edge F1、precision、较高 threshold 决胜；
- threshold 锁定后才读取 eval 标签；eval 同时重放旧 v0.10.3 threshold 0.85；
- 不读取 fresh640、confirm、sealed test、USPTO 或 MR-TADF。

## GO/NO-GO

- calibration/eval 各 640 图，所有 overlap 门为 0；
- calibration 选中 threshold 的 exact 必须高于固定 0.5；
- eval overall 与普通三布局不得低于旧 v0.10.3；
- eval stress 至少 +8，crossed 与 near-nonbonded 各至少 +2；
- eval paired rescue 必须多于 break；
- 所有冻结 state 前后不变。

若 GO，说明 P19-E3 的主要问题是概率校准，可将固定 threshold 写入后续模型协议；若 NO-GO，
说明单个全局 threshold 无法同时压 FP、保 recall，下一步停止 balanced/listwise scratch 目标，
改为旧模型 residual/hard-negative 或整图 pair context。
