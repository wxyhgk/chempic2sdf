# G3 P19-E4 Presence Threshold Calibration

> 日期：2026-08-10
>
> 状态：已完成，校准机制 GO、整体门 NO-GO
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

## 实际结果

calibration/eval 各为 160 个 molecule、160 个唯一 scaffold、640 图；二者以及 pilot/P19-E3
old dev 间的 molecule/scaffold overlap 全部为 0。calibration 共审计 288 个精确区间边界，
544/640 张图存在非空整图 exact 阈值区间。

calibration 选择的全局 threshold 为 `0.99609375`：

- calibration exact 从 threshold 0.5 的 435/640 提高到 529/640；
- Edge precision/recall 从 94.67%/99.96% 变为 98.00%/99.82%。

锁定 threshold 后，独立 eval640 结果为：

| 指标 | P19-E3 @0.5 | P19-E3 @0.99609375 | 旧 v0.10.3 @0.85 |
|---|---:|---:|---:|
| Overall exact | 437/640 | **538/640** | 531/640 |
| 普通三布局 | 379/480 | **456/480** | 453/480 |
| Stress | 58/160 | **82/160** | 78/160 |
| Crossed | 4/39 | **7/39** | 0/39 |
| Crowded | 18/37 | **30/37** | 29/37 |
| Long bond | 34/43 | 39/43 | **40/43** |
| Near-nonbonded | 2/41 | 6/41 | **9/41** |
| Edge precision | 94.37% | 98.09% | **99.39%** |
| Edge recall | **99.98%** | 99.83% | 98.67% |

校准模型为 422 FP / 36 FN，旧模型为 131 FP / 288 FN。逐图相对旧模型为 21 rescue / 14
break：普通布局合计 9/6，stress 12/8；crossed 为 7/0，near-nonbonded 为 2/5。

## 决策

预注册的整体决策为 `presence_threshold_calibration_no_go`：数据边界、校准收益、overall、普通
布局、crossed、paired 净增益与冻结状态门全部通过；stress 只提高 4 张而非 8 张，且
near-nonbonded 从 9 降到 6，两个专项门失败。

但“校准机制”本身明确 GO：独立 eval 从 437 提升到 538，并超过旧模型 531。这证明 P19-E3
已经学到可泛化的 AtomPair relation，此前看起来“越训越差”的主要原因是把未经校准的 0.5
当成最终阈值。剩余问题不是整体架构，而是新模型以 291 个额外 FP 换取 252 个 FN 的恢复，
这些 FP 主要集中在 crossed 与 near-nonbonded。

下一步保留 P19-E3 best 与固定 threshold，不重训整个 head。应冻结两套模型，先审计旧
v0.10.3 与新模型在 calibration 上的 logit 融合/保守分歧规则，并锁定到同一 eval；目标是保留
crossed 的 7 个 rescue，同时恢复 near-nonbonded 的旧模型精度。若零训练融合仍失败，再训练
只处理高置信分歧 pair 的小型整图 context residual。

## 产物

- 审计目录：`outputs/audits/chemvector-g3-p19e4-presence-threshold-calibration/`；
- `audit.json` SHA256：`5a8ee3ff2d65f5419bb0d91d29ac38d9272961a1818f8f4828b73802729493bc`；
- `eval-rows.json` SHA256：`702f55b151631047700969f41272313101be4c43d49c4a4cffdc37aa4739cd2d`；
- `paired-rows.json` SHA256：`4fae07aaf5ac41cf18a338817189360b0e066b871b7d40f3e6892660578c9308`；
- calibration manifest SHA256：`05a7b51067c0453086ef0d69e725f080cc92ce6f5d80a198d7cb2d7f61cbc9da`；
- eval manifest SHA256：`93fa7bfa43091125860570250cd73f2b38a39b1f12d0ed28a826da70fe3546da`；
- 耗时：61.10 秒；
- 本轮无训练，不创建模型 tag。
