# G3 P19-E5 Old/New Logit Fusion

> 日期：2026-08-10
>
> 状态：已完成，融合有效、Pareto NO-GO
>
> 类型：零训练双模型融合审计，不创建模型 tag

## 目标

P19-E4 证明 P19-E3 在阈值校准后 overall 538/640，超过旧 v0.10.3 的 531/640，并将
crossed 从 0 提到 7；但 near-nonbonded 从旧模型的 9 降到 6。两模型错误具有互补性，本轮
先验证不训练的归一化 logit 融合能否同时保留两者优势。

## 固定协议

- 模型固定为 P19-E3 best 与旧 v0.10.3，不训练、不改权重；
- 数据严格复用 P19-E4 的 calibration640/eval640，所有 split 边界不变；
- 新模型 logit 减去 threshold `0.99609375` 对应 logit，旧模型减去 `0.85` 对应 logit；
- 融合分数为 `alpha * new_margin + (1-alpha) * old_margin`；
- calibration 只扫描 alpha `0.00, 0.05, ..., 1.00`；每个 alpha 只按整图 exact 区间选择
  一个全局 score threshold；
- winner 依次按 calibration overall exact、stress exact、Edge F1 选择；
- alpha 与 threshold 锁定后才读取 eval 标签；
- 不读取 fresh640、confirm、sealed test、USPTO 或 MR-TADF。

## GO/NO-GO

融合不是“总分略高”就算成功，必须在 eval 同时满足：

- overall 至少 538/640；
- 普通三布局至少 456/480；
- stress 至少 82/160；
- crossed 至少 7；
- near-nonbonded 至少 9；
- 相对旧模型 rescue 多于 break，相对新模型不得净回退；
- 所有冻结 state 前后不变。

全部通过才允许将双模型融合视为能力候选。若失败，说明全局线性融合不能表达“何时信任哪个
模型”，下一步训练的只能是高置信分歧 pair 上的小型整图 context router，不能重训整个主干。

## 实际结果

calibration 在 21 个 alpha 中选择：

- `alpha_new=0.55`；
- `alpha_old=0.45`；
- score threshold `-0.7528127431869507`；
- calibration exact 543/640、stress 91/160。

参数锁定后的 eval640：

| 指标 | 新模型 | 旧模型 | 融合 |
|---|---:|---:|---:|
| Overall | 538 | 531 | **538** |
| 普通三布局 | 456 | 453 | **456** |
| Stress | 82 | 78 | **82** |
| Crossed | **7** | 0 | 4 |
| Crowded | 30 | 29 | **30** |
| Long bond | 39 | **40** | 39 |
| Near-nonbonded | 6 | **9** | **9** |
| Edge precision | 98.09% | **99.39%** | 98.83% |
| Edge recall | **99.83%** | 98.67% | 99.58% |

融合为 255 FP / 91 FN。相对旧模型为 16 rescue / 9 break；相对新模型为 7 rescue / 7 break。
它完整恢复 near-nonbonded 的旧模型 9/41，同时保持 overall、普通布局和 stress，但将新模型
crossed 的 7/39 降为 4/39。

## 决策

正式决策为 `old_new_logit_fusion_no_go`。九项门中只有 `crossed>=7` 失败，其余全部通过。
因此全局线性融合机制有效，但不满足严格 Pareto 门；不能作为最终能力版本。

证据表明剩余问题不是 alpha 精度：全局混合为了压 near 的 FP，必然削弱 crossed 的真边召回。
下一步先做旧/新/融合三路线逐图 Oracle union 与输出分歧特征审计，确认理论上能否同时保留
overall 538、crossed 7、near 9，以及无需像素的输出级 router 是否有分辨信号。只有 Oracle
上限足够且可观测特征有容量，才训练小 router；否则直接进入局部视觉 context residual。

## 产物

- 审计目录：`outputs/audits/chemvector-g3-p19e5-old-new-logit-fusion/`；
- `audit.json` SHA256：`41fc704c8ed96d0e994ce85fba841787f9e62420eb45788371608b783a9e5592`；
- `fused-eval-rows.json` SHA256：`e7c610df1c3aa4c16c49f24b53bb0cb767b423a3cf0a2e37645d2163c26f8c3d`；
- `paired-vs-old-rows.json` SHA256：`0e9f7589c2cd41c24a388e920937c1f2489d116a870da32b417a88154960075f`；
- `paired-vs-new-rows.json` SHA256：`634739e4b33ece7f491b610a6172d36f85b817058163f9fa49e2e30d18c71306`；
- 耗时：67.82 秒；
- 本轮无训练，不创建模型 tag。
