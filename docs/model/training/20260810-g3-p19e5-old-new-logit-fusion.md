# G3 P19-E5 Old/New Logit Fusion

> 日期：2026-08-10
>
> 状态：已预注册，等待运行
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
