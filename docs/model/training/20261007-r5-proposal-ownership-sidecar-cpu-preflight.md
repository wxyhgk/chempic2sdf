# R5 proposal ownership sidecar CPU preflight

本次只评估一个 proposal-sidecar 候选：对 LSD provisional node 与冻结
scored vertex candidate 的 pair 做 target-free quality scoring，再把 score 作为
Hungarian ownership 的 cost regularizer。它不负责 edge/type，不增加 candidate，
也不能通过阈值删除 frozen proposal prefix。

## 标签和特征契约

训练标签仅在 train identity 上由评估器事后生成：line node 和 candidate 映射到同一
图中 atom 才是正样本。推理时 sidecar 不读取 layers、graph、atom id 或 bond。特征为
`distance`, `distance^2`, candidate score/rank, local candidate density/count 和
相对 `dx/dy`；不含坐标对应的 target 或任何生产 checkpoint 参数。holdout 只在
proposal 完成后用于统计。

## CPU 结果

500-step CPU MLP（10 维 pair feature）得到 ownership holdout P/R=`0.9005/0.9804`。
但将 top-1/threshold ownership 直接重放图会把 full holdout graph P/R/sanitize
降到 `0.9686/0.6667/0.8502`；因此该解码契约不可接受。

保守地仅将 logits 加入 Hungarian cost、保留所有可行 candidate（不 merge、不删除）
的 alpha sweep，最佳初步点 alpha=2 为 graph P/R/sanitize=`0.9681/0.8253/0.9536`。
它仍未超过已审查的 Hungarian+merge+guard baseline
P/R/sanitize=`0.9633/0.8513/0.9515`，且尚未通过 profile-level strict gate。
将同一 scorer 放回已有 merge=20px 路线后，alpha=2 为
`0.9669/0.8309/0.9641`：sanitize 虽升，但 recall 下降 2.04 个百分点，仍为 NO-GO。

历史同类 endpoint-pair sidecar 的 holdout precision 为 `0.8052`，port endpoint
sidecar 的 holdout AUPRC/precision 仅 `0.2578/0.0535`；不能据此重复训练。

## 决策

`GPU shadow: NO-GO`。在出现一个新的、完全 target-free 的 endpoint/provisional-edge
proposal route，且 full holdout 与每个 profile 同时严格改善现有 guard baseline 以前，
不启动 800-step GPU。生产 checkpoint、selector、OCR、bond/type head 和 sealed test
均未触碰。
