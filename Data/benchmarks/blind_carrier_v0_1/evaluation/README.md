# Evaluation

本目录只在 prediction tree 冻结并完成揭盲后使用。评分脚本读取 frozen predictions、sealed GT、
per-bond rendered layers 和事前冻结的匹配规则，但不得调用 proposer 或改变任何阈值。

必须报告四个主指标：

1. GT bond carrier recall（micro、每图最低值及 single/double/long 分桶）；
2. supported carrier precision；
3. candidate graph coverage；
4. singleton safety，包括绝对的 `false_singleton_graph_count`。

还必须报告 5 个 causal erase 的目标 carrier 删除、非目标 carrier 保留、新增 unsupported carrier 和
相对 `G - targetBond` 的 singleton safety。硬门见 `protocol/protocol_v0_1.json`。

结果无论通过或失败都必须永久记录 `training_authorized=false`。通过只允许扩大到新的 200 张 fresh
audit；本批样本不得回流训练或调参。
