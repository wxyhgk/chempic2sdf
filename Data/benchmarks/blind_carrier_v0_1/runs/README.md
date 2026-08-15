# Runs

每个运行目录以 proposer freeze SHA 命名，仅包含：

- 匿名 prediction JSON；
- `run_receipt.json`；
- prediction tree commitment；
- stdout/stderr 的无敏感信息摘要。

每张输入只允许运行一次。运行目录不能包含 GT、SDF、annotation、原始 ID 映射或 evaluator 中间结果。
输出冻结后禁止覆盖；失败重跑必须作为新的 protocol run，并且不能继续使用已经揭盲的 20 张样本。
所有回执必须包含 `training_authorized=false`。
