# Inbox

当前不得放真实输入。数据保管员只能在 proposer 已冻结且 public commitment 已写入后，放入 20 份
original 和 5 份 causal erase 的匿名 canonical input JSON。

模型运行员只能逐文件把完整 JSON 送入 proposer stdin。外层 runner 可以持有 opaque key 以保存输出，
但 opaque key、文件名和输入 SHA 均不得进入 proposer 进程。inbox 不得包含 SDF、annotation、sidecar、
difficulty、GT、原 molecule ID 或任何软链接。
