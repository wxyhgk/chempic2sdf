# Blind Carrier Audit v0.2

本目录是 Blind Carrier v0.1 的并行协议版本，不包含真实样本、ID、PNG、SDF、annotation、GT、
秘密 seed 或选择结果，也不授权训练或执行盲测。

v0.2 只修正 semantic terminal 的可表达性：terminal 显式列出每个物理 segment 对齐的端点索引，
center 是这些端点的确定性整数 rounded mean；endpoint carrier disk 的相交图必须连通，但不要求
所有 disk 有公共交点。由此可表达标准三条平行 lane，同时保留 v0.1 的规范 PNG、真实前景 support、
near-overlay、同 BondBundle lane support 独占、引用和 interpretation ownership 门。

三角色隔离、单次执行、匿名 stdin/stdout、揭盲后永久退出训练以及 causal erase 规则全部继承 v0.1。
本骨架当前为空，`training_authorized=false`。

目录职责：

- `protocol/`：v0.2 事前协议；
- `schemas/`：closed Draft 2020-12 schema；
- `public/`：不泄露身份的 commitment；
- `inbox/`：proposer 冻结后才能出现的匿名输入；
- `runs/`：按 freeze SHA 隔离且只写一次的 prediction；
- `sealed/`：只供保管员及揭盲后评估员使用，当前为空；
- `evaluation/`：prediction 冻结后才可运行的独立评分。
