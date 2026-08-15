# Blind Carrier Audit v0.1

本目录是 20 张真实分子 PNG 的盲测交付骨架，目标是验证从像素提出 carrier、lane、BondBundle、
端点和 primitive ownership 的能力。它不是训练集，不允许选择或修改模型，也不授权训练。

当前骨架不包含任何真实样本 ID、PNG、SDF、annotation、GT、秘密 seed 或因果删除计划。

## 固定结论边界

- 输入只允许规范 JSON 中的一张 PNG；proposer 不得看到文件名、路径、样本 ID、SHA、difficulty、
  SDF、Render JSON、annotation 或 sidecar。
- 输出只描述可见结构候选，不输出元素、SMILES、SDF atom/bond ID 或 benchmark truth。
- 双键和三键必须表示为拥有 2/3 条 physical lane、共享两个 semantic terminal 的一个 BondBundle。
- 每个 interpretation 必须对全部 segment 给出恰好一个 owner。跨 interpretation 可以保留文字/键歧义。
- `support_runs` 必须来自输入前景像素。只有直线参数或置信度、没有非空 support mask 的 carrier
  一律无效。
- 20 张样本揭盲后永久退出训练、增强、阈值选择和 checkpoint 选择。后续复测必须使用新的、
  不重叠的盲样本。

## 三角色边界

1. 数据保管员：可读取候选池、旧 benchmark、SDF 和 annotation；负责抽样、commitment、GT 封存及
   causal erase。不得读取或运行候选 proposer。
2. 模型运行员：只能读取已冻结 proposer、单份匿名输入和公开协议；只能写 prediction 与运行回执。
   不得读取 `sealed/`、原始文件名映射或评估程序的 GT 输入。
3. 独立评估员：只能在 proposer 输出冻结后读取 prediction、sealed mapping/GT 和评分代码；不得重跑
   proposer、改变匹配规则或回写输入。

同一进程和同一脚本不得兼任三种角色。proposer 必须在空工作目录、无网络、无仓库挂载的隔离环境中，
每张图片启动一次，只从标准输入读一份 input JSON，只向标准输出写一份 prediction JSON。

## 目录

- `protocol/`：冻结协议和全部事前阈值。
- `schemas/`：closed Draft 2020-12 JSON Schema。
- `public/`：不泄露 ID 的 commitment 与 proposer freeze 回执。
- `inbox/`：proposer 冻结后才可出现的匿名输入。
- `sealed/`：仅数据保管员和揭盲后的评估员可见；当前必须为空。
- `runs/`：按 proposer freeze SHA 隔离的只写一次预测结果。
- `evaluation/`：揭盲后产生的评分结果，不得被模型运行员用于本批样本调参。

## 执行顺序

1. 冻结本协议、schemas、selector、旧 benchmark 记录和 exposure registry 的完整字节哈希。
2. 数据保管员承诺秘密 seed，按协议分层抽取 20 张，生成 5 张 causal erase，并封存映射与 GT。
3. 在任何匿名 PNG 发布前，冻结 proposer 源码、checkpoint、配置、容器、依赖和训练清单哈希。
4. 发布无 ID 的 inbox commitment；模型运行员只运行一次 proposer。
5. 冻结 predictions tree 和 run receipt；此后独立评估员才可揭盲。
6. 按四个主指标和 causal gate 评分。任何事后阈值修改均使本轮自动 `NO_GO`。

通过本轮只允许扩大到新的 200 张 fresh audit，不授权模型训练、发布或生产使用。所有机器结果必须包含
`training_authorized=false`。
