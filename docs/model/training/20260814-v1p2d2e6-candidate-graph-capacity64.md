# v1 P2-D2-E6 Candidate Graph Capacity64

> 日期：2026-08-14
>
> 状态：已预注册，训练中
>
> 模型：`chemvector-v1p2d2e6-candidate-graph-capacity64`

## 目标

P2-D1 对每个候选 pair 独立分类，无法看到多条边共享同一个 endpoint，也没有整图键数约束；
其 holdout 的 known recall 只有 73.04%，说明独立 pair MLP 不能利用候选图已经提供的结构信息。
E4/E5 已把 fresh max256/top8 候选 Oracle 恢复到 95.53% pair recall、59.50% image exact。
本轮先做固定 64 图容量门，验证共享 endpoint 的图消息传递和集合目标是否具有表达能力，不在
容量未通过时直接扩到 91,740 张。

## 冻结输入

- P1-A separator SHA256：
  `490b5daae3a2059c3d335785a69e0c2a41eed32f0237ef77fc3556c36d134a74`；
- E4 endpoint checkpoint SHA256：
  `f9753b38c5484eea646e70b4b3e40bf0715f667804c239d510114939fdc9590d`；
- style safe manifest SHA256：
  `ee41d57275d8c48ebe9461b69eeef5bbb90abaa10de9319b746f8274024686e6`；
- endpoint threshold 0.05、maximum endpoints 256、每端 line-score top8 partner union；
- 匹配半径 6 px，不设最大键长；
- style calibration/holdout、fresh 全部分组、sealed test、MR-TADF、USPTO 均不读取。

## Capacity64 缓存

每个落在 GT 端口 6 px 邻域内的 endpoint 都共享同一端口语义；每个落在 GT Bond 无序端点
邻域内的 pair 都进入该 Bond 的等价候选集合。训练时每组只选择一个 latent winner，评测也
要求每条 Bond 恰好恢复一次。只保留整图全部可达的样本，四种布局各 16 张；异常或不可达图
不生成 PNG，也不进入 capacity 集。

已生成缓存：

```text
outputs/audits/chemvector-v1p2d2e6-candidate-graph-capacity64-cache/capacity64.pt
```

缓存 SHA256：
`422fda8a3b592a3e5659d47ea50d26d4782c930b02e7e0c3812192a45d8b89b9`。

实际扫描 144 张，从中固定 64 张；四种布局各 16 张，共 1,958 条 GT Bond、2,206 个 GT
端口、11,034 个候选 endpoint 和 59,625 条候选 pair。64/64 图的组级候选 Oracle exact 为
100%，最大 GT 端点度为 4，生成 PNG 数为 0。

## 模型与损失

候选 endpoint 是 node，top8 partner pair 是 edge。node 输入为归一化 x/y、endpoint score、
候选 incidence count；edge 输入保留现有 64 维 ribbon/几何证据。两层
`edge -> node -> edge` 消息传递使用共享 endpoint 聚合，因此 A-B-C 中的 A-C 假边、同一
junction 周围的竞争边不再彼此独立。

训练目标同时包含：

- 每个 GT Bond 的集合内 latent one-of-K edge BCE 与 listwise margin；
- 同一 GT Bond 重复候选抑制；
- endpoint keep 与 incident degree 辅助监督；
- edge 概率总数、整图独立 bond-count 以及两者一致性损失。

键级、OCR、AtomObject 和 SDF 编译均不在本轮，避免混入无关变量。

## 预注册门

必须同时满足：

1. capacity64 的 known-count image exact 为 64/64；
2. 模型自身 bond-count accuracy 为 64/64；
3. 自身 count + edge ranking 的 image exact 为 64/64；
4. coordgen/original/rdkit/stress 四组各自均为 16/16 predicted exact；
5. 所有冻结哈希一致，所有保留集读取数为 0。

任一失败即 `candidate_graph_capacity_no_go`，不得进入 style 规模训练。容量门通过只证明表示和
优化可拟合固定候选图，不代表新 scaffold 或新字体泛化。

## 执行命令

```bash
.venv-train/bin/python -m \
  chemvector.training.bonds.objects.train_v1p2d2e6_candidate_graph_capacity64
```

## 产物

计划写入：

```text
outputs/checkpoints/chemvector-v1p2d2e6-candidate-graph-capacity64/
```

训练完成后补记实际 step、指标、checkpoint SHA256 和正式 GO/NO-GO，并创建与模型名完全相同
的 annotated tag。
