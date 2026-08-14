# v1 P2-D2-E7 Bond Query Set Capacity64

> 日期：2026-08-14
>
> 状态：已完成，正式 NO-GO
>
> 模型：`chemvector-v1p2d2e7-bond-query-set-capacity64`

## 原因

E6 的共享 endpoint edge classifier 后期 pair precision 达 99.59%，但 known-count unique recall
只有 80.39%、exact 2/64；同一 GT Bond 的多个等价候选仍会同时进入 top-K。固定像素 NMS 能
提高 recall，却不能解决 count，也会重新引入风格相关半径。本轮不调 NMS 半径、不增加 E6
训练步数，而是把最终输出从“全部候选 edge 的二分类”改为真正的固定 Bond 集合。

## 冻结协议

- 复用 E6 group-set Oracle exact 64/64 的同一 cache，SHA256：
  `422fda8a3b592a3e5659d47ea50d26d4782c930b02e7e0c3812192a45d8b89b9`；
- P1-A separator、E4 endpoint、0.05 threshold、max256、每端 top8 partner 全部冻结；
- 不读取 style calibration/holdout、fresh 任一分组、sealed test、MR-TADF 或 USPTO；
- 不生成 PNG，不加入键级、OCR、AtomObject 或 SDF 编译。

## 架构

node/edge encoder 与两层共享 endpoint 消息传递产生约 1,000 条 edge memory。固定 128 个可学习
Bond query 经过两层 Transformer decoder 对 edge memory 做 cross-attention，每个 query 输出：

- `existence_logit`：该槽是不是一个真实 Bond；
- `pointer_logits`：该 Bond 指向哪条候选 edge。

训练时先把同一 GT Bond 的几何等价 edge 合成一个概率组，再以组概率构造 Hungarian 代价；
每条 GT Bond 只匹配一个 query，未匹配 query 全部监督为 no-object。这样重复抑制和 cardinality
属于输出协议本身，不依赖固定 px NMS。推理固定 existence threshold 0.5，不做阈值扫描。

## 预注册门

与 E6 保持同样的严格容量门：known-count exact 64/64、0.5 threshold count accuracy 64/64、
predicted exact 64/64，且四种布局分别 16/16。任一失败即 NO-GO，不进入 style 扩量。

## 执行命令

```bash
.venv-train/bin/python -m \
  chemvector.training.bonds.objects.train_v1p2d2e7_bond_query_set_capacity64
```

## 实际结果

967,305 参数训练 4,800 个单图 optimizer step，耗时 47.30 秒。冻结 step 4800：

- known-count precision/recall：97.04%/30.69%，image exact 0/64；
- 0.5 threshold count accuracy：3/64，MAE 14.45；
- predicted F1：37.90%，image exact 0/64；
- 训练过程最高 known recall 仅 41.73%（step 2400）。

正式判定 `bond_query_set_capacity_no_go`，不进入 style 扩量。单图连续更新时 Hungarian
assignment 持续重排，0.5 阈值输出数在接近 0 与大幅过选之间震荡，并出现单样本 loss 超过
100；说明当前实验首先是优化稳定性失败，不能用它否定 query-set 表示。

下一步先做固定单图 overfit 隔离：若单图仍不能 100%，检查 pointer/group loss；若单图能
100%，则保持模型与损失，改用多图梯度累积、较低学习率和 assignment 稳定策略，再回到
capacity64。不得直接增加本轮单图 SGD 的步数。

## 产物

- checkpoint：
  `outputs/checkpoints/chemvector-v1p2d2e7-bond-query-set-capacity64/final.pt`；
- checkpoint SHA256：
  `3e49d063bee4d367cd388a7685e0f7637e7b98a3b481288cf096c3f1c9cefece`；
- result SHA256：
  `1ca5a14933d12795ca994e8ad9f9dfcbe34e3ea67cef098693088cd2b7fbc4a5`；
- 模型 tag：`chemvector-v1p2d2e7-bond-query-set-capacity64`。

所有保留集读取数为 0，生成 PNG 数为 0。
