# v1 P2-D2-E4 Style Endpoint Slots4

> 日期：2026-08-14
>
> 状态：已预注册，等待训练
>
> 模型：`chemvector-v1p2d2e4-style-endpoint-slots4`

## 目标

E3B 全量审计证明 99,336 张中最大 stride-4 cell occupancy 严格为 4，只有三张超过旧三
slot 容量。本轮将 endpoint 输出扩为四 slot，保留所有合法拥挤样本，重新执行 E3 原定的一轮
style 扩量训练。

## 初始化与固定项

- P2-B 初始化 SHA256：
  `3ec7bf27e96f166e0f30e7c5c8b79f33860701acfeb3e16297015fc7a91511ca`；
- backbone、heatmap 前三 slot、offset 前六通道逐张量复制；
- 第四 heatmap slot 和第四组 x/y offset 按 seed 20260825 使用模块默认初始化；
- P1-A separator 和 E3A safe manifest 不变；
- 参数量仍为约 35.7 万，仅输出层增加少量参数；
- fit 91,740 张，固定一个完整 epoch；
- AdamW、学习率 `1e-4`、weight decay `1e-4`、batch 48；
- 不恢复 E3 中止时的半轮 optimizer 状态。

batch 从 64 调到 48 是 E3 首次 OOM 后的工程稳定性修正，不改变样本、step 定义、loss 或能力
门。训练必须实际读完 91,740 张，不能按固定 step 提前结束。

## 评测与门控

完全复用 E3 的固定评测：checkpoint 先落盘，再读 style calibration、fresh calibration，最后
一次性读 style holdout。threshold 固定 0.05，报告 max256/max512。

max512 同时满足：style holdout port/pairable 至少 98%/97.5%，fresh calibration 至少
97.5%/97%，fresh stress pairable 至少 95%。此外必须 slots=4、全部 safe 样本保留、只训练
一轮、fresh holdout/sealed test 读取数为 0。

GO 后重跑 fresh top8 pair Oracle；NO-GO 则否定“slots 修正后同架构扩量足够”。

## 执行命令

```bash
.venv-train/bin/python -m \
  chemvector.training.bonds.objects.train_v1p2d2e4_style_endpoint_slots4
```

## 产物

- checkpoint：`outputs/checkpoints/chemvector-v1p2d2e4-style-endpoint-slots4/final.pt`；
- result：同目录 `result.json`。
