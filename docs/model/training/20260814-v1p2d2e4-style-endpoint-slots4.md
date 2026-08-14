# v1 P2-D2-E4 Style Endpoint Slots4

> 日期：2026-08-14
>
> 状态：已完成，正式 GO
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

## 实际执行

- fit 91,740 张全部读取，固定一个 epoch；
- 参数量 357,004；
- final train loss 1.01549；
- 总耗时 1,594.94 秒；
- checkpoint 在任何 holdout 读取前原子写入；
- style holdout 一次性读取 4,008 张；
- fresh holdout/sealed test 读取数均为 0。

## 结果

| 数据 | Budget | Port recall | Pairable recall | Port exact |
| --- | --- | ---: | ---: | ---: |
| Style calibration | max256 | 98.57% | 98.14% | 86.09% |
| Style calibration | max512 | 99.68% | 99.59% | 90.72% |
| Style holdout | max256 | 99.05% | 98.57% | 87.35% |
| Style holdout | max512 | **99.71%** | **99.62%** | **91.37%** |
| Fresh calibration | max256 | 97.87% | 97.18% | 75.75% |
| Fresh calibration | max512 | **99.74%** | **99.69%** | **91.00%** |

fresh stress 在 max512 的 port/pairable 为 99.46%/99.30%，crossed-bonds 为
99.01%/98.70%。相对 E2 的旧 endpoint max512 96.93%/96.55%，分别提高 2.82pp/3.15pp；
证明主要问题确实是旧 P2-B 仅 8,056 张训练覆盖不足，而不是 bond-first 路线不可行。

14 项能力与隔离门全部通过，决策为 `slots4_style_endpoint_go`。下一步冻结本 checkpoint，使用
max256、threshold 0.05 重跑 fresh top8 pair Oracle；不能直接用 max512 指标宣称键候选已通过。

## 产物

- checkpoint：`outputs/checkpoints/chemvector-v1p2d2e4-style-endpoint-slots4/final.pt`；
- result：同目录 `result.json`。

SHA256：

- `final.pt`：`f9753b38c5484eea646e70b4b3e40bf0715f667804c239d510114939fdc9590d`；
- `result.json`：`4d0207b223fa6acc574096450997828f74b57930f93e83f72eedaef27b146916`。
