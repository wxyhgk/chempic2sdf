# v1 P2-D2-E3 Style Endpoint Field

> 日期：2026-08-14
>
> 状态：已预注册，等待训练
>
> 模型：`chemvector-v1p2d2e3-style-endpoint-field`

## 目标

P2-D2-E2 已排除 threshold 和 maximum-points 是 fresh endpoint 不可达的主因。本轮验证最直接
的数据量假设：保持 P2-B endpoint 架构与 P1-A separator 不变，从 P2-B best 初始化，在
`style_train_100k` 安全集上从 8,056 张扩到 91,740 张唯一 fit 图。

本轮只学习 visible Bond port，不训练 pair、键级、OCR、AtomObject 或 SDF。

## 固定输入

- P1-A separator SHA256：
  `490b5daae3a2059c3d335785a69e0c2a41eed32f0237ef77fc3556c36d134a74`；
- P2-B 初始化 SHA256：
  `3ec7bf27e96f166e0f30e7c5c8b79f33860701acfeb3e16297015fc7a91511ca`；
- E3A safe manifest SHA256：
  `ee41d57275d8c48ebe9461b69eeef5bbb90abaa10de9319b746f8274024686e6`；
- fit/calibration/holdout：91,740/3,588/4,008 张，scaffold 重叠为 0；
- 模型参数 356,713，P1-A 完全冻结；
- AdamW，学习率 `1e-4`、weight decay `1e-4`、batch 64；
- 固定只训练一个完整 epoch，不按中间 checkpoint 或 fresh 指标选择 epoch。

训练目标保持 P2-B 的三 slot stride-4 endpoint heatmap、格内 offset、集合排序与 cell-count
损失。batch=64 的 5090 单步探测占用 25.33 GiB、耗时 1.41 秒。

## 评测隔离

训练完成并原子写入 `final.pt` 后依次评测：

1. style calibration，仅作诊断；
2. 已消费的 fresh calibration 100 分子/400 图，确认跨 scaffold/held-out 字体方向；
3. style holdout 1,002 分子/4,008 图，一次性正式确认。

fresh holdout、sealed test、MR-TADF、USPTO 均不读取。评测固定 threshold 0.05，同时报告
max256/max512，不扫描阈值。

## GO/NO-GO

max512 同时满足：

- style holdout unique-port recall 至少 98%；
- style holdout pairable-Bond recall 至少 97.5%；
- fresh calibration unique-port recall 至少 97.5%；
- fresh calibration pairable-Bond recall 至少 97%；
- fresh stress pairable-Bond recall 至少 95%；
- 只训练一个 epoch，P1-A 冻结；
- fresh holdout 与 sealed test 读取数为 0。

GO 后也不能直接接 graph-set；下一步必须用冻结新 endpoint 重跑 fresh top8 pair Oracle。NO-GO
则否定“只需给同架构更多数据”，改 endpoint 表示或损失。

## 执行命令

```bash
.venv-train/bin/python -m \
  chemvector.training.bonds.objects.train_v1p2d2e3_style_endpoint_field
```

## 产物

- checkpoint：`outputs/checkpoints/chemvector-v1p2d2e3-style-endpoint-field/final.pt`；
- result：同目录 `result.json`。
