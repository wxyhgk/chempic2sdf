# G3 P1-O Atom Center Field Overfit32

> 日期：2026-08-03
>
> 状态：容量 NO-GO
>
> 性质：冻结 P1-A hidden 上的 771 参数线性单中心场

## 目标与模型

P1-N 证明 regional maxima 优于固定 `3px` component，但冻结 P1-A 每图仍产生约 566 个峰。
P1-O 在 detached `object_features [B,256,192,192]` 上只训练：

```text
Conv1x1 256 -> 1 center logit   257 参数
Conv1x1 256 -> 2 offset logits  514 参数
合计                           771 参数
```

权重严格复制 P1-A slot0，step0 center/offset逐tensor相等。训练使用单cell one-hot focal和
`8 * offset SmoothL1`，不训练Count/Bond/OCR，不使用threshold、NMS或像素聚类。

## 固定协议

- 数据：`overfit_32.jsonl`，32张、8个分子、1,424个Atom；
- batch 4，2,400 updates，9,600 exposures；
- AdamW `3e-4`、weight decay `0.01`、clip `1.0`、CUDA BF16；
- vision和完整P1-A冻结；只选择fixed-final，不选中间step、不延长；
- 容量门：known-count与top128都必须`1424/1424、32/32`。

## 结果

| step | known matched | known exact | top128 matched | top128 exact | 峰/图 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1400/1424 | 26/32 | 1408/1424 | 28/32 | 588.63 |
| 400 | 1405/1424 | 26/32 | 1412/1424 | 28/32 | 628.91 |
| 800 | 1404/1424 | 26/32 | 1411/1424 | 28/32 | 640.84 |
| 1200 | 1405/1424 | 26/32 | 1412/1424 | 28/32 | 665.00 |
| 1600 | 1407/1424 | 26/32 | 1414/1424 | 28/32 | 668.94 |
| 2000 | 1407/1424 | 26/32 | 1414/1424 | 28/32 | 678.63 |
| 2400 | 1409/1424 | 26/32 | 1415/1424 | 28/32 | 686.75 |

线性读出提高局部matched，但整图exact完全不变，且低分regional maxima继续增加。冻结hidden上的
pointwise线性可分性不足，不能进入train5k。

```text
decision = capacity_no_go
checkpoint = outputs/smoke/chemvector-g3-p1o-atom-center-field-overfit32/step-002400.pt
SHA256 = 33defc7e733d333469d29e38a3cefd4b7ed830a89fa233d78e0c2ec0f7f5712e
```
