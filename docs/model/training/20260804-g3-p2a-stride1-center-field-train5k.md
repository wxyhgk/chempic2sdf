# G3 P2-A Stride-1 Center Field Train5k

> 日期：2026-08-04
>
> 状态：协议通过 / route因果GO

## 协议

从P1-V capacity64 checkpoint初始化stride-1连续768中心场和视觉最后一个block，其他设置与
P1-Z完全相同：train5k 5 epochs、6250 updates、25000 exposures；route100只读step0/final。

## 结果

| route top128 | matched | recall | exact |
|---|---:|---:|---:|
| step0 capacity64 | 2912/3080 | 94.545% | 26/100 |
| train5k final | **3073/3080** | **99.773%** | **98/100** |

stride-1初始跨域能力比stride-2更弱，但train5k带来+5.228pp recall和+72张exact，最终单尺度
上限超过stride-2。11项协议检查全部通过，两次route重放一致。

```text
model = chemvector-g3-p2a-stride1-center-field-train5k
checkpoint = outputs/checkpoints/chemvector-g3-p2a-stride1-center-field-train5k/step-006250.pt
SHA256 = 2869451245f50f98f5e3f5259179ca338e82ad81a8fd59de2d2037a10056404b
final metrics SHA256 = eba770e4cd531b62993138c916ad96816364593623edc05584e61918ed9ed3a1
```

## 候选选择

重新审计总预算128的两尺度配额后，纯stride-1 top128达到最高route上限`3073/3080、98/100`。
唯一失败的`train_001276_stress`是`crossed_bonds` stress样本，53个GT中5个密集点即使
`top1024+top1024`也没有6px兼容peak。后续保留完整GT评估，同时对selector训练显式标记可达GT。
