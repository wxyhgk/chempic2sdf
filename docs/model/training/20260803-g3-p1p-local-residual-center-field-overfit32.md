# G3 P1-P Local Residual Center Field Overfit32

> 日期：2026-08-03
>
> 状态：容量 NO-GO
>
> 性质：P1-O 后的最小局部非线性容量实验

## 模型

冻结P1-A slot0 center/offset为base，只训练零初始化local residual：

```text
Conv1x1 256 -> 32             8,224
GELU
Depthwise Conv3x3 32 -> 32      320
GELU
zero-init Conv1x1 32 -> 3        99
合计                           8,643
```

三个输出分别加到center、offset-x、offset-y原始logit。增加固定margin `1.0` 的局部峰损失：
每个GT cell必须高于8邻域非GT cell；相邻GT彼此不作为负例。其余数据、优化器、步数和严格
`1424/1424、32/32`容量门与P1-O完全相同。

## 结果

| 路径 | known matched/exact | top128 matched/exact | 峰/图 |
|---|---:|---:|---:|
| P1-O linear final | 1409/1424，26/32 | 1415/1424，28/32 | 686.75 |
| P1-P local residual final | **1414/1424，25/32** | **1421/1424，30/32** | 1118.91 |

全部8,643参数实际更新，冻结state不变，两次final重放逐项一致。局部上下文和peak margin确实将
top128漏点从9个降到3个，但known-count整图没有改善，严格容量门失败，禁止进入train5k。

剩余top128失败只来自两张图：

- `train_000005_coordgen`：相邻真实Atom约10.10px，落在192网格对角相邻cell；
- `train_000006_stress`：相邻真实Atom约6.71px，同样落在对角相邻cell。

8邻接regional maxima从定义上不能让两个直接相邻cell同时成为两个独立峰。这不是增加residual
宽度或epoch能解决的问题，下一步必须提高对象场分辨率。

```text
decision = local_residual_capacity_no_go
checkpoint = outputs/smoke/chemvector-g3-p1p-atom-center-field-local-residual-overfit32/step-002400.pt
SHA256 = e78089a261ccbc25ec86a52bd4ceb75330426f0dec7e9920caa5fb99430c3294
```
