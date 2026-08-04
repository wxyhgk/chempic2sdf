# G3 P1-W Multiscale Peak Rank Audit

> 日期：2026-08-04
>
> 状态：固定多尺度候选协议GO
>
> 性质：冻结P1-U/P1-V的零训练候选覆盖审计

## 协议

冻结P1-U stride-2和P1-V stride-1 checkpoint。每图各提取前1,024个regional maxima到CPU，
后续所有K值和多尺度组合只读取同一份缓存，不重复推理、不训练、不合并坐标。

单尺度固定K：`32/64/96/128/192/256/512/1024`。

多尺度预算按预注册顺序检查：

```text
总128：128+0, 96+32, 64+64, 32+96, 0+128
总192：128+64, 64+128
总256：256+0, 192+64, 128+128, 64+192, 0+256
总384：256+128, 128+256
```

前一项表示stride-2候选数，后一项表示stride-1候选数。只做一对一6px可达性匹配，不用target
改变候选顺序，也不使用NMS或半径聚类。

## 结果

首个双域全量覆盖协议是：

```text
stride2 top96 + stride1 top32 = 128 candidates
```

| 域 | matched | exact |
|---|---:|---:|
| overfit32 | **1424/1424** | **32/32** |
| adjacent32 | **1473/1473** | **32/32** |

单尺度对比：

- stride-2 top128：overfit `32/32`，adjacent `27/32`；
- stride-1 top128：overfit `30/32`，adjacent `32/32`；
- stride-1 top192：两个域都`32/32`；
- 多尺度96+32以更小的128预算同时达到`32/32`。

最小充分K和rank也解释了互补性：

- stride-2 overfit最大充分K为88；
- stride-2 adjacent有3张在前1,024仍不可达，另外两张最小K高达611和967；
- stride-1 adjacent最大充分K为107，全部可达；
- stride-1 overfit最大充分K为157，top128漏2张；
- 将stride-1前32个高分近邻峰补入stride-2前96个候选，正好补齐两个域。

## 重复峰约束

top128中，同一GT附近有多个兼容峰的数量很高：

| 尺度 | overfit | adjacent |
|---|---:|---:|
| stride-2 | 414 | 463 |
| stride-1 | 760 | 846 |

因此下一模型不能把某个固定candidate ID作为唯一正例。P1-X固定输入128个候选及其尺度标识，
使用动态最大基数一对一匹配产生训练期正例；推理不读取GT。第一阶段只做known-count selector
容量门，要求两个域同时全Atom、全图100%。不过门不得进入5k，也不得通过固定半径聚类消除重复。

```text
decision = fixed_multiscale_candidate_protocol_go
audit = outputs/audits/chemvector-g3-p1w-multiscale-peak-rank/audit.json
SHA256 = d9569aec96c0c3b7fad146606fca54f586eb91911aebb60b3cd93cfb037d5b35
```
