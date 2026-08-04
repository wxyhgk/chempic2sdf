# G3 P3-C Fixed Assignment Replay Train4937

> 日期：2026-08-04
>
> 状态：固定身份replay部分有效 / 整体NO-GO

## 实验

P3-C从容量通过的P2-W初始化，不再逐step运行动态Hungarian。首先对4,937张候选可达训练图生成
固定`(query, GT atom, candidate)`Action Cache，两次生成完全一致，共151,210条assignment。
capacity64的query mask强制复用P2-R缓存。训练每批3张一般样本加1张capacity replay，完整模块
训练3 epochs、4,875 steps、19,494 exposures。

首次缓存运行还发现`torch.cdist`与逐对`vector_norm`在6px边界有约`2.1e-6`的float32舍入差；
最终生成与复核统一使用同一个`cdist`兼容协议，没有放宽阈值。

## 结果

| 指标 | P2-W/P2-X | P3-B动态身份 | P3-C固定身份replay |
|---|---:|---:|---:|
| capacity easy exact | 32/32 | 5/32 | **22/32** |
| capacity hard exact | 32/32 | 0/32 | **10/32** |
| route known-count exact | 4/100 | 30/100 | **13/100** |
| route predicted exact | 0/100 | 11/100 | **0/100** |
| route Count exact | 3/100 | 27/100 | **12/100** |
| route predicted matched | 2399/3080 | 2930/3080 | **2855/3080** |

固定identity replay显著缓解P3-B容量崩溃，但未恢复容量Gate，同时route能力低于动态身份训练。

## 结论

- `fixed_assignment_replay_train4937_generalization_no_go`。
- 固定标签与25% replay方向有效，但factorized existence BCE、pointer CE、coordinate L1分别降低并不
  保证joint MAP整组最优；增加replay比例或epoch只是在冲突目标上调权重，不继续尝试。
- P3-D应使用固定cache做loss-augmented fixed-cardinality matching，直接优化
  `target matching score > best competing matching score`的结构化hinge/listwise目标；Count CE可独立保留。
- P3-D仍从P2-W而不是P3-B/P3-C初始化，避免继承已确认的身份回退。

```text
model = chemvector-g3-p3c-fixed-assignment-replay-train4937
checkpoint = outputs/checkpoints/chemvector-g3-p3c-fixed-assignment-replay-train4937/step-004875.pt
checkpoint SHA256 = acdf3bd0a7c53599f7990a0e09f0082ec60b583b7581727168b8c314072494bb
Action Cache entries SHA256 = cfcc96b1fa8a5c0e5d69e668ec7a13300d091ef94b9a22525f5e315e9ad5ad86
training images = 4,937
samples seen = 19,494
trainable parameters = 301,206
decision = NO-GO
```
