# G3 P3-D Structured Matching Replay Train4937

> 日期：2026-08-04
>
> 状态：结构化目标机制有收益 / 固定正matching协议NO-GO

## 实验

P3-D从P2-W初始化并复用P3-C的4,937图固定assignment cache。每张图以
`log P(exists)+log P(pointer)`为边分数，执行带归一化Hamming margin的loss-augmented固定基数
matching，直接优化缓存GT matching分数高于最佳竞争matching；Count CE单独保留。训练仍为
每批3张一般图+1张capacity replay、3 epochs、4,875 steps、19,494 exposures。

## 结果

| 指标 | P3-B factorized动态 | P3-C固定factorized | P3-D固定structured |
|---|---:|---:|---:|
| route known-count exact | 30/100 | 13/100 | **34/100** |
| route known-count matched | 2922/3080 | 2882/3080 | **2954/3080** |
| route predicted exact | 11/100 | 0/100 | **3/100** |
| route Count exact | 27/100 | 12/100 | **9/100** |
| capacity easy exact | 5/32 | 22/32 | **16/32** |
| capacity hard exact | 0/32 | 10/32 | **2/32** |

结构化目标把route known-count提高到目前最高34/100，证明直接对齐joint MAP有价值；但Count与
capacity均未过门。

## 关键诊断

structured loss从6.62降到1.08，但loss-augmented exact在第三epoch仍为0。固定cache只指定一个
`(query,candidate)`排列，而同一分子图存在许多化学等价、候选等价的正确matching。强迫网络把
一个任意排列压过其他同样正确排列，仍然制造identity冲突；结构化loss本身不是问题，固定唯一正解才是。

## 决策

- `structured_matching_replay_train4937_generalization_no_go`。
- 结构化joint目标机制保留；固定query-candidate正matching协议停止。
- P3-E改为latent structured objective：正分支每step选择当前最高分、且每个GT原子都有兼容
  candidate覆盖的matching；负分支为无约束loss-augmented joint MAP。loss只要求最佳有效集合高于
  最佳无效集合，不要求等价query身份固定。
- Count应降低与结构梯度的耦合，至少独立报告/裁剪，避免结构梯度掩盖Count学习。

```text
model = chemvector-g3-p3d-structured-matching-replay-train4937
checkpoint = outputs/checkpoints/chemvector-g3-p3d-structured-matching-replay-train4937/step-004875.pt
SHA256 = 5ee5533a15f290b4e03a990800149a2e4a4a2751d047537b0bfcef21171b8f3e
training images = 4,937
samples seen = 19,494
trainable parameters = 301,206
decision = NO-GO
```
