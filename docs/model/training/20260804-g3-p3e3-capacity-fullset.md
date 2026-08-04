# G3 P3-E3 Capacity Coverage Full-set

> 日期：2026-08-04
>
> 状态：能力保护 GO / 严格 margin Gate NO-GO

## 目标与协议

验证 P3-E2 的失败是否来自 microbatch4 看不到整组约束。保持相同 P2-W 初始化、capacity64、
coverage margin、每图 30 exposures、AdamW `3e-4` 和全部冻结范围；唯一变化是每 epoch 先对
64 张图的 16 个 microbatch 累积精确 sample-mean 梯度，再执行一次 clip/optimizer update。
总计 30 update、1,920 exposures。

## 结果

| 指标 | Step0 | Final |
|---|---:|---:|
| easy matched / exact | 967/967，32/32 | **967/967，32/32** |
| hard matched / exact | 807/807，32/32 | **807/807，32/32** |
| coverage margin satisfied | 63/64 | 62/64 |

相同 exposure 下，microbatch coverage 臂为 `29/32、2/32`，full-set 则完整保住双域全部
Atom 和整图 exact，证明整组约束必须在同一次更新中可见。由于预注册要求 margin `64/64`，
本 checkpoint 仍正式 `NO-GO`，不得事后放宽。

## 产物

- checkpoint：`step-000030.pt`
- SHA256：`2b0d25c843c1bd3978b43a8f03e25da60132d747f453f66d3ead3b9fc7cc5733`
- 输出：`outputs/smoke/chemvector-g3-p3e3-capacity-coverage-fullset-pilot64`
