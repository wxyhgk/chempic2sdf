# G3 P3-E5 Capacity-Anchored A-GEM Pilot256

> 日期：2026-08-04
>
> 状态：NO-GO；raw-gradient projection 不足以保护离散 exact 边界

## 协议

从 P2-W 重开。每 epoch 分别计算 general192 与 capacity64 的完整平均梯度；当点积为负时，
将 general 梯度投影到不增加 capacity 一阶损失的半空间，再按 `3:1` 合并，统一 clip 后交给
AdamW。30 update、7,680 exposures；Count 与其它旧损失保持冻结，final-only 判定。

日志同时记录 raw/projected cosine，以及实际 AdamW 参数位移与 capacity 梯度的点积。

## 结果

| 指标 | Step0 | E4 full-set | E5 A-GEM |
|---|---:|---:|---:|
| easy matched / exact | 967/967，32/32 | 965/967，30/32 | 964/967，29/32 |
| hard matched / exact | 807/807，32/32 | 768/807，8/32 | 767/807，5/32 |
| general matched / exact | 5663/6202，9/192 | 6077/6202，111/192 | 6079/6202，110/192 |
| capacity coverage margin | 63/64 | 28/64 | 30/64 |

第 1 epoch raw cosine 为 `-0.0082`，投影后约为 0；之后多个 epoch 也触发投影。第 3 epoch
实际 AdamW `update·g_capacity` 仍为正；其它 epoch 即使一阶点积为负，有限非线性更新也会跨过
exact 边界。A-GEM 没有提高正式指标，不能作为保护机制。

## 决策与产物

`NO-GO`。下一步不再调 PCGrad/A-GEM 权重，改为事务式实际更新检查：候选 step 更新后若
capacity known-count 不是双域32/32，则回滚模型与optimizer并按预注册倍率缩小步长。

- checkpoint：`step-000030.pt`
- SHA256：`125fe7ce0e3a3912185c49395039a25d8c1c0c8284dee28707903a187b307ae4`
- 输出：`outputs/smoke/chemvector-g3-p3e5-mixed-coverage-agem-pilot256`
