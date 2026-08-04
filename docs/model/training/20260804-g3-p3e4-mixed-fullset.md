# G3 P3-E4 Mixed Coverage Full-set Pilot256

> 日期：2026-08-04
>
> 状态：NO-GO；general 学习与 capacity 保护尚未同时成立

## 目标与协议

将 P3-E3 的 epoch-wide 梯度累积扩到 P3-E1 相同的 192 general + 64 capacity。全部 256 张
每 epoch 各出现一次，30 epoch、30 optimizer update、7,680 exposures。初始化、coverage
margin、优化器与冻结范围不变，final-only Gate 沿用 P3-E1。

general192 仍是旧因果样本：57 个唯一 molecule，其中 99 张与 capacity 的 33 个 molecule
重叠；本实验不能表述为 molecule-disjoint 泛化。

## 结果

| 指标 | Step0 | P3-E1 microbatch | P3-E4 full-set |
|---|---:|---:|---:|
| easy matched / exact | 967/967，32/32 | 966/967，31/32 | 965/967，30/32 |
| hard matched / exact | 807/807，32/32 | 704/807，2/32 | 768/807，8/32 |
| general matched / exact | 5663/6202，9/192 | 6082/6202，155/192 | 6077/6202，111/192 |
| capacity coverage margin | 63/64 | 32/64（E1b复核） | 28/64 |
| general coverage margin | 5/192 | 152/192审计值 | 75/192 |

full-set 将 hard 从 P3-E1 的 `2/32` 改善到 `8/32`，general matched 接近 P3-E1，但仍未保护
capacity。原因是 192 张 general 的 active loss 在共享 sample mean 中主导更新，而已正确的
capacity 大多为零梯度；只把约束放进同一更新还不够，需要显式受约束优化。

## 决策与产物

`NO-GO`，不扩量、不接推理。下一步从 P2-W 重开 capacity-anchored A-GEM；capacity 是必须
保护的不变量，general 梯度只在不增加 capacity 一阶损失的半空间内更新。

- checkpoint：`step-000030.pt`
- SHA256：`13ab3ea350fc847faee7168aa1fc7b148aa05519215f497c9d2907e9d6edd619`
- 输出：`outputs/smoke/chemvector-g3-p3e4-mixed-coverage-fullset-pilot256`
