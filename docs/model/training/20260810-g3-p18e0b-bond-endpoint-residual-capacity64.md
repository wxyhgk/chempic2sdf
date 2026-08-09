# G3 P18-E0B Bond Endpoint Residual Capacity64

> 日期：2026-08-10
>
> 状态：已完成，容量 GO
>
> 模型：`chemvector-g3-p18e0b-bond-endpoint-residual-capacity64`

## 目标

P18-E0A 已排除 top192 排名截断，并确认多数 stress Bond 位于 12--24px 的端点误差带。本轮先做
严格容量实验，不直接扩量：验证冻结 proposal 身份后，一个小型 proposal 级残差头是否有能力
修正两个端点，同时完全保留 easy 图。

## 固定协议

- 数据只来自 P4 pilot train 的 stress 图，不读取 fresh、sealed test、confirm、USPTO 或 MR-TADF；
- 32 easy：冻结 top192 在 12px 已整图正确；
- 32 hard：12px 不正确但 24px Oracle 整图可达，其中 `crossed_bonds` 至少 8 张；
- 冻结 v0.10.1 Bond vision/head，缓存 top192 的 ID、128D feature、endpoint、type、score 和 mask；
- 模型输入为 feature、base endpoint、type 与 score，仅输出最多 32px 的两个端点残差；
- 候选 ID、mask、score、type 和数量不可修改；
- 44,676 个以内参数，batch8、3,000 step、AdamW `3e-4`、weight decay 0、grad clip 1；
- unmatched proposal 使用 preservation loss，避免为修正正例任意移动其他对象；
- 每 500 step 固定评估，checkpoint 以 easy exact、hard exact、总 matched 的字典序选择。

## GO/NO-GO

- 64/64 在训练前均通过 top192/24px Oracle；
- easy 保持 32/32，零 break；
- hard 至少救回 24/32；
- 总体至少 56/64 exact；
- hard 中至少 8 张 `crossed_bonds`；
- 冻结 source state 前后逐位一致。

无论 GO/NO-GO 都保存 checkpoint、提交结果，并创建与模型名完全一致的 annotated Git tag。
容量 NO-GO 时禁止直接扩量；下一步只能增加候选端点附近的局部像素证据。

## 实际结果

train-only pilot 共扫描 344 张 stress 图后确定性得到 32 easy + 32 hard。hard 分桶为
`crossed_bonds=8`、`crowded_subgraph=9`、`near_nonbonded=9`、`long_bond=6`；64 张全部通过
top192/24px Oracle，冻结 source state 不变。

| 指标 | Step 0 | Best step 2500 |
|---|---:|---:|
| 总体 Bond exact | 32/64 | **62/64** |
| Easy exact | 32/32 | **32/32** |
| Hard exact | 0/32 | **30/32** |
| Hard Bond recall | 94.57% | **99.81%** |
| Crossed hard exact | 0/8 | **8/8** |

全部容量门通过，判定 `endpoint_residual_capacity64_go`。模型只有 35,742 参数，训练与 cache
构建总耗时 12.96 秒；step 3000 与 step 2500 同为 62/64，因此按预注册字典序保留更早的
step 2500。

产物：

- best：`outputs/checkpoints/chemvector-g3-p18e0b-bond-endpoint-residual-capacity64/best.pt`；
- best SHA256：`a2e2164e1f7de83dc891a8369b92bcd43657be73b995b54f85b79824459f0357`；
- capacity cache SHA256：`87dd385720d8757b9047763fb62d4fb88bba13d691b52a1ff57662acad3ae77e`；
- capacity sample key SHA256：`b131f93d760f659a6d3025f5c524199ce23b7db8a3fa61f2040482b5ebf33229`。

本轮只证明 feature-only endpoint residual 的容量，不允许据此扩量。下一步固定 best step 2500，
在 fresh640 上做零训练泛化评测；失败时再增加端点局部像素证据。
