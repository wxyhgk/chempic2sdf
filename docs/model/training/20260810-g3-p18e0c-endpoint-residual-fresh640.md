# G3 P18-E0C Endpoint Residual Fresh640

> 日期：2026-08-10
>
> 状态：已完成，NO-GO
>
> 模型：固定 `chemvector-g3-p18e0b-bond-endpoint-residual-capacity64` step 2500

## 目标与协议

P18-E0B 已通过 train-only capacity64。本轮不训练、不选 checkpoint、不调半径或阈值，直接在
P18-E0 已冻结的 fresh640 top192 cache 上评测 proposal endpoint residual 的跨分子/字体泛化。

- 固定 checkpoint SHA256：`a2e2164e1f7de83dc891a8369b92bcd43657be73b995b54f85b79824459f0357`；
- 固定 12px 匹配与 top192，不重新运行或改变视觉 source；
- 报告 overall、四 variant、非 stress pooled 和四个 stress operation；
- 每桶同时报告 rescue 与 break；
- 完整推理重复两次，endpoint digest 必须一致；
- 不读取 sealed test、confirm、USPTO 或 MR-TADF。

## GO/NO-GO

- overall Bond exact ≥80%；
- stress Bond exact ≥50%；
- 非 stress pooled exact 回退不超过 2pp；
- `crossed_bonds` 至少 8/41 exact；
- 两次推理逐项一致，候选 identity/score/type/mask 不可修改。

全部通过才允许扩到小规模 train；否则 feature-only residual 泛化 NO-GO，下一步增加端点局部像素
patch，不得直接增加训练数据量。

## 实际结果

固定 step 2500 在 fresh640 上重复两次的 endpoint digest 均为
`0d682fb02b080f08c2bd67152f95212e90a2afffb3b3e8e117a6ce05663fde3d`，复现与输入隔离门通过。

| 分桶 | Base exact | Corrected exact | Rescue / Break |
|---|---:|---:|---:|
| overall | 499/640 | **492/640** | 5 / 12 |
| non-stress | 442/480 | **434/480** | 2 / 10 |
| stress | 57/160 | **58/160** | 3 / 2 |
| crossed_bonds | 0/41 | **0/41** | 0 / 0 |

overall 为 76.88%，未达到 80%；stress 为 36.25%，未达到 50%；crossed 仍为 0，三项能力门
失败。非 stress 回退 1.67pp，勉强守住 2pp 护栏，但这不能抵消总体净少 7 张 exact。

P18-E0B 的 62/64 容量结果因此只能解释为 feature-only 小头记住了 64 张候选到残差的映射，
不能证明 128D proposal feature 包含可泛化的端点证据。正式判定
`endpoint_residual_fresh640_no_go`，禁止扩大 feature-only 训练数据。

下一周期若继续，应保持相同冻结 proposal 和本轮 fresh640 门，只新增端点附近的原始像素/视觉
patch；必须先重新通过 capacity，再运行同一 fresh640，不得调整 12px 半径或 checkpoint。
