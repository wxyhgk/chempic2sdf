# G3 P18-E0C Endpoint Residual Fresh640

> 日期：2026-08-10
>
> 状态：已预注册，等待评测
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
