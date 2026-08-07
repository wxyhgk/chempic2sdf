# G3 P4-E2 Center Field Pilot20k

> 日期：2026-08-07
>
> 状态：已完成，协议通过 / 能力NO-GO
>
> 模型：`chemvector-g3-p4e2-center-field-pilot20k`

## 目标

P4-E1证明冻结train5k/train25k的256并集在pilot/dev仍有不可达GT，selector无法修复。P4-E2从
train25k stride-1 center field继续训练，使真实Atom峰在新scaffold与stress风格中进入候选预算。

## 固定训练协议

- 初始化checkpoint SHA256：
  `9e9d7725e9e9ffecde7bcbe9fbbb095f09204a480b18acbb014d88ef5ac83668`；
- train为P4-E0 pilot20k，5,000个分子、20,000张图，与旧周期全部门集scaffold-disjoint；
- dev为旧P3-E21 dev640，仅用于step0/final能力判断；confirm_v2保持未读；
- batch4、1 epoch、5,000 steps、20,000 exposures；
- AdamW，lr=1e-4、weight decay=0.01、grad clip=1；
- 只训练stride-1 center field与最后一个stride-4 vision block；Atom head、其余vision和base block冻结；
- one-hot center、offset权重8、peak margin权重1/margin1不变；
- fixed-final-only，不保存或选择中间checkpoint，不追加同协议epoch。

## GO/NO-GO门

- dev top128 Atom recall相对step0至少增加0.1个百分点；
- dev top128 exact至少增加5/640；
- known-Count recall回退不超过0.1个百分点；
- 5,000步/20,000 exposure/训练顺序、optimizer范围和冻结state全部严格通过；
- 两次final dev逐项相同；confirm_v2、fresh、外部和sealed_test未读。

全部通过为`center_field_pilot20k_go`，否则NO-GO。无论结果如何保存final checkpoint、提交结果，
并创建与模型名完全一致的annotated tag。只有GO才允许P4-E3用它重建256 union cache。

## 实际结果

固定5,000步、20,000 exposures完整运行，全部协议检查通过：

| dev指标 | step0 | final | 变化 | 门 |
|---|---:|---:|---:|---:|
| top128 matched | 32,098/32,400 | 32,102/32,400 | +4 | recall至少+0.1pp |
| top128 recall | 99.0679% | 99.0802% | +0.0123pp | +0.1pp |
| top128 exact | 517/640 | 518/640 | +1 | +5 |
| known-Count recall | 95.9630% | 96.1821% | +0.2191pp | 回退≤0.1pp |
| known-Count exact | 270/640 | 277/640 | +7 | 诊断 |
| stress known-Count recall | 94.0000% | 94.2099% | +0.2099pp | 诊断 |

训练使known-Count选择多匹配71个Atom、增加7张完整图，说明已有峰的score排序得到改善；但top128
candidate只多4个Atom和1张exact，远低于门槛。也就是说同一个one-hot center/offset/peak-margin
目标继续看20k唯一图，仍没有把被交叉线和拥挤结构压低的真实峰稳定推进候选预算。

最终决策`center_field_pilot20k_no_go`。不追加同协议epoch，不运行P4-E3 union重建，不读取
confirm_v2。下一路线若继续center field，必须改变直接作用于整图regional maxima排序的训练目标，
而不是再次增加相同像素损失的数据量。

## 产物

- checkpoint SHA256：`672e6df940609517ba41d13e84523dab340bcf55d2390e34a0142e173420bda0`；
- audit SHA256：`275570301bc323eeba617e041ab03cfb9ced423dca7614581d9feb479afdbf2a`；
- final metrics SHA256：`de8c9ba9e2c4f58c469797fb67586232bda1b156502de55f06145bd6f1a5f186`；
- 训练耗时614.73秒；
- confirm_v2、fresh、USPTO、MR-TADF和sealed_test均未读取。

结果提交后创建同名annotated tag：`chemvector-g3-p4e2-center-field-pilot20k`。
