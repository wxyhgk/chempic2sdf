# G3 P4-E2 Center Field Pilot20k

> 日期：2026-08-07
>
> 状态：预注册，待训练
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
