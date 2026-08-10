# G3 P19-E10 Pairwise Route Fixed Confirm20k

> 日期：2026-08-10
>
> 状态：已完成，GT Atom 条件 GO
>
> 审计：`chemvector-g3-p19e10-pairwise-route-fixed-confirm20k`

## 目标

P19-E9 在独立 holdout 只执行 27 次动作，得到 13 rescue/0 harmful，满足内部泛化门但安全
样本量仍小。本轮不训练模型、不调阈值，冻结 P19-E9 step250、78 维 feature 和
`rescue>=0.5/harmful<=0.02`，在更大的全新 confirm 上检验动作精度和 harmful 上界。

## 数据与隔离

从 full80k 中排除 P19-E9 及此前全部 P19 train/calibration/holdout/dev/eval 的 molecule 和
scaffold，按固定 SHA 选择 5,000 个唯一 scaffold 分子、20,000 张四布局图。

confirm 只运行一次，不用于 checkpoint、模型、feature 或规则选择。禁止读取 fresh640、
sealed test、USPTO 和 MR-TADF。

## 固定协议

- checkpoint SHA256：`e48cd0622e82ad38939632011f35cb82190f14f43f5aac2fcf433a8e92b14278`；
- step：250；
- rescue threshold：0.5；
- harmful threshold：0.02；
- P19-E9 normalization、78 维 feature schema、旧/新 relation 与 fusion 全部冻结；
- 不搜索阈值，不做 early stop，不训练任何参数。

## GO/NO-GO

- 20,000 张、5,000 molecule、5,000 scaffold，和所有已消费数据 overlap 为 0；
- 实际非 KEEP 动作至少 100 次，rescue 至少 50 次；
- rescue/action 至少 35%；
- harmful 为 0，且单侧 95% Clopper-Pearson harmful 上界不超过 2%；
- overall、四个 variant 和所有 stress operation 都是 0 harmful；
- selected exact 严格高于 fusion，所有冻结 source state 不变。

无论 GO/NO-GO 都保存完整 cache、结果、训练时间线记录，并创建同名 annotated Git tag。失败
不得在 confirm 上调规则；成功也只允许进入推理链 shadow A/B，不能直接替换默认前端。

## 实际结果

数据与协议检查全部通过：20,000 张来自 5,000 个分子和 5,000 个唯一 scaffold，与此前所有
P19 数据的 molecule/scaffold overlap 均为 0。checkpoint、step250、78 维 feature 和
`0.5/0.02` 规则逐项锁定，运行中没有模型或规则选择。

整体结果：

- fusion exact：16,719/20,000（83.595%）；
- selected exact：16,883/20,000（84.415%），净增 164，提升 0.82pp；
- 226 次非 KEEP 动作：164 rescue、0 harmful；
- action rescue precision：72.57%；
- 0/226 harmful 的单侧 95% Clopper-Pearson 上界：1.3168%。

四布局均为零 harmful：

| variant | fusion | selected | rescue | harmful |
| --- | ---: | ---: | ---: | ---: |
| coordgen | 4,782 | 4,806 | 24 | 0 |
| original | 4,667 | 4,695 | 28 | 0 |
| rdkit | 4,704 | 4,719 | 15 | 0 |
| stress | 2,566 | 2,663 | 97 | 0 |

四类 stress 也全部零 harmful：crossed `111->134`、crowded `1005->1030`、long-bond
`1079->1086`、near-nonbonded `371->413`。USE_NEW 执行 165 次，126 rescue；USE_OLD 执行
61 次，38 rescue。其余为 61 次 BOTH_WRONG 和 1 次 BOTH_EXACT。

## 关键边界

P19 的冻结评测函数会读取 GT `atom_anchors` 与 `atom_mask`，因此这里证明的是：**给定正确
Atom 集合时，pairwise relation risk router 在全新 scaffold 上可安全提高 skeleton exact**。
它没有证明预测 Atom 条件或真实端到端 SDF 同样 0 harmful，不能直接接默认前端。

## 决策

预注册决策为 `pairwise_route_fixed_confirm20k_go`，13 项检查全部通过；准确表述为 GT Atom
条件 relation 组件 GO。P19-E9 的风险目标和固定规则已经通过规模确认，不再继续扩大同类 GT
Atom 测试或调阈值。

下一步 P19-E11 必须冻结当前模型，改用部署路径的 predicted AtomObject/anchor/mask 构建全部
无序 AtomPair，先做兼容性和路线 Oracle 审计，再决定是否运行 risk router。若 predicted Atom
条件出现协议漂移或 harmful，本组件继续只保留为 Oracle 诊断，不进入端到端推理。

## 产物

- confirm manifest SHA256：`f035a481689c5a03e106e02963fd01da44778696a0442d7d9653468706fd3a63`；
- confirm cache SHA256：`269bff8fc3726824a6363a2ad1bcb943cf388ab2214cd7c2f422ddfef4ee39fe`；
- result SHA256：`1ac1dcf2a86005199fcc0ed99c3a589d2aeb9d9990da577451546cafb5e535d9`；
- 总耗时：664.02 秒。
