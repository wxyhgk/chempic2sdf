# G3 P19-E9 Pairwise Route Risk12k

> 日期：2026-08-10
>
> 状态：已完成，GO
>
> 模型：`chemvector-g3-p19e9-pairwise-route-risk12k`

## 目标

P19-E8 的三动作分类在全新 holdout 得到 4 rescue/1 harmful。它证明 78 维输出统计存在少量
泛化信号，也暴露出目标错配：所有 fusion exact 图都被压成 KEEP，无法区分替代路线同样正确
还是会破坏正确结果。本轮对 NEW/OLD 分别预测四种相对结果：`BOTH_WRONG / BOTH_EXACT /
RESCUE / HARMFUL`，显式拆开收益和风险。

## 数据协议

从 full80k 中排除 pilot20k、P19-E3 dev、P19-E4 calibration/eval、P19-E8 train/calibration/
holdout 的全部 molecule 与 scaffold，再按固定 SHA 和唯一 scaffold 选择：

- train：2,000 molecule / 8,000 图；
- calibration：500 molecule / 2,000 图；
- holdout：500 molecule / 2,000 图。

三个 split 与所有已消费数据的 molecule/scaffold overlap 必须为 0。holdout 的图片和标签只在
checkpoint 与风险规则锁定后首次读取。

## 固定模型与训练

- 冻结旧/新 relation、融合协议与 vision/Atom；
- 复用固定 78 维输出 feature，128 hidden 共享 MLP，两个四分类 outcome head；
- 每条路线分别使用 class-balanced CE，batch256、3,000 step、AdamW `1e-3`；
- 每 250 step 只在 calibration 上搜索预注册的 5x5 固定规则网格；
- 只有 `P(RESCUE)` 达到门槛且 `P(HARMFUL)` 不超过门槛才允许离开 fusion；
- 多条路线合格时选择 `P(RESCUE)-P(HARMFUL)` 更高者；否则保持 fusion；
- 禁止读取 fresh640、confirm、sealed test、USPTO 或 MR-TADF。

固定规则网格：

- rescue threshold：`0.50/0.60/0.70/0.80/0.90`；
- harmful threshold：`0.20/0.10/0.05/0.02/0.01`。

## GO/NO-GO

- split 数量正确且所有 overlap 为 0；
- train 中 NEW/OLD 的 RESCUE 与 HARMFUL 各至少 20；
- calibration 0 harmful、至少 5 rescue；
- 一次性 holdout 0 harmful、至少 5 rescue，exact 严格高于 fusion；
- 冻结 source state 不变。

无论 GO/NO-GO 都保存 checkpoint、结果并创建同名 annotated Git tag。P19-E8 holdout 只用于
排除，不允许参与本轮特征、checkpoint、规则或阈值选择。

## 实际结果

8k train 的四结果标签数量为：

| 路线 | BOTH_WRONG | BOTH_EXACT | RESCUE | HARMFUL |
| --- | ---: | ---: | ---: | ---: |
| USE_NEW | 1,208 | 6,518 | 120 | 154 |
| USE_OLD | 1,217 | 6,414 | 111 | 258 |

两条路线的 RESCUE/HARMFUL 均远超过 20，风险监督不是空标签。best 为 step 250，固定规则为
`rescue_threshold=0.5`、`harmful_threshold=0.02`。

Calibration 2,000 张：

- fusion 1,665，selected 1,683；
- 18 rescue、0 harmful；
- 1,973 KEEP、21 USE_NEW、6 USE_OLD；
- 普通三布局分别增加 4/4/2，stress 增加 8。

锁定 checkpoint 和规则后才首次读取 holdout。一次性 2,000 张结果：

- fusion 1,653，selected 1,666，净增 13；
- 13 rescue、0 harmful；
- 1,973 KEEP、17 USE_NEW、10 USE_OLD；
- coordgen/original/rdkit/stress 分别增加 1/3/3/6；
- crossed/crowded/long/near 分别增加 2/2/1/1，全部 0 harmful。

27 次实际切换中，13 次为 RESCUE，14 次为 BOTH_WRONG，没有 BOTH_EXACT 或 HARMFUL。三路线
Oracle 为 1,699/2,000，相对 fusion 有 46 张 headroom，当前安全取回 13 张。

## 决策

正式决策为 `pairwise_route_risk12k_go`，全部预注册门通过。相对 P19-E8 的 4 rescue/1 harmful，
本轮直接建模 harmful 风险后，在更大的独立 holdout 得到 13/0，证明目标函数修改有效。

这仍不是生产部署结论：holdout 只有 27 次非 KEEP 动作，零 harmful 的统计证据有限；14 次
BOTH_WRONG 也说明 utility precision 仍可提高。下一步冻结 step 250、78 维 feature 和
`0.5/0.02` 规则，在剩余全新 scaffold 上做更大规模 fixed confirm。confirm 不得选 step、
改阈值或回训；当前 P19-E9 holdout 从此只用于排除。

## 产物

- best：`outputs/checkpoints/chemvector-g3-p19e9-pairwise-route-risk12k/best.pt`；
- best SHA256：`e48cd0622e82ad38939632011f35cb82190f14f43f5aac2fcf433a8e92b14278`；
- train/calibration cache SHA256：`e33eff9921d6cc593d4a99f1b7a5a42e60aa774c0968d0db622bfaf56f34f067`；
- holdout cache SHA256：`adbc5fb7693e8b52f91164d6f7578e17b70e5e8c02569cc3177ce77d8aa65b1a`；
- result SHA256：`6e532bf10b68695c1072949586d5d95297e7bdc0a4afccd252724251941acdb9`；
- 参数量：27,912；
- 总耗时：434.46 秒。

## 后验边界澄清

P19-E10 复核确认 P19-E9 的 route cache 使用 GT `atom_anchors` 和 `atom_mask`。因此本页全部
指标都是 GT Atom 条件 relation 结果，不是 predicted Atom 或端到端 SDF 指标。该澄清不改变
P19-E9 的预注册 GO，但限制了适用范围；后续必须经过 predicted Atom bridge 才能接推理链。
