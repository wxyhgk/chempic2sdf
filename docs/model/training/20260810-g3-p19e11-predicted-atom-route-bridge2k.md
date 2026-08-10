# G3 P19-E11 Predicted Atom Route Bridge2k

> 日期：2026-08-10
>
> 状态：已完成，NO-GO
>
> 审计：`chemvector-g3-p19e11-predicted-atom-route-bridge2k`

## 目标

P19-E10 证明 risk router 在 GT Atom 条件下可稳定提高 relation exact，但部署路径使用预测
AtomObject。本轮冻结所有模型和规则，把 relation 输入改为实际部署的 vision -> Atom/Bond
proposal -> set head -> predicted Atom mask/anchor/feature/score，验证协议漂移和条件能力。

## 数据

从已消费的 P19-E10 confirm20k 中按固定 SHA 选择 500 个完整四布局分子、2,000 张图。该数据
只用于桥接 screening，不训练、不选模型、不调阈值，不消耗剩余约 2,900 个新 scaffold。

## 固定协议

- P19-E9 checkpoint、step250、78 维 normalization 与 `0.5/0.02` 规则冻结；
- 旧/新 relation、vision、Atom、Bond、set head 与部署 Atom threshold 全部冻结；
- 枚举预测 Atom mask 中的全部无序 AtomPair，不做距离或近邻裁剪；
- router 只读预测 route 输出统计，不接收 GT anchor、mask、bond 或 matching；
- GT 仅在前向结束后用于 6px Hungarian Atom matching 和 exact 指标。

同时报告：

1. Atom exact 与 Atom recall；
2. 不考虑 Atom 是否完整的 pair exact；
3. `Atom exact AND pair exact` 的 skeleton exact；
4. fusion/new/old Oracle 与 risk router rescue/harmful。

## GO/NO-GO

- 2,000 图/500 molecule/500 scaffold 协议完整，未读取新 scaffold；
- 78 维 feature 全部有限，至少 99% 图可构建 AtomPair feature；
- Atom exact 支持至少 500 张；
- Atom exact 条件 fusion exact 至少 70%，三路线 Oracle headroom 至少 5；
- 固定 router 执行至少 5 次且不超过 5% 图片；
- overall 至少 3 rescue、0 harmful，Atom exact 子集也是 0 harmful；
- selected skeleton exact 严格高于 fusion，冻结 source state 不变。

GO 才允许在剩余全新 scaffold 上做 predicted-Atom fixed confirm；NO-GO 则保留 GT Atom relation
组件，不得接端到端推理，也不得在本 screen 上调 P19-E9 阈值。

## 实际结果

部署 predicted Atom 协议已完整打通，2,000/2,000 图可构建 78 维 feature，所有数值有限，
router eligible 为 100%。Atom 层结果为：

- 单原子 matched recall：59,423/61,036 = 97.36%；
- 整图 Atom exact：1,186/2,000 = 59.30%；
- Bond representable：1,412/2,000；
- coordgen/original/rdkit/stress Atom exact：358/336/339/153；
- stress 中 crossed Atom exact 仅 1/126，near-nonbonded 仅 11/128。

关系路线结果：

| 路线 | pair exact | skeleton exact | 相对 fusion rescue | break |
| --- | ---: | ---: | ---: | ---: |
| fusion | 1,335 | 1,101 | - | - |
| new | 1,247 | 1,036 | 7 | 72 |
| old | 1,359 | 1,135 | 41 | 7 |
| Oracle | - | 1,149 | 48 headroom | - |

Atom exact 条件下 fusion 为 1,101/1,186 = 92.83%，说明 pair relation 本身仍强；主要端到端
损失来自 Atom 集合不完整。预测 Atom 分布也使路线优劣反转：旧 relation 明显优于 fusion，
新 relation 大幅变差。

固定 P19-E9 router 执行 29 次：

- 6 rescue、3 harmful，selected `1101->1104`；
- USE_NEW 22 次：3 rescue、3 harmful、16 BOTH_WRONG；
- USE_OLD 7 次：3 rescue、0 harmful、4 BOTH_WRONG；
- harmful 分布在 coordgen/original/rdkit 各 1，stress 为 0；
- action rate 1.45%，feature 95% 标准化绝对值 1.94，但最大值 51.72。

## 工程复核

正式结果前有三次无指标工程停止，失败目录均保留：

1. 旧 `AtomBondSceneCollator` 在坐标重叠样本触发 ChemSVG v0.4 collision；改用 P19 原生
   `G3StructuralCollator`；
2. 旧 `stack_targets` 强制要求无关 OCR 字段；改为五个结构字段白名单；
3. 全量 cache 后对 inference tensor 原地写回失败；改为非原地 `torch.where`。

三次均未生成 result 或用于修改数据、模型、规则和门槛；每个修复单独提交并增加回归测试。

## 决策

正式决策为 `predicted_atom_route_bridge2k_no_go`。协议、支持量、Oracle 和净增门均通过，但
overall 与 Atom exact 子集都出现 3 harmful，固定 GT Atom router 不允许读取剩余新 scaffold，
也不得接推理链。screen 后观察到的 OLD-only 3 rescue/0 harmful 不能在同一数据上宣称成功。

下一步 P19-E12 应使用已消费 P19-E10 confirm20k 生成 predicted-Atom 路线 cache，把本 screen
只放入训练部分，从其余分子建立 calibration/internal holdout，重新监督 NEW/OLD 相对 fusion
的四结果风险。剩余约 2,900 个新 scaffold 保留为最终一次 predicted-Atom confirm。同时，
crossed 的 1/126 Atom exact 说明 relation 支线不能替代 Atom 集合主线。

## 产物

- screen manifest SHA256：`107f700e194c166927e1f49473417ee77f3448733b798708c8f07bd96017d3b1`；
- cache SHA256：`12316d090da19f139e0948b44a891748e05074c399a65f330ed6310aaf7268c4`；
- result SHA256：`845a97d4cec53738f1adb572ea52e33a479c694264fdc1a91b819a0c1e3aa3d5`；
- 最终运行耗时：113.46 秒。
