# G3 P19-E11 Predicted Atom Route Bridge2k

> 日期：2026-08-10
>
> 状态：已预注册，等待运行
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
