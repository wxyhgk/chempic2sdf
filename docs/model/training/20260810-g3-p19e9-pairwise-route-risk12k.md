# G3 P19-E9 Pairwise Route Risk12k

> 日期：2026-08-10
>
> 状态：已预注册，等待运行
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
