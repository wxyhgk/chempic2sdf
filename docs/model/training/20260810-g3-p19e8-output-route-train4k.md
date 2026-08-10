# G3 P19-E8 Output-route Train4k

> 日期：2026-08-10
>
> 状态：已预注册，等待运行
>
> 模型：`chemvector-g3-p19e8-output-route-train4k`

## 目标

P19-E7 在已消费 calibration640 达到 Oracle，但只是容量记忆。本轮用全新 scaffold 隔离数据
训练同一 78 维、27,267 参数 router，并以一次性 holdout 检验是否能安全泛化。

## 数据协议

从 full80k 中排除 pilot20k、P19-E3 old dev、P19-E4 calibration/eval 的全部 molecule 和
scaffold，再按固定 SHA 与唯一 scaffold 选择：

- train：1,000 molecule / 4,000 图；
- calibration：160 / 640；
- holdout：160 / 640。

三个 split 及所有已消费数据的 molecule/scaffold overlap 必须为 0。holdout 只能在 checkpoint
与 action threshold 锁定后读取一次。

## 固定模型与训练

- 冻结旧/新 relation、融合协议与 vision/Atom；
- 固定 P19-E7 的 78 维 feature schema 与 128 hidden MLP；
- 默认 KEEP；train 使用 class-balanced CE；
- batch128、3,000 step、AdamW `1e-3`、weight decay `1e-4`；
- 每 250 step 在 calibration 选择一个 action confidence threshold：低于阈值一律 KEEP；
- checkpoint 依次按 calibration 少 harmful、exact、rescue、action accuracy 选择；
- 禁止读取 P19-E4 eval640、fresh640、confirm、sealed test、USPTO 或 MR-TADF。

## GO/NO-GO

- split 数量正确且所有 overlap 为 0；
- train 的 USE_NEW/USE_OLD 各至少 20；
- calibration 0 harmful、至少 3 rescue；
- 一次性 holdout 0 harmful、至少 3 rescue，整图 exact 严格高于融合；
- 冻结 source state 不变。

无论 GO/NO-GO 都保存 checkpoint、结果并创建同名 annotated Git tag。GO 仍只是内部泛化门，
之后才允许在已消费 P19-E4 eval640 做一次固定 shadow，不得再调模型。
