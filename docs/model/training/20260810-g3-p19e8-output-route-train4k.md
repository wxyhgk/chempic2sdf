# G3 P19-E8 Output-route Train4k

> 日期：2026-08-10
>
> 状态：已完成，NO-GO
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

## 实际结果

split 与冻结协议全部通过：train/calibration/holdout 分别为 4,000/640/640 张，三个 split
及已消费数据之间的 molecule/scaffold overlap 全部为 0。训练动作标签为：

- `KEEP_FUSION`：3,896；
- `USE_NEW`：65；
- `USE_OLD`：39。

best 为 step 250，action confidence threshold 为 `0.9352269769`。calibration 从融合
541/640 提高到 544/640，3 rescue、0 harmful；动作输出为 633 KEEP、6 NEW、1 OLD。

checkpoint 与阈值锁定后才首次读取 holdout。一次性结果为：

- 融合 534/640，router 537/640，净增 3 张；
- 4 rescue、1 harmful，动作输出 631 KEEP、5 NEW、4 OLD；
- 普通三布局合计 443/480 -> 444/480；
- stress 91/160 -> 93/160；
- crossed 5/41 -> 6/41，near-nonbonded 12/32 -> 13/32；
- crowded 与 long-bond 均不变。

唯一 harmful 为 `train_009693_original`：fusion/old exact，new non-exact，但 router 以
`0.9497629` 置信度执行 `USE_NEW`。该样本不会用于本轮调 threshold 或重选 checkpoint。

## 结论

正式决策为 `output_route_train4k_no_go`。输出统计具备少量跨 scaffold 泛化信号，但当前
三分类置信度不是安全概率，尚不能安全部署。由于 holdout 已消费，本轮不得在其上提高阈值、
修改 feature 或重选 step；也不运行 P19-E4 eval640 shadow，不接前端。

下一周期不应只扩大相同三分类训练。当前目标把“fusion 与替代路线都正确”和“fusion 正确、
替代路线错误”都压入 KEEP，无法直接学习 harmful 风险。更合理的后续是用全新隔离数据分别
监督每条替代路线相对 fusion 的 `RESCUE/HARMFUL/SAME`，独立校准 harmful 风险，再在全新
holdout 检验零 harmful；P19-E8 calibration/holdout 只能用于排除，不能再参与选模。

## 产物

- best step：250；
- best：`outputs/checkpoints/chemvector-g3-p19e8-output-route-train4k/best.pt`；
- best SHA256：`d762e2a60e9be72b909f17cc3c1adec51ba9a207338d5b59a15089c6d4f4bb3b`；
- train/calibration cache SHA256：`526d43e48512d67c97c06019ef17e2aad5fbae17c364774bfddfab041459f17d`；
- holdout cache SHA256：`8c58b7df7f32224a7a62e269dc9d44268e93fdc2f4bb869e9113aeb662f2d3cd`；
- result SHA256：`fa7b42e2d7420f9e5689af23dbeaf47ae5abd103fdce4b9b635e7f027aa59c76`；
- 总耗时：212.38 秒。
