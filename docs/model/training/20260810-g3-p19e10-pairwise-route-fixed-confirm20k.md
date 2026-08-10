# G3 P19-E10 Pairwise Route Fixed Confirm20k

> 日期：2026-08-10
>
> 状态：已预注册，等待运行
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
